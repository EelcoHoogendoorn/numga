"""Every extensor carries an immutable `GAType` describing its output-first `SubSpace` axes,
per-axis `Orientation`s (`intrinsic` or `extrinsic`), certified `TraitSet` facts, and coupled field
site counts within a single `Algebra`. Nullary values (`arity == 0`) have a single output axis,
unary maps (`arity == 1`) pair an output axis with one input axis, and multilinear maps or forms add
further input axes in argument order. When axes range over discrete sites as well as blades,
`fields` records `(axis, n_sites)` in axis order, placing site axes between leading batch axes and
trailing blade-coefficient axes in the underlying array.

Exact equality (`==`) and hashing include exact `SubSpace` layouts, per-axis orientations, traits,
and fields so two extensors have equal types only when their kernels share identical storage
conventions. Semantic refinement (`<=`, `<`, `refines`, `strictly_refines`) instead compares
unordered blade supports and trait entailment independently of basis ordering and blade signs,
forming a preorder over concrete layouts used by `GATypeDispatch`.

A nullary `GAType` stands directly as the exact identity map on its space: calling `Boundary(x)`
restricts an argument to `Boundary`, arithmetic and products (`Vector + shear`,
`Bivector.reverse()`, `Point & Point`) construct symbolic `Extensor` maps in the algebra's exact
context, and `.dual()` or `.dual_inverse()` builds the duality map on that space. Attributes that
belong only to concrete values (`kernel`, `context`, `shape`, `ndim`, `dtype`) are not forwarded.

Whereas operations on `gatype` itself build maps, `gatype.derive` computes related `GAType`s:
stripping explicit traits (`structural`) or field axes (`plain`), flipping or replacing
orientations, taking complementary blade supports (`dual`), restricting output blades to
subalgebras or generator sets (`within`, `touching`, `without`, `containing`), and deducing
quadratic self-product supports. Domain-specific inspections live on cached query namespaces
(`axis_roles`, `form`, `over_sites`, `metric`, `self_products`). Orientation rules across types and
maps are set out in `docs/orientation.md`, and kernel layouts, binding plans, and trait propagation
in `docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cached_property, lru_cache
from numbers import Number
from types import NotImplementedType
from typing import TYPE_CHECKING

import numpy as np

import numga
from numga.algebra.self_product import grade_transform_signs, symmetric_product_support
from numga.subspace import SubSpace
from numga.units import (
    Unit,
    UnitGrid,
    _dual_permutation,
    _trace_axes,
    coerce_unit_grid,
    dim,
    invert_map_units,
    remap_grid_subspaces,
    validate_reverse_product_one,
)

from .errors import GATypeError
from .metric import Metric
from .orientation import Orientation
from .queries import AxisRoles, Form, OverSites, SelfProducts
from .traits import (
    EMPTY_TRAITS,
    ProductFact,
    ProductRelation,
    ProductResult,
    ReverseProductOne,
    SelfProduct,
    Trait,
    TraitSet,
    Versor,
    VersorProduct,
    _trait_closure,
    normalize_explicit_traits,
    structurally_implied_traits,
    validate_traits_for_subspaces,
)

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.extensor import Extensor


# Attributes owned by concrete Extensor values rather than symbolic identity maps on GATypes.
_VALUE_DATA = frozenset({"kernel", "context", "gatype", "shape", "ndim", "dtype", "subspace"})

_SELF_PRODUCT_TRANSFORMS = (
    "identity", "reverse", "clifford_conjugate", "scalar_negation",
    "pseudoscalar_negation", "involute",
)


class GAType:
    """Output-first `SubSpace` axes, per-axis `Orientation`s, certified `TraitSet` facts, and
    coupled field site counts define the static type of an extensor."""

    __slots__ = ("_subspaces", "_orientations", "_traits", "_fields", "_units", "_algebra", "_hash", "__dict__")
    # Disable NumPy ufunc coercion so array * gatype uses GAType.__rmul__ instead of building an object array.
    __array_ufunc__ = None

    def __init__(
        self,
        subspaces: Iterable[SubSpace],
        orientations: Iterable[Orientation],
        traits: TraitSet = EMPTY_TRAITS,
        fields: tuple = (),
        units: UnitGrid | None = None,
    ) -> None:
        if isinstance(subspaces, SubSpace):
            raise TypeError("GAType subspaces must be an output-first iterable")
        try:
            normalized_subspaces = tuple(subspaces)
        except TypeError as error:
            raise TypeError(
                "GAType subspaces must be an output-first iterable"
            ) from error
        if not normalized_subspaces:
            raise ValueError("a GAType requires at least an output SubSpace")
        if not all(isinstance(axis, SubSpace) for axis in normalized_subspaces):
            raise TypeError("every GAType axis must be a SubSpace")

        algebra = normalized_subspaces[0].algebra
        if any(axis.algebra is not algebra for axis in normalized_subspaces[1:]):
            raise ValueError("every GAType axis must belong to the same algebra")

        blade_shape = tuple(len(axis) for axis in normalized_subspaces)
        if units is None:
            resolved_units = UnitGrid.unchecked(blade_shape)
        elif not isinstance(units, UnitGrid):
            raise TypeError(f"GAType units must be a UnitGrid; got {units!r}")
        elif units.shape != blade_shape:
            raise GATypeError(f"UnitGrid shape {units.shape} does not match type shape {blade_shape}")
        else:
            resolved_units = units

        normalized_traits = normalize_explicit_traits(
            normalized_subspaces,
            traits,
        )
        arity = len(normalized_subspaces) - 1
        for trait in normalized_traits:
            trait.validate_arity(arity)
        validate_traits_for_subspaces(
            normalized_subspaces,
            normalized_traits,
        )
        normalized_fields = _normalize_fields(fields, len(normalized_subspaces))
        normalized_orientations = tuple(orientations)
        if len(normalized_orientations) != len(normalized_subspaces) or not all(
            isinstance(orientation, Orientation) for orientation in normalized_orientations
        ):
            raise ValueError("a GAType needs one Orientation per axis")
        object.__setattr__(self, "_subspaces", normalized_subspaces)
        object.__setattr__(self, "_orientations", normalized_orientations)
        object.__setattr__(self, "_traits", normalized_traits)
        object.__setattr__(self, "_fields", normalized_fields)
        object.__setattr__(self, "_units", resolved_units)
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(
            self,
            "_hash",
            hash((type(self), normalized_subspaces, normalized_orientations, normalized_traits, normalized_fields, resolved_units)),
        )
        if not resolved_units.is_unchecked and any(
            isinstance(trait, ProductFact) and trait.result is ProductResult.ONE
            for trait in normalized_traits
        ):
            validate_reverse_product_one(self)

    @property
    def subspaces(self) -> tuple[SubSpace, ...]:
        return self._subspaces

    @property
    def axes(self) -> tuple[SubSpace, ...]:
        return self.subspaces

    @property
    def orientations(self) -> tuple[Orientation, ...]:
        """Per-axis orientations are ordered output first to match `subspaces`."""
        return self._orientations

    @property
    def orientation(self) -> Orientation:
        """Axis `0` fixes the output orientation of the extensor."""
        return self._orientations[0]

    @property
    def input_orientations(self) -> tuple[Orientation, ...]:
        return self._orientations[1:]

    @cached_property
    def structural_shape(self) -> tuple[int, ...]:
        """Trailing kernel dimensions place coupled site axes before per-axis blade dimensions."""
        return self.site_shape + tuple(len(axis) for axis in self.subspaces)

    @property
    def fields(self) -> tuple:
        """Each `(axis, n_sites)` pair marks an output (`0`) or input (`1..arity`) axis ranging over
        discrete sites."""
        return self._fields

    @cached_property
    def site_shape(self) -> tuple[int, ...]:
        return tuple(sites for _, sites in self._fields)

    @cached_property
    def has_fields(self) -> bool:
        return bool(self._fields)

    @cached_property
    def has_input_fields(self) -> bool:
        """Input axes carrying sites contract by summing across paired sites during binding."""
        return any(axis for axis, _ in self._fields)

    @property
    def traits(self) -> TraitSet:
        return self._traits

    @property
    def has_units(self) -> bool:
        """Whether this type carries checked physical units (false when unchecked)."""
        return not self._units.is_unchecked

    @property
    def units(self) -> UnitGrid:
        """Per-entry physical units (`is_unchecked` when unit checking is off)."""
        return self._units

    @property
    def unit(self) -> Unit:
        """Single physical unit shared by every nonzero entry; raises `GATypeError` if unchecked,
        structurally zero, or non-uniform."""
        if self._units.is_unchecked:
            raise GATypeError(f"{self.signature} is unchecked and has no unit")
        if self._units.is_zero:
            raise GATypeError(f"{self.signature} is structurally zero and has no unit")
        if self._units.uniform is None:
            raise GATypeError(
                f"{self.signature} has multiple blade units ({self._units.format_for(self.subspaces)}); inspect .units"
            )
        return self._units.uniform

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    def output_subspace(self) -> SubSpace:
        return self.subspaces[0]

    @property
    def input_subspaces(self) -> tuple[SubSpace, ...]:
        return self.subspaces[1:]

    @property
    def arity(self) -> int:
        return len(self.subspaces) - 1

    @cached_property
    def is_scalar(self) -> bool:
        """Nullary scalar types have output support contained in grade `0`, including the empty
        support."""
        return self <= self.algebra.subspace.scalar()

    @cached_property
    def is_reoriented_scalar(self) -> bool:
        """Reoriented scalars store their single coordinate against basis blade `-1` rather than `+1`."""
        return self.is_scalar and self.output_subspace.signs == (-1,)

    @cached_property
    def is_empty(self) -> bool:
        """Empty nullary types have zero basis blades and represent identically zero values."""
        return self <= self.algebra.subspace.empty()

    @cached_property
    def derive(self) -> Derivations:
        """Type-level derivations (`structural`, `dual`, `within`, `squared`) live on `derive`,
        keeping operations on the `GAType` itself free to construct identity maps."""
        return Derivations(self)

    @cached_property
    def axis_roles(self) -> AxisRoles:
        """Axis-role queries determine indexed axes, contraction pairs, and input domain narrowing."""
        return AxisRoles(self)

    @cached_property
    def form(self) -> Form:
        """Form queries test whether a scalar-valued binary map is square or matches a right-hand side."""
        return Form(self)

    @cached_property
    def over_sites(self) -> OverSites:
        """Site-coupled queries inspect maps and forms as block operators over coupled field sites."""
        return OverSites(self)

    @cached_property
    def metric(self) -> Metric:
        """Metric queries build and classify the canonical symmetric-reverse form on this space."""
        return Metric(self)

    @cached_property
    def self_products(self) -> SelfProducts:
        """Self-product queries inspect certified `ProductFact` and relational closure traits."""
        return SelfProducts(self)

    @lru_cache(maxsize=None)
    def grade_transform_is_identity(self, transform: str) -> bool:
        """A grade transform acts as the identity on this space when every basis blade of
        `output_subspace` receives sign `+1`."""
        return bool(np.all(grade_transform_signs(self.algebra, transform, self.output_subspace.masks) == 1))

    @cached_property
    def is_study(self) -> bool:
        """Generalized Study numbers combine a scalar part with a nonscalar part whose square is scalar."""
        return self.derive.symmetric_scalar_negation.is_scalar

    @cached_property
    def is_scalar_bivector(self) -> bool:
        """Scalar-bivector nullary types confine their support to grades `0` and `2`."""
        spaces = self.algebra.subspace
        return self <= spaces.scalar() + spaces.bivector()

    @cached_property
    def is_bivector(self) -> bool:
        """Bivector nullary types have support contained in grade `2` with either orientation."""
        return self <= self.algebra.subspace.bivector()

    @cached_property
    def is_even(self) -> bool:
        """Even nullary types have support contained in even grades with either orientation."""
        return self <= self.algebra.subspace.even()

    @cached_property
    def is_rotor(self) -> bool:
        """Rotor types refine `algebra.gatype.rotor()` (extrinsic even versors with unit reverse product)."""
        return self <= self.algebra.gatype.rotor()

    @cached_property
    def is_nilpotent(self) -> bool:
        """Nilpotent types square identically to zero from blade support alone."""
        return self.derive.squared.is_empty

    @cached_property
    def has_null_wedge(self) -> bool:
        """The grade-4 component of `squared` itself squares to zero when no pair of planes wedges
        non-trivially."""
        algebra = self.algebra
        wedge = self.derive.squared.output_subspace.intersection(algebra.subspace.k_vector(4))
        # A component of a square carries extrinsic orientation.
        return algebra.gatype(wedge, orientation=Orientation.extrinsic).derive.squared.is_empty

    @cached_property
    def is_form(self) -> bool:
        """Bilinear forms have `arity == 2` and scalar output support."""
        return self.arity == 2 and self.output_subspace.same_support(self.algebra.subspace.scalar())

    @cached_property
    def is_exterior_map(self) -> bool:
        """Exterior maps act unary between grade-1 vectors or grade-`(n - 1)` antivectors on each
        side, covering vector endomorphisms, antivector endomorphisms, and duality maps between them."""
        if self.arity != 1:
            return False
        algebra = self.algebra
        generators = ({1}, {algebra.dimension - 1})
        inputs = {algebra.grade(mask) for mask in self.input_subspaces[0].masks}
        outputs = {algebra.grade(mask) for mask in self.output_subspace.masks}
        return inputs in generators and outputs in generators

    @property
    def maps_equal_dimensions(self) -> bool:
        """Square unary maps match output and input subspace dimensions, enabling `inverse` and `solve`."""
        return self.arity == 1 and len(self.subspaces[0]) == len(self.subspaces[1])

    @property
    def is_endomorphism(self) -> bool:
        """Endomorphisms map a space to itself with matching blade support and orientation, allowing
        powers, traces, and standard eigenvalue decompositions."""
        return (
            self.arity == 1
            and self.subspaces[0].same_support(self.subspaces[1])
            and self.orientations[0] is self.orientations[1]
        )

    @property
    def is_correlation(self) -> bool:
        """Duality maps carry a space to its oppositely oriented complement (such as points to planes
        or rates to forques); regressive pairing with the input produces an extrinsic scalar, while
        eigendecomposition requires a second duality map as metric."""
        if self.arity != 1:
            return False
        complements = {self.algebra.complement(mask) for mask in self.subspaces[1].masks}
        return set(self.subspaces[0].masks) == complements and self.orientations[0] is not self.orientations[1]

    @lru_cache(maxsize=None)
    def reduces_to_scalar(self, steps: int) -> bool:
        """At most `steps` successive quadratic self-products suffice to reach scalar support on
        nullary types whose self-product chain terminates in grade `0`."""
        if self.arity:
            return False
        if steps <= 0:
            return self.is_scalar
        return any(
            self.derive.self_product(transform).reduces_to_scalar(steps - 1)
            for transform in _SELF_PRODUCT_TRANSFORMS
        )

    @cached_property
    def inverse_traits(self) -> TraitSet:
        """Inverting an invertible nullary value preserves `Versor` and promotes scalar self-products
        to `NONZERO` (keeping `ONE` unchanged)."""
        if self.arity:
            return EMPTY_TRAITS
        traits = [Versor] if self.entails(Versor) else []
        # Implied facts count too: a Versor's reverse product is a scalar, and inverting makes it nonzero.
        for fact in _trait_closure(self.effective_traits.traits):
            if (
                isinstance(fact, ProductFact)
                and fact.self_product.product == "geometric_product"
                and fact.self_product.transform in _SELF_PRODUCT_TRANSFORMS
                and fact.result is not ProductResult.ZERO
            ):
                result = (
                    ProductResult.ONE if fact.result is ProductResult.ONE
                    else ProductResult.NONZERO
                )
                traits.append(ProductFact(fact.self_product, result))
        return TraitSet(traits)

    @cached_property
    def normalized_traits(self) -> TraitSet:
        """Reverse-product normalization certifies `ReverseProductOne`, and also `Versor` on scalars,
        vectors, or even elements below six dimensions."""
        spaces = self.algebra.subspace
        establishes_versor = (
            self.entails(Versor)
            or self <= spaces.scalar()
            or self <= spaces.vector()
            or (self.algebra.dimension < 6 and self <= spaces.even())
        )
        return TraitSet(
            (ReverseProductOne, Versor) if establishes_versor
            else (ReverseProductOne,)
        )

    @property
    def representation_key(self) -> tuple:
        return self.subspaces, self.traits

    @property
    def semantic_key(self) -> tuple:
        return (
            self.algebra,
            tuple(axis.support_key for axis in self.subspaces),
            self.traits,
        )

    def entails(self, required: Trait | TraitSet | Iterable[Trait]) -> bool:
        """Entailment checks both explicit `traits` and facts structurally implied by `subspaces`."""
        return self.effective_traits.entails(required)

    @cached_property
    def effective_traits(self) -> TraitSet:
        """Combining explicit `traits` with `structurally_implied_traits` gives all certified facts."""
        structural = structurally_implied_traits(self.subspaces)
        return TraitSet(self.traits.traits + structural.traits)

    def refines(self, other: object) -> bool:
        """Semantic refinement (`<=`) checks unordered blade-support inclusion on every axis together
        with trait entailment, forming a layout-independent preorder."""
        other = _comparison_gatype(other, self)
        return (
            other is not None
            and self.algebra is other.algebra
            and self.arity == other.arity
            and all(
                actual.support_is_subset_of(pattern)
                for actual, pattern in zip(self.subspaces, other.subspaces)
            )
            and self.entails(other.traits)
        )

    def strictly_refines(self, other: object) -> bool:
        """Strict refinement (`<`) holds when `self` refines `other` without mutual refinement."""
        other = _comparison_gatype(other, self)
        return (
            other is not None
            and self.refines(other)
            and not other.refines(self)
        )

    def overlaps(self, other: object) -> bool:
        """Two types overlap when their traits are compatible and every axis shares at least one
        canonical blade mask."""

        other = _comparison_gatype(other, self)
        return (
            other is not None
            and self.algebra is other.algebra
            and self.arity == other.arity
            and self.traits.is_compatible_with(other.traits)
            and all(
                bool(set(left.masks).intersection(right.masks))
                for left, right in zip(self.subspaces, other.subspaces)
            )
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("GAType is immutable")

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, GAType):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.subspaces == other.subspaces
            and self.orientations == other.orientations
            and self.traits == other.traits
            and self.fields == other.fields
            and self._units == other._units
        )

    def __hash__(self) -> int:
        return self._hash

    def __le__(self, other: object) -> bool | NotImplementedType:
        other = _comparison_gatype(other, self)
        if other is None:
            return NotImplemented
        return self.refines(other)

    def __lt__(self, other: object) -> bool | NotImplementedType:
        other = _comparison_gatype(other, self)
        if other is None:
            return NotImplemented
        return self.strictly_refines(other)

    def __ge__(self, other: object) -> bool | NotImplementedType:
        other = _comparison_gatype(other, self)
        if other is None:
            return NotImplemented
        return other.refines(self)

    def __gt__(self, other: object) -> bool | NotImplementedType:
        other = _comparison_gatype(other, self)
        if other is None:
            return NotImplemented
        return other.strictly_refines(self)

    def _identity(self) -> Extensor:
        if self.arity:
            raise TypeError("only a nullary GAType promotes to an identity map")
        return self.algebra.operator.identity(self)

    def __call__(self, *operands: object) -> Extensor:
        """A nullary type is its identity map, applied or composed like any other map."""
        return self._identity()(*operands)

    def __getattr__(self, name: str) -> object:
        """A nullary type is its identity map: Extensor attributes it does not define itself act on
        that map, so `Bivector.reverse()` is the reversion and `Vector.scalar_product(Vector)` the
        metric. What only a value has, its coefficients and their backend, a type does not."""
        if name.startswith("_") or name in _VALUE_DATA or not hasattr(numga.extensor.Extensor, name) or self.arity:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._identity(), name)

    # A nullary type in linear arithmetic is its identity map, promoted as any other operand is.
    def __add__(self, other: object) -> Extensor | NotImplementedType:
        if numga.extensor.operands.kind(other) is numga.extensor.operands.Kind.FOREIGN:
            return NotImplemented
        return self._identity().__add__(other)

    def __radd__(self, other: object) -> Extensor | NotImplementedType:
        if numga.extensor.operands.kind(other) is numga.extensor.operands.Kind.FOREIGN:
            return NotImplemented
        return self._identity().__radd__(other)

    def __sub__(self, other: object) -> Extensor | NotImplementedType:
        if numga.extensor.operands.kind(other) is numga.extensor.operands.Kind.FOREIGN:
            return NotImplemented
        return self._identity().__sub__(other)

    def __rsub__(self, other: object) -> Extensor | NotImplementedType:
        if numga.extensor.operands.kind(other) is numga.extensor.operands.Kind.FOREIGN:
            return NotImplemented
        return self._identity().__rsub__(other)

    def __neg__(self) -> Extensor:
        return -self._identity()

    def __invert__(self) -> Extensor:
        return ~self._identity()

    def __truediv__(self, other: object) -> Extensor | NotImplementedType:
        if numga.extensor.operands.kind(other) is numga.extensor.operands.Kind.FOREIGN:
            return NotImplemented
        return self._identity().__truediv__(other)

    # A type is its identity map; numbers and backend arrays scale that map.
    def __mul__(self, other: object) -> Extensor | NotImplementedType:
        if numga.expression.is_expression_operand(other):
            return numga.expression.geometric_product(self, other)
        if numga.extensor.operands.kind(other) is numga.extensor.operands.Kind.FOREIGN:
            return NotImplemented
        return self._identity().__mul__(other)

    def __rmul__(self, other: object) -> Extensor | NotImplementedType:
        if numga.expression.is_expression_operand(other):
            return numga.expression.geometric_product(other, self)
        if numga.extensor.operands.kind(other) is numga.extensor.operands.Kind.FOREIGN:
            return NotImplemented
        return self._identity().__rmul__(other)

    def dual(self) -> Extensor:
        """Right-complement duality maps elements of this nullary space to its oppositely oriented
        complement."""
        return self.algebra.operator.dual(self)

    def dual_inverse(self) -> Extensor:
        """Inverse duality constructs the exact map undoing `dual()` on elements of this nullary
        space."""
        return self.algebra.operator.dual_inverse(self)

    def wedge(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.wedge(self, other)

    def inner(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.inner(self, other)

    def dot(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.dot(self, other)

    def commutator(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.commutator(self, other)

    def anticommutator(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.anticommutator(self, other)

    def regressive(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.regressive(self, other)

    def transform(self, passenger: GAType | Extensor) -> Extensor:
        """Versor action `self >> passenger` constructs the symbolic sandwich map of elements in
        this space acting on `passenger`."""
        return numga.expression.transform(self, passenger)

    def __xor__(self, other: object) -> Extensor | NotImplementedType:
        return self.wedge(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __and__(self, other: object) -> Extensor | NotImplementedType:
        return self.regressive(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __or__(self, other: object) -> Extensor | NotImplementedType:
        return self.inner(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rxor__(self, other: object) -> Extensor | NotImplementedType:
        return numga.expression.wedge(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rand__(self, other: object) -> Extensor | NotImplementedType:
        return numga.expression.regressive(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __ror__(self, other: object) -> Extensor | NotImplementedType:
        return numga.expression.inner(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rshift__(self, other: GAType | Extensor) -> Extensor:
        return self.transform(other)

    def __lshift__(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.reverse_transform(self, other)

    def __reduce__(self) -> tuple:
        return type(self), (self.subspaces, self.orientations, self.traits, self.fields, self._units)

    @property
    def signature(self) -> str:
        """Human-readable signature formatting lists output and inputs by orientation and subspace
        name (`intrinsic bivector <- extrinsic bivector`), followed by any traits after `|` and any
        non-dimensionless unit after `,`."""
        sites = dict(self.fields)
        output, *inputs = (
            f"{orientation.value} {space.type_name}"
            + (f"[{sites[axis]}]" if axis in sites else "")
            for axis, (space, orientation) in enumerate(zip(self.subspaces, self.orientations))
        )
        text = f"{output} <- {', '.join(inputs)}" if inputs else output
        if self.traits:
            text = f"{text} | {', '.join(trait.name for trait in self.traits)}"
        suffix = self._units.format_for(self.subspaces)
        if suffix:
            return f"{text}, {suffix}"
        return text

    def __repr__(self) -> str:
        return f"GAType({self.signature})"


def _generator_mask(algebra: Algebra, generators: str) -> int:
    masks = algebra.subspace(generators).masks
    if any(mask.bit_count() != 1 for mask in masks):
        raise ValueError(f"name basis vectors, not blades: {generators!r}")
    span = 0
    for mask in masks:
        span |= mask
    return span


def _part_mask(algebra: Algebra, part: Algebra | str) -> int:
    if isinstance(part, str):
        return _generator_mask(algebra, part)
    span = 0
    for name, square in zip(part.basis_names, part.signature):
        if name not in algebra.basis_names or algebra.signature[algebra.basis_names.index(name)] != square:
            raise GATypeError(f"{part} is no subalgebra of {algebra}: its generator {name} is not one of "
                              f"{algebra}'s with the same square")
        span |= 1 << algebra.basis_names.index(name)
    return span


def _comparison_gatype(value: object, like: GAType) -> GAType | None:
    # Bare subspaces compare by support alone by adopting the orientation of `like`.
    if isinstance(value, GAType):
        return value
    if isinstance(value, SubSpace):
        return value.algebra.gatype.oriented((value,), (like.orientation,))
    return None


def _normalize_fields(fields: Iterable, axes: int) -> tuple:
    normalized = tuple(sorted((int(axis), int(sites)) for axis, sites in fields))
    if any(axis < 0 or axis >= axes or sites < 1 for axis, sites in normalized):
        raise ValueError(f"field axes must lie in [0, {axes}) and hold at least one site; got {normalized}")
    if len({axis for axis, _ in normalized}) != len(normalized):
        raise ValueError(f"an axis holds at most one site axis; got {normalized}")
    return normalized


class Derivations:
    """Type-level derivations on `gatype.derive` construct related `GAType`s without building
    symbolic `Extensor` maps."""

    __slots__ = ("_gatype", "__dict__")

    def __init__(self, gatype: GAType) -> None:
        self._gatype = gatype

    @cached_property
    def structural(self) -> GAType:
        """Stripping explicit `traits` leaves only the subspaces, orientations, field sites, and units."""
        gatype = self._gatype
        return gatype.algebra.gatype.oriented(
            gatype.subspaces, gatype.orientations, fields=gatype.fields, units=gatype.units,
        )

    @lru_cache(maxsize=None)
    def with_traits(self, *traits: Trait) -> GAType:
        """Adding certified `Trait` facts preserves all subspaces, orientations, field sites, and units."""
        gatype = self._gatype
        return gatype.algebra.gatype.oriented(
            gatype.subspaces, gatype.orientations, (*gatype.traits, *traits), gatype.fields, units=gatype.units,
        )

    @cached_property
    def plain(self) -> GAType:
        """Dropping coupled `fields` reinterprets any site dimensions as ordinary batch axes."""
        gatype = self._gatype
        return (
            gatype.algebra.gatype.oriented(gatype.subspaces, gatype.orientations, gatype.traits, units=gatype.units)
            if gatype.fields else gatype
        )

    @lru_cache(maxsize=None)
    def with_fields(self, fields: tuple) -> GAType:
        """Attaching `(axis, n_sites)` pairs marks output or input axes as coupled field axes."""
        gatype = self._gatype
        return gatype.algebra.gatype.oriented(
            gatype.subspaces, gatype.orientations, gatype.traits, fields, units=gatype.units,
        )

    def unit(self, spec: object) -> GAType:
        """Replacing the per-entry physical units preserves subspaces, orientations, traits, and fields."""
        gatype = self._gatype
        resolved = coerce_unit_grid(spec, gatype.subspaces)
        if gatype.units == resolved:
            return gatype
        return gatype.algebra.gatype.oriented(
            gatype.subspaces, gatype.orientations, gatype.traits, gatype.fields, units=resolved,
        )

    def scaled_unit(self, factor: Unit) -> GAType:
        """Scaling every nonzero blade or entry unit by `factor` attaches `factor` to an unchecked
        type and preserves subspaces, orientations, traits, and fields."""
        factor = Unit.require(factor)
        gatype = self._gatype
        if gatype.units.is_unchecked:
            return self.unit(factor)
        return self.unit(gatype.units.scale(factor))

    def projective(self, unit: Unit) -> GAType:
        """Scaling degenerate blades by `unit` (with `dim.none` on non-degenerate blades when unchecked,
        or multiplying existing degenerate blade units by `unit`) builds projective types such as
        metre points, pixel points, screws, or Galilean boosts in a single-null algebra."""
        unit = Unit.require(unit)
        gatype = self._gatype
        if gatype.arity:
            raise GATypeError("projective unit scaling applies to nullary types")
        deg = gatype.algebra.degenerate_mask
        if deg.bit_count() != 1:
            raise GATypeError("projective unit scaling requires an algebra with a single null generator")
        space = gatype.output_subspace
        if gatype.units.is_unchecked:
            entries = {i: unit if (mask & deg) else dim.none for i, mask in enumerate(space.masks)}
        else:
            entries = {
                i: (u * unit if (space.masks[i] & deg) else u)
                for i, u in gatype.units.items
            }
        return self.unit(UnitGrid.from_entries((len(space),), entries))

    @cached_property
    def unchecked(self) -> GAType:
        """Marking a type unchecked disables unit checks on values that carry it."""
        return self.unit(None)

    @cached_property
    def dimensionless(self) -> GAType:
        """Marking a type dimensionless (`dim.none`) preserves subspaces, orientations, traits, and fields."""
        return self.unit(dim.none)

    @cached_property
    def inverted(self) -> GAType:
        """Reversing a map's arrow and inverting its physical units gives the type of its inverse."""
        gatype = self._gatype
        return self.transposed.derive.unit(invert_map_units(gatype))

    @lru_cache(maxsize=None)
    def traced(self, axis: int) -> GAType:
        """Tracing the output against input `axis` over the same support yields a scalar output whose
        orientation multiplies both paired orientations (`extrinsic` on endomorphisms, `intrinsic` on
        opposite orientations), retaining the remaining inputs."""
        gatype = self._gatype
        inputs = gatype.input_subspaces
        if not inputs[axis].same_support(gatype.output_subspace):
            raise TypeError("trace pairs an output with an input of the same space")
        orientations = gatype.input_orientations
        traced_units = None if gatype.units.is_unchecked else _trace_axes(gatype, 0, axis + 1, "trace")
        return gatype.algebra.gatype.oriented(
            (gatype.algebra.subspace.scalar(),) + inputs[:axis] + inputs[axis + 1:],
            (gatype.orientation * orientations[axis],) + orientations[:axis] + orientations[axis + 1:],
            units=traced_units,
        )

    @lru_cache(maxsize=None)
    def paired(self, first: int, second: int, pairing: Orientation) -> GAType:
        """Pairing and summing two inputs drops both input axes and multiplies the output orientation
        by `pairing`."""
        gatype = self._gatype
        remaining = [axis for axis in range(1, gatype.arity + 1) if axis - 1 not in (first, second)]
        return gatype.algebra.gatype.oriented(
            (gatype.output_subspace,) + tuple(gatype.subspaces[axis] for axis in remaining),
            (gatype.orientation * pairing,) + tuple(gatype.orientations[axis] for axis in remaining),
        )

    @lru_cache(maxsize=None)
    def axis(self, index: int) -> GAType:
        """Extracting axis `index` (`0` for output, `1..arity` for inputs) gives its nullary `GAType`
        with its orientation."""
        gatype = self._gatype
        return gatype.algebra.gatype.oriented((gatype.subspaces[index],), (gatype.orientations[index],))

    @property
    def output(self) -> GAType:
        """Axis `0` as a nullary `GAType` retains the extensor's output subspace and orientation."""
        return self.axis(0)

    @lru_cache(maxsize=None)
    def with_orientations(self, orientations: tuple[Orientation, ...]) -> GAType:
        """Replacing per-axis orientations preserves subspaces, explicit traits, field sites, and units."""
        gatype = self._gatype
        return gatype.algebra.gatype.oriented(
            gatype.subspaces, orientations, gatype.traits, gatype.fields, units=gatype.units,
        )

    @cached_property
    def transposed(self) -> GAType:
        """Reversing axis order reverses per-axis orientations and retains `CoefficientOrthogonal`."""
        gatype = self._gatype
        return numga.gatype.rules.TRANSPOSE.result((gatype,), tuple(reversed(gatype.subspaces)))

    def _with_output_space(self, space: SubSpace) -> GAType:
        gatype = self._gatype
        axes = (space,) + gatype.input_subspaces
        units = None if gatype.units.is_unchecked else remap_grid_subspaces(gatype.units, gatype.subspaces, axes)
        return gatype.algebra.gatype.oriented(axes, gatype.orientations, units=units)

    @cached_property
    def nonscalar(self) -> GAType:
        """Removing the grade-0 blade from the output subspace retains output orientation and all
        input axes."""
        gatype = self._gatype
        space = gatype.output_subspace.restrict(
            mask for mask in gatype.output_subspace.masks if mask
        )
        return self._with_output_space(space)

    @cached_property
    def degenerate(self) -> GAType:
        """Restricting the output to blades containing at least one null generator selects ideal
        elements in projective algebras (such as directions and translation bivectors) while
        preserving orientation and input axes."""
        return self._with_output_space(self._gatype.output_subspace.degenerate())

    @cached_property
    def nondegenerate(self) -> GAType:
        """Restricting the output to blades free of null generators selects origin-passing elements
        in projective algebras (such as Euclidean planes and rotation bivectors) while preserving
        orientation and input axes."""
        return self._with_output_space(self._gatype.output_subspace.nondegenerate())

    def _restricted(self, keep: Callable) -> GAType:
        gatype = self._gatype
        space = gatype.output_subspace.restrict(mask for mask in gatype.output_subspace.masks if keep(mask))
        return self._with_output_space(space)

    @lru_cache(maxsize=None)
    def within(self, part: Algebra | str) -> GAType:
        """Restricting output blades to those built solely from generators of `part` selects embedded
        subalgebra types (`Plane.derive.within(space)` for spatial planes in spacetime, or
        `Even.derive.within(first_space)` in a direct-sum algebra); the scalar blade lies within
        every part."""
        span = _part_mask(self._gatype.algebra, part)
        return self._restricted(lambda mask: not mask & ~span)

    @lru_cache(maxsize=None)
    def touching(self, part: Algebra, *parts: Algebra) -> GAType:
        """Restricting output blades to those containing at least one generator from every given
        subalgebra selects mixed-sector blades (`Bivector.derive.touching(space, amounts)` for
        thermodynamic drives, or `Bivector.derive.touching(Algebra("t-"))` for boost planes)."""
        spans = tuple(_part_mask(self._gatype.algebra, each) for each in (part, *parts))
        return self._restricted(lambda mask: all(mask & span for span in spans))

    @lru_cache(maxsize=None)
    def without(self, part: Algebra) -> GAType:
        """Restricting output blades to those containing no generator from `part` drops a sector
        (`Bivector.derive.without(Algebra("t-"))` for purely spatial bivectors in spacetime)."""
        span = _part_mask(self._gatype.algebra, part)
        return self._restricted(lambda mask: not mask & span)

    @lru_cache(maxsize=None)
    def containing(self, generators: str) -> GAType:
        """Restricting output blades to those containing every basis vector named in `generators`
        selects blades through a chosen subspace."""
        span = _generator_mask(self._gatype.algebra, generators)
        return self._restricted(lambda mask: mask & span == span)

    @cached_property
    def dual(self) -> GAType:
        """Taking the complementary output blades with flipped orientation gives the output type of
        `dual()` while retaining every input axis (`Plane.derive.dual` matches `Plane.dual()`)."""
        gatype = self._gatype
        algebra = gatype.algebra
        space = algebra.subspace.from_masks(algebra.complement(mask) for mask in gatype.output_subspace.masks)
        orientations = (gatype.orientation.flipped,) + gatype.input_orientations
        units = (
            None if gatype.units.is_unchecked
            else gatype.units.remap_axis(0, len(space), _dual_permutation(gatype.output_subspace, space))
        )
        return algebra.gatype.oriented((space,) + gatype.input_subspaces, orientations, units=units)

    @cached_property
    def flipped_orientation(self) -> GAType:
        """Flipping only the output orientation keeps the same blades, traits, fields, and units (such
        as passing between rates and forques in 3D PGA, or rotation centres and mass points in 2D PGA)."""
        gatype = self._gatype
        orientations = (gatype.orientation.flipped,) + gatype.input_orientations
        return gatype.algebra.gatype.oriented(
            gatype.subspaces, orientations, gatype.traits, gatype.fields, units=gatype.units,
        )

    @cached_property
    def reverse_fixed_subspace(self) -> SubSpace:
        """Filtering `output_subspace` to blades with `+1` reversion sign preserves existing order
        and signs without widening support."""
        gatype = self._gatype
        return gatype.output_subspace.restrict(
            mask for mask in gatype.output_subspace.masks
            if gatype.algebra.reverse_sign(mask) == 1
        )

    @lru_cache(maxsize=None)
    def self_product(self, transform: str) -> GAType:
        """Symmetrizing `x * T(x)` gives an extrinsic output over the reachable self-product blades
        (narrowed to grade `0` when a scalar `ProductFact` is entailed) and duplicates input axes."""
        gatype = self._gatype
        fact = ProductFact(SelfProduct(transform), ProductResult.SCALAR)
        if gatype.entails(fact):
            space = gatype.algebra.subspace.scalar()
        else:
            support = symmetric_product_support(gatype.output_subspace, transform)
            space = gatype.algebra.subspace.from_masks(support)
        # Multiplying equal orientations on both sides always gives an extrinsic output.
        out_axes = (space,) + gatype.input_subspaces * 2
        if gatype.units.is_unchecked:
            units: UnitGrid | None = None
        elif gatype.units.is_zero:
            units = UnitGrid.zero(tuple(len(s) for s in out_axes))
        else:
            from numga.binding import BindingPlan, TypeRules

            plain = gatype.derive.plain
            op = gatype.algebra.operator._symmetric_product(plain.derive.unchecked, transform)
            bound = TypeRules.bind(BindingPlan.from_types(op.gatype, ((0, plain), (1, plain))))
            units = remap_grid_subspaces(bound.units, bound.subspaces, out_axes)
        return gatype.algebra.gatype.oriented(
            out_axes, (Orientation.extrinsic,) + gatype.input_orientations * 2,
            units=units,
        )

    @property
    def squared(self) -> GAType:
        return self.self_product("identity")

    @property
    def symmetric_reverse(self) -> GAType:
        return self.self_product("reverse")

    @property
    def symmetric_conjugate(self) -> GAType:
        return self.self_product("clifford_conjugate")

    @property
    def symmetric_scalar_negation(self) -> GAType:
        return self.self_product("scalar_negation")

    @property
    def symmetric_pseudoscalar_negation(self) -> GAType:
        return self.self_product("pseudoscalar_negation")

    @property
    def symmetric_involute(self) -> GAType:
        return self.self_product("involute")

    @cached_property
    def minimal_subalgebra(self) -> GAType:
        """Closing `output_subspace` under pairwise blade XOR products with `{0}` spans the smallest
        extrinsic unital subalgebra containing the output blades, used by general multivector
        inversion."""
        gatype = self._gatype
        masks = {0}
        for generator in gatype.output_subspace.masks:
            masks.update(mask ^ generator for mask in tuple(masks))
        return gatype.algebra.gatype(gatype.algebra.subspace.from_masks(masks), orientation=Orientation.extrinsic)
