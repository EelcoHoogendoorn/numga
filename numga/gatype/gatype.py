"""Whole-extensor structural types."""

from __future__ import annotations

from numbers import Number

from functools import cached_property, lru_cache
from types import NotImplementedType
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np

import numga
from numga.algebra.self_product import grade_transform_signs, symmetric_product_support
from numga.subspace import SubSpace

from .metric import Metric
from .orientation import Orientation
from .queries import Form, OverSites, SelfProducts, AxisRoles
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
    from numga.subspace import SupportKey


# What a value has and its type does not: coefficients, their backend, and their layout.
_VALUE_DATA = frozenset({"kernel", "context", "gatype", "shape", "ndim", "dtype", "subspace"})

_SELF_PRODUCT_TRANSFORMS = (
    "identity", "reverse", "clifford_conjugate", "scalar_negation",
    "pseudoscalar_negation", "involute",
)


class GAType:
    """Immutable type of an entire extensor, with output-first axes.

    At least one SubSpace is required because every extensor has an output
    carrier, including nullary extensors.  All axes must be owned by the exact
    same algebra object.

    A axis is a field axis when it ranges over sites as well as blades: ``fields``
    pairs each such axis with its number of sites. The kernel holds one site axis
    per field axis, in axis order, between the batch axes and the coefficient axes.

    Every axis has an orientation, intrinsic or extrinsic (see ``numga.gatype.orientation``): two
    types over the same blades with different orientations are different types.
    """

    __slots__ = ("_subspaces", "_orientations", "_traits", "_fields", "_algebra", "_hash", "__dict__")

    def __init__(
        self,
        subspaces: Iterable[SubSpace],
        orientations: Iterable[Orientation],
        traits: TraitSet = EMPTY_TRAITS,
        fields: tuple[tuple[int, int], ...] = (),
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
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(
            self,
            "_hash",
            hash((type(self), normalized_subspaces, normalized_orientations, normalized_traits, normalized_fields)),
        )

    @property
    def subspaces(self) -> Tuple[SubSpace, ...]:
        return self._subspaces

    @property
    def axes(self) -> Tuple[SubSpace, ...]:
        return self.subspaces

    @property
    def orientations(self) -> Tuple[Orientation, ...]:
        """The orientation of every axis, output first."""
        return self._orientations

    @property
    def orientation(self) -> Orientation:
        """The orientation of the output."""
        return self._orientations[0]

    @property
    def input_orientations(self) -> Tuple[Orientation, ...]:
        return self._orientations[1:]

    @cached_property
    def structural_shape(self) -> tuple[int, ...]:
        """The trailing kernel axes: the site axes of the field axes, then one coefficient axis per axis."""
        return self.site_shape + tuple(len(axis) for axis in self.subspaces)

    @property
    def fields(self) -> tuple[tuple[int, int], ...]:
        """Each field axis, output 0 and inputs from 1, with its number of sites."""
        return self._fields

    @cached_property
    def site_shape(self) -> tuple[int, ...]:
        return tuple(sites for _, sites in self._fields)

    @cached_property
    def has_fields(self) -> bool:
        return bool(self._fields)

    @cached_property
    def has_input_fields(self) -> bool:
        """Whether an input ranges over sites, so that binding it sums over them."""
        return any(axis for axis, _ in self._fields)

    @property
    def traits(self) -> TraitSet:
        return self._traits

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    def output_subspace(self) -> SubSpace:
        return self.subspaces[0]

    @property
    def input_subspaces(self) -> Tuple[SubSpace, ...]:
        return self.subspaces[1:]

    @property
    def arity(self) -> int:
        return len(self.subspaces) - 1

    @cached_property
    def is_scalar(self) -> bool:
        """A nullary scalar carrier, including the empty zero carrier."""

        return self <= self.algebra.subspace.scalar()

    @cached_property
    def is_reoriented_scalar(self) -> bool:
        """A scalar stored against the basis element -1 rather than +1."""

        return self.is_scalar and self.output_subspace.signs == (-1,)

    @cached_property
    def is_empty(self) -> bool:
        """A nullary extensor with no coefficient support."""

        return self <= self.algebra.subspace.empty()

    @cached_property
    def derive(self) -> Derivations:
        """The types derived from this one: the type of its square, of its transpose, without its
        traits. Operations on the type itself build maps, since a type acts as its identity map."""

        return Derivations(self)

    @cached_property
    def axis_roles(self) -> AxisRoles:
        """Its axes: which carry an index, which pair, and what an operand given for one binds."""
        return AxisRoles(self)

    @cached_property
    def form(self) -> Form:
        """The type read as a form: whether it pairs one space with itself, and what it solves against."""
        return Form(self)

    @cached_property
    def over_sites(self) -> OverSites:
        """The type read as one map or form over the sites of its fields."""
        return OverSites(self)

    @cached_property
    def metric(self) -> Metric:
        """The type read as a axis: its default metric, and what that metric is like."""
        return Metric(self)

    @cached_property
    def self_products(self) -> SelfProducts:
        """What its facts say about the products of its values with their own transforms."""
        return SelfProducts(self)

    @lru_cache(maxsize=None)
    def grade_transform_is_identity(self, transform: str) -> bool:
        """Whether the transform fixes every blade on the output axis."""

        return bool(np.all(grade_transform_signs(self.algebra, transform, self.output_subspace.masks) == 1))

    @cached_property
    def is_study(self) -> bool:
        """A generalized Study number: scalar plus a part with scalar square."""

        return self.derive.symmetric_scalar_negation.is_scalar

    @cached_property
    def is_scalar_bivector(self) -> bool:
        """Nullary support confined to grades zero and two."""

        spaces = self.algebra.subspace
        return self <= spaces.scalar() + spaces.bivector()

    @cached_property
    def is_bivector(self) -> bool:
        """A nullary type within the bivectors, in either orientation."""
        return self <= self.algebra.subspace.bivector()

    @cached_property
    def is_even(self) -> bool:
        """A nullary type within the even grades, in either orientation."""
        return self <= self.algebra.subspace.even()

    @cached_property
    def is_rotor(self) -> bool:
        """A type that refines the rotor: even, with unit reverse product, a versor."""
        return self <= self.algebra.gatype.rotor()

    @cached_property
    def is_nilpotent(self) -> bool:
        """A type that squares to zero by its grades alone."""
        return self.derive.squared.is_empty

    @cached_property
    def has_null_wedge(self) -> bool:
        """The 4-vector part of this type's square squares to zero."""
        algebra = self.algebra
        wedge = self.derive.squared.output_subspace.intersection(algebra.subspace.k_vector(4))
        # a part of a square, so extrinsic
        return algebra.gatype(wedge, orientation=Orientation.extrinsic).derive.squared.is_empty

    @cached_property
    def is_form(self) -> bool:
        """A scalar-valued map of two axes."""
        return self.arity == 2 and self.output_subspace.same_support(self.algebra.subspace.scalar())

    @cached_property
    def is_exterior_map(self) -> bool:
        """A map from vectors to vectors, or from antivectors to antivectors."""
        if self.arity != 1:
            return False
        algebra = self.algebra
        inputs = {algebra.grade(mask) for mask in self.input_subspaces[0].masks}
        outputs = {algebra.grade(mask) for mask in self.output_subspace.masks}
        return inputs == outputs and inputs in ({1}, {algebra.dimension - 1})

    @property
    def maps_equal_dimensions(self) -> bool:
        """A map between two spaces of equal dimension: what an inverse or a solve needs."""
        return self.arity == 1 and len(self.subspaces[0]) == len(self.subspaces[1])

    @property
    def is_endomorphism(self) -> bool:
        """A map from a space to itself: the same blades and the same orientation in and out. Only such
        a map can be fed back into itself, as eigenvectors, traces and powers do."""
        return (
            self.arity == 1
            and self.subspaces[0].same_support(self.subspaces[1])
            and self.orientations[0] is self.orientations[1]
        )

    @property
    def is_correlation(self) -> bool:
        """A map from a space to its dual: the complementary blades, oppositely oriented, as from points
        to planes, or from rates to forques. Paired with its input by the regressive product it gives an
        ordinary number; its eigenvectors exist only against a second correlation."""
        if self.arity != 1:
            return False
        complements = {self.algebra.complement(mask) for mask in self.subspaces[1].masks}
        return set(self.subspaces[0].masks) == complements and self.orientations[0] is not self.orientations[1]

    @lru_cache(maxsize=None)
    def reduces_to_scalar(self, steps: int) -> bool:
        """Whether self-products reach a scalar extensor in these steps.

        Scalar output support alone does not eliminate open input axes.
        """

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
        """Facts retained by inversion under its invertible-input contract."""

        if self.arity:
            return EMPTY_TRAITS
        traits: list[Trait] = [Versor] if self.entails(Versor) else []
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
        """Result facts under the reverse-product normalization contract."""

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
    def representation_key(self) -> tuple[tuple[SubSpace, ...], TraitSet]:
        return self.subspaces, self.traits

    @property
    def semantic_key(self) -> tuple[Algebra, tuple[SupportKey, ...], TraitSet]:
        return (
            self.algebra,
            tuple(axis.support_key for axis in self.subspaces),
            self.traits,
        )

    def entails(self, required: Trait | TraitSet | Iterable[Trait]) -> bool:
        """Whether this type's certified facts prove ``required``."""

        return self.effective_traits.entails(required)

    @cached_property
    def effective_traits(self) -> TraitSet:
        """Explicit facts plus universal facts implied by the carrier."""

        structural = structurally_implied_traits(self.subspaces)
        return TraitSet(self.traits.traits + structural.traits)

    def refines(self, other: object) -> bool:
        """Semantic support/trait inclusion, independent of coefficient layout.

        This is a preorder on concrete layouts: mutual refinement need not
        imply equality. Exact equality and hashing still include every axis.
        """

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
        """Whether this type refines ``other`` but not conversely."""

        other = _comparison_gatype(other, self)
        return (
            other is not None
            and self.refines(other)
            and not other.refines(self)
        )

    def overlaps(self, other: object) -> bool:
        """Whether both patterns can describe at least one common type."""

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
    def __add__(self, other: object) -> Extensor:
        return self._identity() + other

    def __radd__(self, other: object) -> Extensor:
        return other + self._identity()

    def __sub__(self, other: object) -> Extensor:
        return self._identity() - other

    def __rsub__(self, other: object) -> Extensor:
        return other - self._identity()

    def __neg__(self) -> Extensor:
        return -self._identity()

    def __invert__(self) -> Extensor:
        return ~self._identity()

    def __truediv__(self, other: object) -> Extensor:
        return self._identity() / other

    # In a product, a type is an open axis; a number scales its identity map.
    def __mul__(self, other: object) -> Extensor | NotImplementedType:
        if isinstance(other, Number):
            return self._identity() * other
        if not numga.expression.is_expression_operand(other):
            return NotImplemented
        return numga.expression.geometric_product(self, other)

    def __rmul__(self, other: object) -> Extensor | NotImplementedType:
        if isinstance(other, Number):
            return self._identity() * other
        if not numga.expression.is_expression_operand(other):
            return NotImplemented
        return numga.expression.geometric_product(other, self)

    def dual(self) -> Extensor:
        """The right-Hodge dual as a map with this space in its open axis."""
        return self.algebra.operator.dual(self)

    def dual_inverse(self) -> Extensor:
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
        """The action of a versor of this type on the passenger, `self >> passenger`."""
        return numga.expression.transform(self, passenger)

    def __xor__(self, other: GAType | Extensor) -> Extensor:
        return self.wedge(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __and__(self, other: GAType | Extensor) -> Extensor:
        return self.regressive(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __or__(self, other: GAType | Extensor) -> Extensor:
        return self.inner(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rxor__(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.wedge(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rand__(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.regressive(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __ror__(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.inner(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rshift__(self, other: GAType | Extensor) -> Extensor:
        return self.transform(other)

    def __lshift__(self, other: GAType | Extensor) -> Extensor:
        return numga.expression.reverse_transform(self, other)

    def __reduce__(self) -> tuple[type[GAType], tuple[object, ...]]:
        return type(self), (self.subspaces, self.orientations, self.traits, self.fields)

    @property
    def signature(self) -> str:
        """Output and inputs by orientation and subspace name: intrinsic in bivector <- extrinsic bivector,
        with any traits after a bar."""

        sites = dict(self.fields)
        output, *inputs = (
            f"{orientation.value} {space.type_name}"
            + (f"[{sites[axis]}]" if axis in sites else "")
            for axis, (space, orientation) in enumerate(zip(self.subspaces, self.orientations))
        )
        text = f"{output} <- {', '.join(inputs)}" if inputs else output
        return f"{text} | {', '.join(trait.name for trait in self.traits)}" if self.traits else text

    def __repr__(self) -> str:
        return f"GAType({self.signature})"


def _generator_mask(algebra, generators: str) -> int:
    """The basis vectors named in a whitespace-separated string, as one mask."""
    masks = algebra.subspace(generators).masks
    if any(mask.bit_count() != 1 for mask in masks):
        raise ValueError(f"name basis vectors, not blades: {generators!r}")
    span = 0
    for mask in masks:
        span |= mask
    return span


def _comparison_gatype(value: object, like: GAType) -> GAType | None:
    """The other side of a comparison with `like`. Comparisons are by support and traits, so a bare
    subspace compares as the nullary type over it with `like`'s own orientation: nothing is assumed."""

    if isinstance(value, GAType):
        return value
    if isinstance(value, SubSpace):
        return value.algebra.gatype.oriented((value,), (like.orientation,))
    return None


def _normalize_fields(fields: Iterable[tuple[int, int]], axes: int) -> tuple[tuple[int, int], ...]:
    normalized = tuple(sorted((int(axis), int(sites)) for axis, sites in fields))
    if any(axis < 0 or axis >= axes or sites < 1 for axis, sites in normalized):
        raise ValueError(f"field axes must lie in [0, {axes}) and hold at least one site; got {normalized}")
    if len({axis for axis, _ in normalized}) != len(normalized):
        raise ValueError(f"a axis holds one site axis; got {normalized}")
    return normalized


class Derivations:
    """The types the type rules derive from one type, apart from the type itself."""

    __slots__ = ("_gatype", "__dict__")

    def __init__(self, gatype: GAType) -> None:
        self._gatype = gatype

    @cached_property
    def structural(self) -> GAType:
        """The same axes and fields without explicit trait assertions."""

        gatype = self._gatype
        return gatype.algebra.gatype.oriented(gatype.subspaces, gatype.orientations, fields=gatype.fields)

    @lru_cache(maxsize=None)
    def with_traits(self, *traits: Trait) -> GAType:
        gatype = self._gatype
        return gatype.algebra.gatype.oriented(gatype.subspaces, gatype.orientations, (*gatype.traits, *traits), gatype.fields)

    @cached_property
    def plain(self) -> GAType:
        """The same axes and traits over blades alone: a field's sites read as batch axes."""

        gatype = self._gatype
        return gatype.algebra.gatype.oriented(gatype.subspaces, gatype.orientations, gatype.traits) if gatype.fields else gatype

    @lru_cache(maxsize=None)
    def with_fields(self, fields: tuple[tuple[int, int], ...]) -> GAType:
        gatype = self._gatype
        return gatype.algebra.gatype.oriented(gatype.subspaces, gatype.orientations, gatype.traits, fields)

    @lru_cache(maxsize=None)
    def traced(self, axis: int) -> GAType:
        """The type of the trace of the output against one input of the same space: a scalar, with
        the remaining inputs. Equal orientations pair to an ordinary number, as on an endomorphism;
        opposite ones to a chirality."""
        gatype = self._gatype
        inputs = gatype.input_subspaces
        if not inputs[axis].same_support(gatype.output_subspace):
            raise TypeError("trace pairs an output with an input of the same space")
        orientations = gatype.input_orientations
        return gatype.algebra.gatype.oriented(
            (gatype.algebra.subspace.scalar(),) + inputs[:axis] + inputs[axis + 1:],
            (gatype.orientation * orientations[axis],) + orientations[:axis] + orientations[axis + 1:],
        )

    @lru_cache(maxsize=None)
    def paired(self, first: int, second: int, pairing: Orientation) -> GAType:
        """The type left when two inputs are paired and summed over: the remaining inputs, and an
        output whose orientation multiplies with that of the pairing the sum runs through."""
        gatype = self._gatype
        remaining = [axis for axis in range(1, gatype.arity + 1) if axis - 1 not in (first, second)]
        return gatype.algebra.gatype.oriented(
            (gatype.output_subspace,) + tuple(gatype.subspaces[axis] for axis in remaining),
            (gatype.orientation * pairing,) + tuple(gatype.orientations[axis] for axis in remaining),
        )

    @lru_cache(maxsize=None)
    def axis(self, index: int) -> GAType:
        """The nullary type of one axis, output 0 and inputs from 1, with its orientation."""
        gatype = self._gatype
        return gatype.algebra.gatype.oriented((gatype.subspaces[index],), (gatype.orientations[index],))

    @property
    def output(self) -> GAType:
        """The nullary type of the output, with its orientation."""
        return self.axis(0)

    @lru_cache(maxsize=None)
    def with_orientations(self, orientations: tuple[Orientation, ...]) -> GAType:
        """The same axes, traits and fields with the given orientations, one per axis."""
        gatype = self._gatype
        return gatype.algebra.gatype.oriented(gatype.subspaces, orientations, gatype.traits, gatype.fields)

    @cached_property
    def transposed(self) -> GAType:
        gatype = self._gatype
        return numga.gatype.rules.TRANSPOSE.result((gatype,), tuple(reversed(gatype.subspaces)))

    @cached_property
    def nonscalar(self) -> GAType:
        """Nonscalar output support, retaining every open input axis."""

        gatype = self._gatype
        space = gatype.output_subspace.restrict(
            mask for mask in gatype.output_subspace.masks if mask
        )
        return gatype.algebra.gatype.oriented((space,) + gatype.input_subspaces, gatype.orientations)

    @cached_property
    def degenerate(self) -> GAType:
        """The output's blades that contain a degenerate direction, with its orientation, retaining every
        open input axis: in PGA the ideal elements, such as directions and translations."""

        gatype = self._gatype
        space = gatype.output_subspace.degenerate()
        return gatype.algebra.gatype.oriented((space,) + gatype.input_subspaces, gatype.orientations)

    @cached_property
    def nondegenerate(self) -> GAType:
        """The output's blades free of degenerate directions, with its orientation, retaining every open
        input axis: in PGA the elements through the origin, such as planes and rotation axes."""

        gatype = self._gatype
        space = gatype.output_subspace.nondegenerate()
        return gatype.algebra.gatype.oriented((space,) + gatype.input_subspaces, gatype.orientations)

    @lru_cache(maxsize=None)
    def within(self, generators: str) -> GAType:
        """The output's blades built from these basis vectors alone, such as `"x y z"`, with its
        orientation, retaining every open input axis."""

        gatype = self._gatype
        span = _generator_mask(gatype.algebra, generators)
        space = gatype.output_subspace.restrict(mask for mask in gatype.output_subspace.masks if not mask & ~span)
        return gatype.algebra.gatype.oriented((space,) + gatype.input_subspaces, gatype.orientations)

    @lru_cache(maxsize=None)
    def containing(self, generators: str) -> GAType:
        """The output's blades that contain every one of these basis vectors, such as `"z"`, with its
        orientation, retaining every open input axis."""

        gatype = self._gatype
        span = _generator_mask(gatype.algebra, generators)
        space = gatype.output_subspace.restrict(mask for mask in gatype.output_subspace.masks if mask & span == span)
        return gatype.algebra.gatype.oriented((space,) + gatype.input_subspaces, gatype.orientations)

    @cached_property
    def flipped_orientation(self) -> GAType:
        """The same blades, traits and fields with the output oriented the other way: a rate and a
        forque in 3D PGA, a rotation centre and a mass point in 2D PGA."""

        gatype = self._gatype
        orientations = (gatype.orientation.flipped,) + gatype.input_orientations
        return gatype.algebra.gatype.oriented(gatype.subspaces, orientations, gatype.traits, gatype.fields)

    @cached_property
    def reverse_fixed_subspace(self) -> SubSpace:
        """Output blades fixed by reversal, without widening their support."""

        gatype = self._gatype
        return gatype.output_subspace.restrict(
            mask for mask in gatype.output_subspace.masks
            if gatype.algebra.reverse_sign(mask) == 1
        )

    @lru_cache(maxsize=None)
    def self_product(self, transform: str) -> GAType:
        """Symmetric self-product type, retaining both operands' open inputs."""

        gatype = self._gatype
        fact = ProductFact(SelfProduct(transform), ProductResult.SCALAR)
        if gatype.entails(fact):
            space = gatype.algebra.subspace.scalar()
        else:
            support = symmetric_product_support(gatype.output_subspace, transform)
            space = gatype.algebra.subspace.from_masks(support)
        # A product of a value with itself has equal orientations on both sides, so it is extrinsic.
        return gatype.algebra.gatype.oriented(
            (space,) + gatype.input_subspaces * 2, (Orientation.extrinsic,) + gatype.input_orientations * 2,
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
        """Unital blade-generated carrier used by the general inverse solve."""

        gatype = self._gatype
        masks = {0}
        for generator in gatype.output_subspace.masks:
            masks.update(mask ^ generator for mask in tuple(masks))
        # The subalgebra holds products of the value's blades with themselves: extrinsic.
        return gatype.algebra.gatype(gatype.algebra.subspace.from_masks(masks), orientation=Orientation.extrinsic)
