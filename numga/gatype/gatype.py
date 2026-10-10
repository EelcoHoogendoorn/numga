"""Whole-extensor structural types."""

from __future__ import annotations

from numbers import Number

from functools import cached_property, lru_cache
from types import NotImplementedType
from typing import TYPE_CHECKING, Iterable, Tuple

from numga.subspace import SubSpace

from .traits import (
    EMPTY_TRAITS,
    ProductFact,
    ProductResult,
    ReverseProductOne,
    SelfProduct,
    Trait,
    TraitSet,
    Versor,
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

    A slot is a field slot when it ranges over sites as well as blades: ``fields``
    pairs each such slot with its number of sites. The kernel holds one site axis
    per field slot, in slot order, between the batch axes and the coefficient axes.
    """

    __slots__ = ("_subspaces", "_traits", "_fields", "_algebra", "_hash", "__dict__")

    def __init__(
        self,
        subspaces: Iterable[SubSpace],
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
        object.__setattr__(self, "_subspaces", normalized_subspaces)
        object.__setattr__(self, "_traits", normalized_traits)
        object.__setattr__(self, "_fields", normalized_fields)
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(
            self,
            "_hash",
            hash((type(self), normalized_subspaces, normalized_traits, normalized_fields)),
        )

    @property
    def subspaces(self) -> Tuple[SubSpace, ...]:
        return self._subspaces

    @property
    def axes(self) -> Tuple[SubSpace, ...]:
        return self.subspaces

    @cached_property
    def structural_shape(self) -> tuple[int, ...]:
        """The trailing kernel axes: the site axes of the field slots, then one coefficient axis per slot."""
        return self.site_shape + tuple(len(axis) for axis in self.subspaces)

    @property
    def fields(self) -> tuple[tuple[int, int], ...]:
        """Each field slot, output 0 and inputs from 1, with its number of sites."""
        return self._fields

    @cached_property
    def site_shape(self) -> tuple[int, ...]:
        return tuple(sites for _, sites in self._fields)

    @cached_property
    def has_fields(self) -> bool:
        return bool(self._fields)

    @cached_property
    def has_input_fields(self) -> bool:
        """Whether an input slot ranges over sites, so that binding it sums over them."""
        return any(slot for slot, _ in self._fields)

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

    @lru_cache(maxsize=None)
    def grade_transform_is_identity(self, transform: str) -> bool:
        """Whether the transform fixes every blade on the output axis."""

        from numga.algebra.self_product import grade_transform_sign

        return all(
            grade_transform_sign(self.algebra, transform, mask) == 1
            for mask in self.output_subspace.masks
        )

    @cached_property
    def is_study(self) -> bool:
        """A generalized Study number: scalar plus a part with scalar square."""

        return self.derive.symmetric_scalar_negation.is_scalar

    @cached_property
    def is_scalar_bivector(self) -> bool:
        """Nullary support confined to grades zero and two."""

        spaces = self.algebra.subspace
        return self <= spaces.scalar() + spaces.bivector()

    @property
    def is_square_map(self) -> bool:
        return self.arity == 1 and len(self.subspaces[0]) == len(self.subspaces[1])

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

        other = _comparison_gatype(other)
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

        other = _comparison_gatype(other)
        return (
            other is not None
            and self.refines(other)
            and not other.refines(self)
        )

    def overlaps(self, other: object) -> bool:
        """Whether both patterns can describe at least one common type."""

        other = _comparison_gatype(other)
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
            and self.traits == other.traits
            and self.fields == other.fields
        )

    def __hash__(self) -> int:
        return self._hash

    def __le__(self, other: object) -> bool | NotImplementedType:
        other = _comparison_gatype(other)
        if other is None:
            return NotImplemented
        return self.refines(other)

    def __lt__(self, other: object) -> bool | NotImplementedType:
        other = _comparison_gatype(other)
        if other is None:
            return NotImplemented
        return self.strictly_refines(other)

    def __ge__(self, other: object) -> bool | NotImplementedType:
        other = _comparison_gatype(other)
        if other is None:
            return NotImplemented
        return other.refines(self)

    def __gt__(self, other: object) -> bool | NotImplementedType:
        other = _comparison_gatype(other)
        if other is None:
            return NotImplemented
        return other.strictly_refines(self)

    def _identity(self) -> Extensor:
        if self.arity:
            raise TypeError("only a nullary GAType promotes to an identity map")
        return self.algebra.operator.identity(self.output_subspace)

    def __call__(self, *operands: object) -> Extensor:
        """A nullary type is its identity map, applied or composed like any other map."""
        return self._identity()(*operands)

    def __getattr__(self, name: str) -> object:
        """A nullary type is its identity map: Extensor attributes it does not define itself act on
        that map, so `Bivector.reverse()` is the reversion and `Vector.scalar_product(Vector)` the
        metric. What only a value has, its coefficients and their backend, a type does not."""
        from numga.extensor import Extensor

        if name.startswith("_") or name in _VALUE_DATA or not hasattr(Extensor, name) or self.arity:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._identity(), name)

    def __add__(self, other: object) -> Extensor | NotImplementedType:
        from numga.extensor import Extensor

        if isinstance(other, Extensor):
            return self._identity() + other
        return NotImplemented

    def __sub__(self, other: object) -> Extensor | NotImplementedType:
        from numga.extensor import Extensor

        if isinstance(other, Extensor):
            return self._identity() - other
        return NotImplemented

    def __neg__(self) -> Extensor:
        return -self._identity()

    def __mul__(self, other: object) -> Extensor | NotImplementedType:
        from numga.expression import geometric_product, is_expression_operand

        if isinstance(other, Number):
            return self._identity() * other
        if not is_expression_operand(other):
            return NotImplemented
        return geometric_product(self, other)

    def __rmul__(self, other: object) -> Extensor | NotImplementedType:
        from numga.expression import geometric_product, is_expression_operand

        if isinstance(other, Number):
            return self._identity() * other
        if not is_expression_operand(other):
            return NotImplemented
        return geometric_product(other, self)

    def dual(self) -> Extensor:
        """The right-Hodge dual as a map with this space in its open slot."""
        return self.algebra.operator.dual(self.output_subspace)

    def dual_inverse(self) -> Extensor:
        return self.algebra.operator.dual_inverse(self.output_subspace)

    def wedge(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import wedge

        return wedge(self, other)

    def inner(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import inner

        return inner(self, other)

    def commutator(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import commutator

        return commutator(self, other)

    def anticommutator(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import anticommutator

        return anticommutator(self, other)

    def regressive(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import regressive

        return regressive(self, other)

    def sandwich(self, passenger: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import sandwich

        return sandwich(self, passenger)

    def __xor__(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import is_expression_operand

        return self.wedge(other) if is_expression_operand(other) else NotImplemented

    def __and__(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import is_expression_operand

        return self.regressive(other) if is_expression_operand(other) else NotImplemented

    def __or__(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import is_expression_operand

        return self.inner(other) if is_expression_operand(other) else NotImplemented

    def __rshift__(self, other: SubSpace | GAType | Extensor) -> Extensor:
        return self.sandwich(other)

    def __reduce__(self) -> tuple[type[GAType], tuple[tuple[SubSpace, ...], TraitSet, tuple[tuple[int, int], ...]]]:
        return type(self), (self.subspaces, self.traits, self.fields)

    @property
    def signature(self) -> str:
        """Output and inputs by subspace name: in bivector <- bivector, with any traits after a bar."""

        sites = dict(self.fields)
        output, *inputs = (
            space.type_name + (f"[{sites[slot]}]" if slot in sites else "")
            for slot, space in enumerate(self.subspaces)
        )
        text = f"{output} <- {', '.join(inputs)}" if inputs else output
        return f"{text} | {', '.join(trait.name for trait in self.traits)}" if self.traits else text

    def __repr__(self) -> str:
        return f"GAType({self.signature})"


def _comparison_gatype(value: object) -> GAType | None:
    """Lift a structural nullary shorthand only for semantic comparisons."""

    if isinstance(value, GAType):
        return value
    if isinstance(value, SubSpace):
        return value.algebra.gatype(value)
    return None


def _normalize_fields(fields: Iterable[tuple[int, int]], slots: int) -> tuple[tuple[int, int], ...]:
    normalized = tuple(sorted((int(slot), int(sites)) for slot, sites in fields))
    if any(slot < 0 or slot >= slots or sites < 1 for slot, sites in normalized):
        raise ValueError(f"field slots must lie in [0, {slots}) and hold at least one site; got {normalized}")
    if len({slot for slot, _ in normalized}) != len(normalized):
        raise ValueError(f"a slot holds one site axis; got {normalized}")
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
        return gatype.algebra.gatype(gatype.subspaces, fields=gatype.fields)

    @lru_cache(maxsize=None)
    def with_traits(self, *traits: Trait) -> GAType:
        gatype = self._gatype
        return gatype.algebra.gatype(gatype.subspaces, (*gatype.traits, *traits), gatype.fields)

    @cached_property
    def plain(self) -> GAType:
        """The same slots and traits over blades alone: a field's sites read as batch axes."""

        gatype = self._gatype
        return gatype.algebra.gatype(gatype.subspaces, gatype.traits) if gatype.fields else gatype

    @lru_cache(maxsize=None)
    def with_fields(self, fields: tuple[tuple[int, int], ...]) -> GAType:
        gatype = self._gatype
        return gatype.algebra.gatype(gatype.subspaces, gatype.traits, fields)

    @cached_property
    def transposed(self) -> GAType:
        from numga.binding import TypeRules

        gatype = self._gatype
        return TypeRules.operation("transpose", (gatype,), tuple(reversed(gatype.subspaces)))

    @cached_property
    def nonscalar(self) -> GAType:
        """Nonscalar output support, retaining every open input axis."""

        gatype = self._gatype
        space = gatype.output_subspace.restrict(
            mask for mask in gatype.output_subspace.masks if mask
        )
        return gatype.algebra.gatype((space,) + gatype.input_subspaces)

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

        from numga.algebra.self_product import symmetric_product_support

        gatype = self._gatype
        fact = ProductFact(SelfProduct(transform), ProductResult.SCALAR)
        if gatype.entails(fact):
            space = gatype.algebra.subspace.scalar()
        else:
            support = symmetric_product_support(gatype.output_subspace, transform)
            space = gatype.algebra.subspace.from_masks(support)
        return gatype.algebra.gatype((space,) + gatype.input_subspaces * 2)

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
        return gatype.algebra.gatype(gatype.algebra.subspace.from_masks(masks))
