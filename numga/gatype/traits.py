"""Immutable trait facts and logical implication for whole Extensor types."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from types import NotImplementedType
from typing import TYPE_CHECKING, Callable, Tuple

if TYPE_CHECKING:
    from numga.subspace import SubSpace


class Trait:
    """An immutable fact with local implication and compatibility rules.

    Structured facts store their state in hashable parameters. The container
    need not know their mathematics: subclasses provide implications and peer
    validation. Subclasses with additional state supply their own pickling.
    """

    __slots__ = ("_name", "_valid_arities", "_parameters", "_hash")

    def __init__(
        self,
        name: str,
        *,
        valid_arities: Iterable[int] | None,
        parameters: tuple[object, ...] = (),
    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("a trait name must be a non-empty string")
        if valid_arities is None:
            normalized_arities = None
        else:
            normalized_arities = frozenset(valid_arities)
            if not normalized_arities or any(
                isinstance(arity, bool) or not isinstance(arity, int) or arity < 0
                for arity in normalized_arities
            ):
                raise TypeError("trait arities must be non-negative integers")
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_valid_arities", normalized_arities)
        object.__setattr__(self, "_parameters", tuple(parameters))
        object.__setattr__(
            self,
            "_hash",
            hash((type(self), name, normalized_arities, self._parameters)),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def valid_arities(self) -> frozenset[int] | None:
        """Finite arities usable for registration shorthand, if any."""

        return self._valid_arities

    def validate_arity(self, arity: int) -> None:
        if self.valid_arities is None:
            raise NotImplementedError(
                f"dynamic-arity trait {self.name} must implement validate_arity"
            )
        if arity not in self.valid_arities:
            expected_values = sorted(self.valid_arities)
            if len(expected_values) == 1:
                expected = f"arity {expected_values[0]}"
            else:
                expected = "arities {" + ", ".join(
                    str(value) for value in expected_values
                ) + "}"
            raise ValueError(
                f"trait {self.name} is valid only for {expected}, "
                f"not arity {arity}"
            )

    @property
    def sort_key(self) -> tuple[str, str, str, tuple[int, ...], str]:
        return (
            type(self).__module__, type(self).__qualname__, self.name,
            tuple(sorted(self.valid_arities or ())), repr(self._parameters),
        )

    def implied_traits(self) -> tuple["Trait", ...]:
        return _DIRECT_IMPLICATIONS.get(self, ())

    def validate_peers(self, traits: frozenset["Trait"]) -> None:
        """Reject incompatible known facts; absence never means false."""

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("Trait is immutable")

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, Trait):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.name == other.name
            and self.valid_arities == other.valid_arities
            and self._parameters == other._parameters
        )

    def __hash__(self) -> int:
        return self._hash

    def __reduce__(self) -> tuple[Callable[..., Trait], tuple[object, ...]]:
        assert self.valid_arities is not None
        return _restore_trait, (
            type(self), self.name, tuple(sorted(self.valid_arities)), self._parameters,
        )

    def __repr__(self) -> str:
        return self.name


class ProductResult(str, Enum):
    """Category of a full self-product, not its scalar projection or magnitude.

    These are compile-time facts, not stored coefficient values. ``SCALAR``
    includes both zero and nonzero scalars; ``NONZERO`` does not imply ``ONE``.
    Scalars need not be real or positive. These categories also work over C.
    ``ONE`` means scalar 1, not an implicitly chosen identity for any product.
    """

    SCALAR = "scalar"
    ZERO = "zero"
    NONZERO = "nonzero"
    ONE = "one"


@dataclass(frozen=True, slots=True)
class SelfProduct:
    """Identify the full expression ``product(x, transform(x))``.

    Operation names are mathematical identities, not executable callbacks.
    Declaring a family grants no propagation law or numerical implementation.
    In particular, reversion never means complex coefficient conjugation.
    """

    transform: str
    product: str = "geometric_product"

    def __post_init__(self) -> None:
        for name in (self.transform, self.product):
            if not isinstance(name, str) or not name:
                raise TypeError("self-product operations need non-empty names")


ReverseProduct = SelfProduct("reverse")
CliffordConjugateProduct = SelfProduct("clifford_conjugate")
GradeInvolutionProduct = SelfProduct("involute")

_PRODUCT_NAMES = {
    ReverseProduct: "ReverseProduct",
    CliffordConjugateProduct: "CliffordConjugateProduct",
    GradeInvolutionProduct: "GradeInvolutionProduct",
}


def _result_entails(actual: ProductResult, required: ProductResult) -> bool:
    return (
        actual is required
        or required is ProductResult.SCALAR
        or (actual is ProductResult.ONE and required is ProductResult.NONZERO)
    )


def _results_conflict(left: ProductResult, right: ProductResult) -> bool:
    return (
        left is ProductResult.ZERO
        and right in (ProductResult.NONZERO, ProductResult.ONE)
    ) or (
        right is ProductResult.ZERO
        and left in (ProductResult.NONZERO, ProductResult.ONE)
    )


class ProductFact(Trait):
    """A nullary value's full self-product has the specified result category."""

    __slots__ = ()

    def __init__(self, self_product: SelfProduct, result: ProductResult) -> None:
        if not isinstance(self_product, SelfProduct):
            raise TypeError("ProductFact requires a SelfProduct")
        if not isinstance(result, ProductResult):
            raise TypeError("ProductFact result must be a ProductResult")
        name = _PRODUCT_NAMES.get(self_product, repr(self_product))
        super().__init__(
            name + result.value.title(), valid_arities=(0,),
            parameters=(self_product, result),
        )

    @property
    def self_product(self) -> SelfProduct:
        return self._parameters[0]

    @property
    def result(self) -> ProductResult:
        return self._parameters[1]

    def implied_traits(self) -> tuple[Trait, ...]:
        return tuple(
            ProductFact(self.self_product, result)
            for result in ProductResult
            if result is not self.result and _result_entails(self.result, result)
        )

    def validate_peers(self, traits: frozenset[Trait]) -> None:
        for other in traits:
            if (
                isinstance(other, ProductFact)
                and other.self_product == self.self_product
                and _results_conflict(self.result, other.result)
            ):
                raise ValueError(f"{self!r} contradicts {other!r}")

    def __reduce__(self) -> tuple[type[ProductFact], tuple[SelfProduct, ProductResult]]:
        return type(self), (self.self_product, self.result)


class ProductRelation(Trait):
    """Conditional self-product monomial for a positive-arity Extensor.

    If every referenced input has a scalar self-product in this family, the
    output's category is ``factor`` times their categories. Repeated slots
    are powers, not redundant requirements. Different families coexist.
    ``nonzero_inputs`` strengthens the premise to nonzero self-products.
    """

    __slots__ = ()

    def __init__(
        self,
        self_product: SelfProduct,
        factor: ProductResult,
        slots: Iterable[int],
        nonzero_inputs: bool = False,
    ) -> None:
        if not isinstance(self_product, SelfProduct):
            raise TypeError("ProductRelation requires a SelfProduct")
        if not isinstance(factor, ProductResult):
            raise TypeError("ProductRelation factor must be a ProductResult")
        try:
            normalized_slots = tuple(slots)
        except TypeError as error:
            raise TypeError("ProductRelation slots must be iterable") from error
        if not normalized_slots:
            raise ValueError("ProductRelation requires at least one input slot")
        if any(
            isinstance(slot, bool) or not isinstance(slot, int) or slot < 0
            for slot in normalized_slots
        ):
            raise TypeError(
                "ProductRelation slots must be non-negative integers"
            )

        super().__init__(
            "ProductRelation", valid_arities=None,
            parameters=(self_product, factor, tuple(sorted(normalized_slots)), nonzero_inputs),
        )

    @property
    def self_product(self) -> SelfProduct:
        return self._parameters[0]

    @property
    def factor(self) -> ProductResult:
        return self._parameters[1]

    @property
    def slots(self) -> tuple[int, ...]:
        return self._parameters[2]

    @property
    def nonzero_inputs(self) -> bool:
        return self._parameters[3]

    def validate_arity(self, arity: int) -> None:
        if isinstance(arity, bool) or not isinstance(arity, int) or arity < 1:
            raise ValueError(
                "ProductRelation is valid only for positive-arity GATypes"
            )
        invalid = tuple(slot for slot in self.slots if slot >= arity)
        if invalid:
            raise ValueError(
                "ProductRelation references input slots outside arity "
                f"{arity}: {invalid!r}"
            )

    def implied_traits(self) -> tuple[Trait, ...]:
        return tuple(
            ProductRelation(self.self_product, factor, self.slots, self.nonzero_inputs)
            for factor in ProductResult
            if factor is not self.factor and _result_entails(self.factor, factor)
        )

    def validate_peers(self, traits: frozenset[Trait]) -> None:
        for other in traits:
            if not isinstance(other, ProductRelation):
                continue
            if other.self_product != self.self_product:
                continue
            if other.slots != self.slots:
                raise ValueError(
                    "at most one ProductRelation slot signature per self-product"
                )
            if _results_conflict(self.factor, other.factor):
                raise ValueError(f"{self!r} contradicts {other!r}")

    def __reduce__(
        self,
    ) -> tuple[type[ProductRelation], tuple[SelfProduct, ProductResult, tuple[int, ...], bool]]:
        return type(self), (self.self_product, self.factor, self.slots, self.nonzero_inputs)

    def __repr__(self) -> str:
        return (
            f"ProductRelation({self.self_product!r}, "
            f"factor=ProductResult.{self.factor.name}, slots={self.slots!r}"
            f"{', nonzero_inputs=True' if self.nonzero_inputs else ''})"
        )


class VersorProduct(Trait):
    """The output is a versor when every referenced input is a versor."""

    __slots__ = ()

    def __init__(self, slots: Iterable[int]) -> None:
        try:
            normalized_slots = tuple(slots)
        except TypeError as error:
            raise TypeError("VersorProduct slots must be iterable") from error
        if not normalized_slots:
            raise ValueError("VersorProduct requires at least one input slot")
        if any(
            isinstance(slot, bool) or not isinstance(slot, int) or slot < 0
            for slot in normalized_slots
        ):
            raise TypeError("VersorProduct slots must be non-negative integers")

        # Unlike product powers, requiring the same input fact twice adds no
        # information. Distinct residual variables retain distinct slot ids.
        super().__init__(
            "VersorProduct", valid_arities=None,
            parameters=(tuple(sorted(set(normalized_slots))),),
        )

    @property
    def slots(self) -> tuple[int, ...]:
        return self._parameters[0]

    def validate_arity(self, arity: int) -> None:
        if isinstance(arity, bool) or not isinstance(arity, int) or arity < 1:
            raise ValueError("VersorProduct is valid only for positive-arity GATypes")
        invalid = tuple(slot for slot in self.slots if slot >= arity)
        if invalid:
            raise ValueError(
                "VersorProduct references input slots outside arity "
                f"{arity}: {invalid!r}"
            )

    def validate_peers(self, traits: frozenset[Trait]) -> None:
        if any(isinstance(t, VersorProduct) and t != self for t in traits):
            raise ValueError("a TraitSet may carry at most one VersorProduct relation")

    def __reduce__(self) -> tuple[type[VersorProduct], tuple[tuple[int, ...]]]:
        return type(self), (self.slots,)

    def __repr__(self) -> str:
        return f"VersorProduct(slots={self.slots!r})"


Versor = Trait("Versor", valid_arities=(0,))
ReverseProductScalar = ProductFact(ReverseProduct, ProductResult.SCALAR)
ReverseProductZero = ProductFact(ReverseProduct, ProductResult.ZERO)
ReverseProductNonzero = ProductFact(ReverseProduct, ProductResult.NONZERO)
ReverseProductOne = ProductFact(ReverseProduct, ProductResult.ONE)
CliffordConjugateProductScalar = ProductFact(CliffordConjugateProduct, ProductResult.SCALAR)
CliffordConjugateProductZero = ProductFact(CliffordConjugateProduct, ProductResult.ZERO)
CliffordConjugateProductNonzero = ProductFact(CliffordConjugateProduct, ProductResult.NONZERO)
CliffordConjugateProductOne = ProductFact(CliffordConjugateProduct, ProductResult.ONE)
CoefficientOrthogonal = Trait("CoefficientOrthogonal", valid_arities=(1,))
# The unrestricted trilinear expression L * P * reverse(R). Equal bound
# versors in slots 0 and 2 permit the sandwich law during specialization.
Sandwich = Trait("Sandwich", valid_arities=(3,))
Identity = Trait("Identity", valid_arities=(1,))

_BUILTIN_TRAITS = {
    (trait.name, trait.valid_arities, trait._parameters): trait
    for trait in (
        Versor,
        CoefficientOrthogonal,
        Sandwich,
    )
}

_DIRECT_IMPLICATIONS = {
    Versor: (ReverseProductNonzero,),
}


def _restore_trait(
    trait_type: type[Trait], name: str,
    valid_arities: tuple[int, ...], parameters: tuple[object, ...],
) -> Trait:
    key = name, frozenset(valid_arities), parameters
    if trait_type is Trait and key in _BUILTIN_TRAITS:
        return _BUILTIN_TRAITS[key]
    trait = object.__new__(trait_type)
    Trait.__init__(trait, name, valid_arities=valid_arities, parameters=parameters)
    return trait


def _trait_closure(traits: Iterable[Trait]) -> frozenset[Trait]:
    closure = set(traits)
    pending = list(closure)
    while pending:
        trait = pending.pop()
        for implied in trait.implied_traits():
            if not isinstance(implied, Trait):
                raise TypeError("trait implications must be Trait values")
            if implied not in closure:
                closure.add(implied)
                pending.append(implied)
    return frozenset(closure)


class TraitSet:
    """An immutable canonical conjunction of certified trait facts."""

    __slots__ = ("_traits", "_hash")

    def __init__(self, traits: Iterable[Trait] = ()) -> None:
        if isinstance(traits, TraitSet):
            raw_traits = traits.traits
        else:
            try:
                raw_traits = tuple(traits)
            except TypeError as error:
                raise TypeError("TraitSet facts must be iterable") from error
        if not all(isinstance(trait, Trait) for trait in raw_traits):
            raise TypeError("TraitSet facts must be Trait values")

        unique = frozenset(raw_traits)
        closure = _trait_closure(unique)
        for trait in closure:
            trait.validate_peers(closure)

        # Store only maximal facts. Entailed redundancies do not produce a
        # second GAType key: {One, Nonzero, Scalar} canonicalizes to {One}.
        # Include implied facts when choosing representatives, so mutually
        # implying names canonicalize identically even if only one was given.
        implications = {trait: _trait_closure((trait,)) for trait in closure}
        irredundant = (
            trait
            for trait in closure
            if not any(
                other != trait
                and trait in implications[other]
                and (
                    other not in implications[trait]
                    or other.sort_key < trait.sort_key
                )
                for other in closure
            )
        )
        normalized = tuple(sorted(irredundant, key=lambda trait: trait.sort_key))
        object.__setattr__(self, "_traits", normalized)
        object.__setattr__(self, "_hash", hash((type(self), normalized)))

    @property
    def traits(self) -> Tuple[Trait, ...]:
        return self._traits

    def __len__(self) -> int:
        return len(self.traits)

    def __iter__(self) -> Iterator[Trait]:
        return iter(self.traits)

    def __contains__(self, trait: object) -> bool:
        return trait in self.traits

    @property
    def closure(self) -> Tuple[Trait, ...]:
        """All known facts, including implications, in deterministic order."""

        return tuple(sorted(_trait_closure(self.traits), key=lambda trait: trait.sort_key))

    def entails(self, required: Trait | "TraitSet" | Iterable[Trait]) -> bool:
        """Whether these known facts prove every required fact."""

        if isinstance(required, Trait):
            requirements = (required,)
        elif isinstance(required, TraitSet):
            requirements = required.traits
        else:
            requirements = TraitSet(required).traits
        known = frozenset(self.closure)
        return all(trait in known for trait in requirements)

    def refines(self, other: "TraitSet") -> bool:
        """Whether this conjunction is at least as informative as ``other``."""

        return isinstance(other, TraitSet) and self.entails(other)

    def is_compatible_with(self, other: "TraitSet") -> bool:
        """Whether both sets of positive facts can hold simultaneously."""

        if not isinstance(other, TraitSet):
            return False
        try:
            TraitSet(self.traits + other.traits)
        except ValueError:
            return False
        return True

    def __bool__(self) -> bool:
        return bool(self.traits)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("TraitSet is immutable")

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, TraitSet):
            return NotImplemented
        return self.traits == other.traits

    def __hash__(self) -> int:
        return self._hash

    def __reduce__(self) -> tuple[type[TraitSet], tuple[tuple[Trait, ...]]]:
        return type(self), (self.traits,)

    def __repr__(self) -> str:
        return f"TraitSet({self.traits!r})" if self.traits else "TraitSet()"


EMPTY_TRAITS = TraitSet()
ROTOR_TRAITS = TraitSet((Versor, ReverseProductOne))


def structurally_implied_traits(subspaces: Iterable[SubSpace]) -> TraitSet:
    """Return value facts universally proved by the structural carrier."""

    axes = tuple(subspaces)
    if len(axes) == 1:
        from numga.algebra.self_product import symmetric_product_support

        # Support proves scalar-valuedness or identically zero, never nonzero
        # or unit coefficients.
        # Pairwise cancellation also captures vectors and small even carriers.
        facts: list[Trait] = []
        for family in (ReverseProduct, CliffordConjugateProduct):
            support = symmetric_product_support(axes[0], family.transform)
            if support <= {0}:
                result = ProductResult.SCALAR if support else ProductResult.ZERO
                facts.append(ProductFact(family, result))
        return TraitSet(facts)
    return EMPTY_TRAITS


def normalize_explicit_traits(
    subspaces: Iterable[SubSpace],
    traits: TraitSet | Iterable[Trait],
) -> TraitSet:
    """Remove explicit facts already implied universally by the carrier."""

    axes = tuple(subspaces)
    explicit = traits if isinstance(traits, TraitSet) else TraitSet(traits)
    structural = structurally_implied_traits(axes)
    normalized = TraitSet(
        trait for trait in explicit if not structural.entails(trait)
    )
    return normalized if normalized else EMPTY_TRAITS


def validate_traits_for_subspaces(
    subspaces: Iterable[SubSpace],
    traits: TraitSet,
) -> None:
    """Validate trait contracts that constrain structural axes."""

    axes = tuple(subspaces)
    if traits.entails(CoefficientOrthogonal) and len(axes[0]) != len(axes[1]):
        raise ValueError(
            "CoefficientOrthogonal requires equally sized output and input axes"
        )
