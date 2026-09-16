"""Immutable coefficient axes: ordered bit masks with orientation signs."""

from __future__ import annotations

from numbers import Number

from functools import lru_cache
from operator import index
from types import NotImplementedType
from typing import TYPE_CHECKING, Iterable, Iterator, Tuple

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.extensor import Extensor
    from numga.gatype import GAType


def _normalize_masks(algebra: Algebra, masks: Iterable[int]) -> Tuple[int, ...]:
    """Validate masks and return the epoch-one canonical coordinate order."""

    upper_bound = 1 << algebra.dimension
    normalized = []
    seen = set()
    for raw_mask in masks:
        if isinstance(raw_mask, bool):
            raise TypeError("boolean values are not blade masks")
        mask = index(raw_mask)
        if mask < 0 or mask >= upper_bound:
            raise ValueError(
                f"blade mask {mask} is outside the algebra's valid range "
                f"[0, {upper_bound})"
            )
        if mask in seen:
            raise ValueError(f"duplicate canonical blade mask {mask}")
        seen.add(mask)
        normalized.append(mask)

    normalized.sort(key=lambda mask: (algebra.grade(mask), mask))
    return tuple(normalized)


class SupportKey:
    """Hashable, layout-independent identity of one algebraic blade support.

    Algebra ownership intentionally uses object identity.  This prevents two
    separately configured algebra instances from sharing structural caches even
    if they happen to compare equal.
    """

    __slots__ = ("_algebra", "_masks", "_hash")

    def __init__(self, algebra: Algebra, masks: Iterable[int]) -> None:
        canonical_masks = tuple(sorted(masks))
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_masks", canonical_masks)
        object.__setattr__(self, "_hash", hash((id(algebra), canonical_masks)))

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    def masks(self) -> Tuple[int, ...]:
        return self._masks

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("SupportKey is immutable")

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, SupportKey):
            return NotImplemented
        return self.algebra is other.algebra and self.masks == other.masks

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"SupportKey(masks={self.masks!r})"


class SubSpace:
    """One immutable coefficient axis described by ordered signed bit blades.

    The value class deliberately contains no flyweight machinery.  A bound
    :class:`SubSpaceFactory` owns canonical construction, while structural
    equality and hashing keep direct construction correct as well.
    """

    __slots__ = (
        "_algebra",
        "_masks",
        "_signs",
        "_mask_set",
        "_support_key",
        "_hash",
    )

    def __init__(
        self, algebra: Algebra, masks: Iterable[int], signs: Iterable[int] | None = None,
    ) -> None:
        masks = tuple(masks)
        canonical = _normalize_masks(algebra, masks)
        normalized_masks = canonical if signs is None else masks
        signs = (1,) * len(masks) if signs is None else tuple(signs)
        if len(signs) != len(masks) or any(sign not in (-1, 1) for sign in signs):
            raise ValueError("each basis blade needs an orientation sign of +1 or -1")
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_masks", normalized_masks)
        object.__setattr__(self, "_signs", signs)
        object.__setattr__(self, "_mask_set", frozenset(normalized_masks))
        object.__setattr__(
            self,
            "_support_key",
            SupportKey(algebra, normalized_masks),
        )
        object.__setattr__(
            self,
            "_hash",
            hash((type(self), id(algebra), normalized_masks, signs)),
        )

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    def masks(self) -> Tuple[int, ...]:
        """Canonical bit masks in coefficient-axis order."""

        return self._masks

    @property
    def signs(self) -> tuple[int, ...]:
        return self._signs

    @property
    @lru_cache(maxsize=None)
    def canonical(self) -> SubSpace:
        """This support in unsigned grade/mask order, independent of defaults."""

        masks = _normalize_masks(self.algebra, self.masks)
        return self.algebra.subspace.from_layout(masks, (1,) * len(masks))

    def restrict(self, masks: Iterable[int]) -> SubSpace:
        """Filter support without changing surviving coordinates."""

        support = frozenset(masks)
        entries = [(mask, sign) for mask, sign in zip(self.masks, self.signs) if mask in support]
        return self.algebra.subspace.from_layout(
            (mask for mask, _ in entries), (sign for _, sign in entries),
        )

    @property
    def support_key(self) -> SupportKey:
        """A hashable support identity independent of axis layout."""

        return self._support_key

    @property
    @lru_cache(maxsize=None)
    def gatype(self) -> "GAType":
        """The canonical empty-trait nullary GAType represented by this axis."""

        return self.algebra.gatype(self)

    def same_support(self, other: object) -> bool:
        """Whether both axes belong to one algebra and span the same masks."""

        return (
            isinstance(other, SubSpace)
            and self.algebra is other.algebra
            and self._mask_set == other._mask_set
        )

    def support_is_subset_of(self, other: object) -> bool:
        """Whether this axis's canonical-mask support is contained in another."""

        return (
            isinstance(other, SubSpace)
            and self.algebra is other.algebra
            and self._mask_set.issubset(other._mask_set)
        )

    def union(self, other: "SubSpace") -> "SubSpace":
        """Return the canonical SubSpace spanning either operand's support."""

        if not isinstance(other, SubSpace):
            raise TypeError(
                "SubSpace union requires another SubSpace; "
                f"got {type(other).__name__}"
            )
        if self.algebra is not other.algebra:
            raise ValueError("cannot union SubSpaces from different algebras")
        right_only = tuple(mask for mask in other.masks if mask not in self._mask_set)
        return self.algebra.subspace.from_masks(self.masks + right_only)

    def intersection(self, other: "SubSpace") -> "SubSpace":
        """Intersect support, preserving this axis's order and signs."""

        if self.algebra is not other.algebra:
            raise ValueError("cannot intersect SubSpaces from different algebras")
        return self.restrict(other.masks)

    def __len__(self) -> int:
        return len(self.masks)

    def __iter__(self) -> Iterator[int]:
        return iter(self.masks)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("SubSpace is immutable")

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, SubSpace):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.algebra is other.algebra
            and self.masks == other.masks
            and self.signs == other.signs
        )

    def __hash__(self) -> int:
        return self._hash

    def __add__(self, other: object) -> SubSpace | Extensor | NotImplementedType:
        from numga.extensor import Extensor

        if isinstance(other, SubSpace):
            return self.union(other)
        if isinstance(other, Extensor):
            return self.algebra.operator.identity(self) + other
        return NotImplemented

    def __sub__(self, other: object) -> Extensor | NotImplementedType:
        from numga.extensor import Extensor

        if isinstance(other, Extensor):
            return self.algebra.operator.identity(self) - other
        return NotImplemented

    def __neg__(self) -> Extensor:
        return -self.algebra.operator.identity(self)

    def __mul__(self, other: object) -> Extensor | NotImplementedType:
        from numga.expression import geometric_product, is_expression_operand

        if isinstance(other, Number):
            return self.algebra.operator.identity(self) * other
        if not is_expression_operand(other):
            return NotImplemented
        return geometric_product(self, other)

    def __rmul__(self, other: object) -> Extensor | NotImplementedType:
        from numga.expression import geometric_product, is_expression_operand

        if isinstance(other, Number):
            return self.algebra.operator.identity(self) * other
        if not is_expression_operand(other):
            return NotImplemented
        return geometric_product(other, self)

    def dual(self) -> Extensor:
        """The right-Hodge dual as a map with this space in its open slot."""
        return self.algebra.operator.dual(self)

    def dual_inverse(self) -> Extensor:
        return self.algebra.operator.dual_inverse(self)

    def wedge(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import wedge

        return wedge(self, other)

    def inner(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import inner

        return inner(self, other)

    def commutator(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import commutator

        return commutator(self, other)

    def regressive(self, other: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import regressive

        return regressive(self, other)

    def sandwich(self, passenger: SubSpace | GAType | Extensor) -> Extensor:
        from numga.expression import sandwich

        return sandwich(self, passenger)

    def __xor__(self, other: SubSpace | GAType | Extensor) -> Extensor:
        return self.wedge(other)

    def __and__(self, other: SubSpace | GAType | Extensor) -> Extensor:
        return self.regressive(other)

    def __or__(self, other: SubSpace | GAType | Extensor) -> Extensor:
        return self.inner(other)

    def __rshift__(self, other: SubSpace | GAType | Extensor) -> Extensor:
        return self.sandwich(other)

    def __reduce__(self) -> tuple[type[SubSpace], tuple[Algebra, tuple[int, ...], tuple[int, ...]]]:
        return type(self), (self.algebra, self.masks, self.signs)

    def __repr__(self) -> str:
        return f"SubSpace(algebra={self.algebra!r}, masks={self.masks!r}, signs={self.signs!r})"
