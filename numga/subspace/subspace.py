"""Each structural axis of an `Extensor` is described by a `SubSpace`, an immutable sequence of
distinct canonical blade masks paired with `+1` or `-1` basis-blade signs. Separating coordinate
layout from algebraic support lets an algebra adopt human-readable basis conventions such as
right-handed cyclic bivectors `("yz", "zx", "xy")` without altering canonical integer bit-mask
arithmetic underneath; `"zx"` stores canonical mask `xz` with sign `-1`.

Two levels of identity coexist on subspaces. `SubSpace` equality (`==`) and hashing compare the
`Algebra` instance (`is`), exact mask ordering, and per-blade signs, because two extensors can
share a raw coefficient array without reordering or sign flips only when their axis layouts match
identically. `SupportKey`, `same_support`, and `support_is_subset_of` compare only the unordered
set of canonical masks on the same `Algebra` instance, governing mathematical subspace containment
and support-indexed operator caches across layouts.

Set-like operations distinguish layout-preserving filtering from default-ordered support
construction. `restrict`, `intersection`, `difference`, `degenerate`, and `nondegenerate` filter an
existing `SubSpace` while preserving the surviving masks' order and signs, whereas `union` (`+`)
and `complement` build new supports through `algebra.subspace.from_masks` in the algebra's default
layout. `SubSpace` itself holds no flyweight pool so direct construction obeys the same structural
equality contract as `SubSpaceFactory` interning. Signed canonical blade layouts and their role in
structural kernel axes are described in `docs/internals.md` (section 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from operator import index
from types import NotImplementedType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numga.algebra import Algebra


def _normalize_masks(algebra: Algebra, masks: Iterable[int]) -> tuple[int, ...]:
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

    # Canonical layout orders blades by ascending grade, then ascending bit mask.
    normalized.sort(key=lambda mask: (algebra.grade(mask), mask))
    return tuple(normalized)


class SupportKey:
    """Unordered canonical blade masks paired with `Algebra` object identity (`is`) form a hashable
    support key independent of blade ordering and signs."""

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
    def masks(self) -> tuple[int, ...]:
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
    """An ordered tuple of distinct canonical blade masks and `+1`/`-1` orientation signs defines
    the coordinate basis of one structural `Extensor` axis."""

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
    def masks(self) -> tuple[int, ...]:
        """Canonical integer blade masks appear in the exact coordinate order of this axis."""
        return self._masks

    @property
    def signs(self) -> tuple[int, ...]:
        return self._signs

    @property
    @lru_cache(maxsize=None)
    def canonical(self) -> SubSpace:
        """Sorting this support by `(grade, mask)` with all `+1` signs strips any custom ordering
        or blade sign conventions."""
        masks = _normalize_masks(self.algebra, self.masks)
        return self.algebra.subspace.from_layout(masks, (1,) * len(masks))

    def restrict(self, masks: Iterable[int]) -> SubSpace:
        """Intersecting with `masks` keeps surviving blades in their existing axis order and
        orientation signs."""
        support = frozenset(masks)
        entries = [(mask, sign) for mask, sign in zip(self.masks, self.signs) if mask in support]
        return self.algebra.subspace.from_layout(
            (mask for mask, _ in entries), (sign for _, sign in entries),
        )

    @property
    def support_key(self) -> SupportKey:
        """All signed reorderings of the same blade set share one `SupportKey`."""
        return self._support_key

    def same_support(self, other: object) -> bool:
        """Two subspaces share support when they belong to the same `Algebra` instance and span
        identical canonical masks regardless of order or signs."""
        return (
            isinstance(other, SubSpace)
            and self.algebra is other.algebra
            and self._mask_set == other._mask_set
        )

    def support_is_subset_of(self, other: object) -> bool:
        """Containment holds when `other` belongs to the same `Algebra` instance and includes every
        canonical mask of `self`."""
        return (
            isinstance(other, SubSpace)
            and self.algebra is other.algebra
            and self._mask_set.issubset(other._mask_set)
        )

    def union(self, other: SubSpace) -> SubSpace:
        """Joining two supports builds the `SubSpace` spanning blades in either axis, ordered
        according to `algebra.subspace`."""
        if not isinstance(other, SubSpace):
            raise TypeError(
                "SubSpace union requires another SubSpace; "
                f"got {type(other).__name__}"
            )
        if self.algebra is not other.algebra:
            raise ValueError("cannot union SubSpaces from different algebras")
        right_only = tuple(mask for mask in other.masks if mask not in self._mask_set)
        return self.algebra.subspace.from_masks(self.masks + right_only)

    def intersection(self, other: SubSpace) -> SubSpace:
        """Blades present in both subspaces survive in the order and signs of `self`."""
        if self.algebra is not other.algebra:
            raise ValueError("cannot intersect SubSpaces from different algebras")
        return self.restrict(other.masks)

    def difference(self, other: SubSpace) -> SubSpace:
        """Removing every canonical mask of `other` keeps the remaining blades in the order and
        signs of `self`."""
        if self.algebra is not other.algebra:
            raise ValueError("cannot subtract SubSpaces from different algebras")
        return self.restrict(mask for mask in self.masks if mask not in other._mask_set)

    def complement(self) -> SubSpace:
        """Toggling every generator bit against the pseudoscalar maps each blade to its
        complementary grade in `algebra.subspace` order."""
        return self.algebra.subspace.from_masks(self.algebra.complement(self.masks).tolist())

    def degenerate(self) -> SubSpace:
        """Filtering to blades that contain at least one null generator preserves the order and
        signs of `self`."""
        return self.restrict(mask for mask in self.masks if mask & self.algebra.degenerate_mask)

    def nondegenerate(self) -> SubSpace:
        """Filtering to blades with no null generator preserves the order and signs of `self`."""
        return self.restrict(mask for mask in self.masks if not mask & self.algebra.degenerate_mask)

    def __contains__(self, mask: int) -> bool:
        return mask in self._mask_set

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

    def __add__(self, other: object) -> SubSpace | NotImplementedType:
        """Adding two `SubSpace` instances forms the union of their supports via `union`."""
        if isinstance(other, SubSpace):
            return self.union(other)
        return NotImplemented

    def __reduce__(self) -> tuple:
        return type(self), (self.algebra, self.masks, self.signs)

    @property
    def blade_names(self) -> tuple[str, ...]:
        """Signed blade names follow axis coordinate order and round-trip through
        `algebra.subspace.from_blades`."""
        return tuple(self.algebra.oriented_blade_name(mask, sign) for mask, sign in zip(self.masks, self.signs))

    @property
    def type_name(self) -> str:
        """Display formatting names the first matching standard support (such as `"bivector"`), or
        `"in <name>"` for the smallest named support containing `self`."""
        named = self.algebra.subspace.named()
        exact = next((name for name, space in named if space.same_support(self)), None)
        containing = min((len(space), i, name) for i, (name, space) in enumerate(named)
                         if self.support_is_subset_of(space))
        return exact or "in " + containing[2]

    def __repr__(self) -> str:
        return f"SubSpace({self.type_name}: {' '.join(self.blade_names)})"
