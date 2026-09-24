"""Algebra-independent patterns for whole-Extensor dispatch."""

from __future__ import annotations

from collections.abc import Iterable
from types import NotImplementedType

from .gatype import GAType
from .traits import EMPTY_TRAITS, Trait, TraitSet


class GATypePattern:
    """Required arity and facts for an Extensor in any algebra.

    A pattern deliberately has no SubSpaces.  The concrete :class:`GAType` on
    every Extensor remains the complete, layout-bearing record; omitted axes
    here mean that an implementation is independent of their representation.
    A registered GAType binds one algebra and matches by support, in any
    layout; an algorithm written against one coefficient layout registers a
    predicate comparing subspaces instead.
    """

    __slots__ = ("_arity", "_traits", "_hash")

    def __init__(
        self,
        arity: int,
        traits: TraitSet | Iterable[Trait] = EMPTY_TRAITS,
    ) -> None:
        if isinstance(arity, bool) or not isinstance(arity, int) or arity < 0:
            raise TypeError("a GATypePattern arity must be a non-negative integer")
        normalized_traits = traits if isinstance(traits, TraitSet) else TraitSet(traits)
        for trait in normalized_traits:
            trait.validate_arity(arity)
        object.__setattr__(self, "_arity", arity)
        object.__setattr__(self, "_traits", normalized_traits)
        object.__setattr__(
            self,
            "_hash",
            hash((type(self), arity, normalized_traits)),
        )

    @classmethod
    def map(cls, *traits: Trait) -> "GATypePattern":
        """Return an algebra-independent unary Extensor pattern."""

        return cls(1, traits)

    @classmethod
    def nary(cls, arity: int, *traits: Trait) -> "GATypePattern":
        """Return an algebra-independent pattern of explicit Extensor arity."""

        return cls(arity, traits)

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def traits(self) -> TraitSet:
        return self._traits

    def matches(self, actual: object) -> bool:
        """Whether one concrete GAType proves this pattern's requirements."""

        return (
            isinstance(actual, GAType)
            and actual.arity == self.arity
            and actual.entails(self.traits)
        )

    def refines(self, other: object) -> bool:
        """Whether this generic pattern is semantically more restrictive."""

        return (
            isinstance(other, GATypePattern)
            and self.arity == other.arity
            and self.traits.refines(other.traits)
        )

    def strictly_refines(self, other: object) -> bool:
        return self.refines(other) and not other.refines(self)

    def overlaps(self, other: object) -> bool:
        return (
            isinstance(other, GATypePattern)
            and self.arity == other.arity
            and self.traits.is_compatible_with(other.traits)
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("GATypePattern is immutable")

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, GATypePattern):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.arity == other.arity
            and self.traits == other.traits
        )

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"GATypePattern(arity={self.arity!r}, traits={self.traits!r})"


DispatchPattern = GAType | GATypePattern


def pattern_matches(actual: GAType, pattern: DispatchPattern) -> bool:
    if isinstance(pattern, GATypePattern):
        return pattern.matches(actual)
    return actual.refines(pattern)


def pattern_refines(left: DispatchPattern, right: DispatchPattern) -> bool:
    """Whether every type accepted by ``left`` is also accepted by ``right``."""

    if isinstance(left, GATypePattern):
        # An algebra-independent pattern can never be contained by a pattern
        # tied to one concrete algebra and set of carrier bounds.
        return isinstance(right, GATypePattern) and left.refines(right)
    if isinstance(right, GATypePattern):
        return left.arity == right.arity and left.entails(right.traits)
    return left.refines(right)


def patterns_overlap(left: DispatchPattern, right: DispatchPattern) -> bool:
    """Whether the two declarative patterns may accept one common GAType."""

    if isinstance(left, GATypePattern) and isinstance(right, GATypePattern):
        return left.overlaps(right)
    if isinstance(left, GATypePattern):
        generic, concrete = left, right
    elif isinstance(right, GATypePattern):
        generic, concrete = right, left
    else:
        return left.overlaps(right)
    return (
        generic.arity == concrete.arity
        and generic.traits.is_compatible_with(concrete.effective_traits)
    )
