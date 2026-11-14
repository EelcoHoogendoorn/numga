"""Declarative dispatch signatures in `GATypeDispatch` combine two pattern kinds under
`DispatchPattern`: algebra-independent `GATypePattern` specifications and single-algebra `GAType`
values. `GATypePattern` fixes an extensor's `arity` and required algebraic `TraitSet` while omitting
blade subspaces, so an algorithm that depends only on arity and traits (such as versor inversion via
the reverse product or trace on unary maps) registers once across every algebra and layout.

Concrete `GAType` patterns belong to one `Algebra` and match any subtype whose blade support,
orientations, and entailed traits refine the pattern. Refinement across the two kinds is asymmetric
in `pattern_refines(specific, general)`: a concrete `GAType` refines a `GATypePattern` of equal
arity whenever it entails the pattern's traits, whereas an algebra-independent `GATypePattern`
ranges over all algebras and never refines a single-algebra `GAType`.

`Trait` and `TraitSet` values whose `valid_arities` imply a unique extensor arity normalize directly
into `GATypePattern`s during registration, while multi-arity traits use `GATypePattern.map(...)` or
`GATypePattern.nary(arity, ...)`. Two patterns overlap whenever some concrete `GAType` can satisfy
both, requiring equal arity, compatible traits, and compatible blade subspaces when both patterns
are concrete `GAType`s.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import NotImplementedType

from .gatype import GAType
from .traits import EMPTY_TRAITS, Trait, TraitSet


class GATypePattern:
    """Portable dispatch patterns match concrete `GAType`s across all algebras by `arity` and
    entailed `TraitSet` alone."""

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
    def map(cls, *traits: Trait) -> GATypePattern:
        """Unary linear maps (`arity == 1`) in any algebra match when their type entails `traits`."""
        return cls(1, traits)

    @classmethod
    def nary(cls, arity: int, *traits: Trait) -> GATypePattern:
        """Extensors of the given `arity` in any algebra match when their type entails `traits`."""
        return cls(arity, traits)

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def traits(self) -> TraitSet:
        return self._traits

    def matches(self, actual: object) -> bool:
        """A concrete `GAType` matches when it has equal arity and entails all required traits."""
        return (
            isinstance(actual, GAType)
            and actual.arity == self.arity
            and actual.entails(self.traits)
        )

    def refines(self, general: object) -> bool:
        """One portable pattern refines `general` when both have equal arity and `self.traits` is at
        least as restrictive."""
        return (
            isinstance(general, GATypePattern)
            and self.arity == general.arity
            and self.traits.refines(general.traits)
        )

    def strictly_refines(self, general: object) -> bool:
        return self.refines(general) and not general.refines(self)

    def overlaps(self, other: object) -> bool:
        """Two portable patterns of equal arity overlap when no required trait of one contradicts
        the other."""
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
    """A concrete `GAType` matches either a portable `GATypePattern` (by arity and traits) or a
    concrete `GAType` registration (by subtype refinement)."""
    if isinstance(pattern, GATypePattern):
        return pattern.matches(actual)
    return actual.refines(pattern)


def pattern_refines(specific: DispatchPattern, general: DispatchPattern) -> bool:
    """`specific` refines `general` when every concrete `GAType` matching `specific` also matches
    `general`."""
    if isinstance(specific, GATypePattern):
        # An algebra-independent pattern ranges over all algebras and never refines a
        # single-algebra GAType.
        return isinstance(general, GATypePattern) and specific.refines(general)
    if isinstance(general, GATypePattern):
        return specific.arity == general.arity and specific.entails(general.traits)
    return specific.refines(general)


def patterns_overlap(left: DispatchPattern, right: DispatchPattern) -> bool:
    """Two dispatch patterns overlap when some concrete `GAType` can satisfy both."""
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
