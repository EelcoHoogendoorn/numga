"""Specificity-aware dispatch over complete GAType values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar
import warnings

from .gatype import GAType
from .pattern import (
    DispatchPattern,
    GATypePattern,
    pattern_matches,
    pattern_refines,
    patterns_overlap,
)
from .traits import Trait, TraitSet

if TYPE_CHECKING:
    from numga.algebra import Algebra

_Implementation = TypeVar("_Implementation", bound=Callable[..., Any])


class AmbiguousGATypeDispatchWarning(UserWarning):
    """Two incomparable registrations overlap and use declared precedence."""


@dataclass(frozen=True, slots=True)
class _Registration:
    patterns: tuple[DispatchPattern, ...]
    implementation: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class _PredicateRegistration:
    predicate: Callable[..., object]
    implementation: Callable[..., Any]


class GATypeDispatch:
    """Declarative whole-GAType dispatch with an explicit low-level tier.

    A dispatcher may be scoped to one concrete algebra or constructed with
    ``algebra=None`` for cross-algebra semantic dispatch. Declarative patterns
    use true specificity. Opaque predicate registrations are presumed to be
    low-level specializations: matching predicates take priority over every
    declarative registration and use their own list order as the tie-breaker.
    """

    __slots__ = (
        "_name",
        "_algebra",
        "_operand_count",
        "_registrations",
        "_predicates",
        "_cache",
    )

    def __init__(
        self,
        name: str,
        algebra: Algebra | None,
        operand_count: int,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("a dispatcher name must be a non-empty string")
        if (
            isinstance(operand_count, bool)
            or not isinstance(operand_count, int)
            or operand_count < 1
        ):
            raise TypeError("dispatcher operand_count must be a positive integer")
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_operand_count", operand_count)
        object.__setattr__(self, "_registrations", [])
        object.__setattr__(self, "_predicates", [])
        object.__setattr__(self, "_cache", {})

    @property
    def name(self) -> str:
        return self._name

    @property
    def algebra(self) -> Algebra | None:
        return self._algebra

    @property
    def operand_count(self) -> int:
        return self._operand_count

    @property
    def registrations(self) -> tuple[tuple[DispatchPattern, ...], ...]:
        return tuple(entry.patterns for entry in self._registrations)

    @property
    def predicates(self) -> tuple[Callable[..., object], ...]:
        return tuple(entry.predicate for entry in self._predicates)

    @property
    def resolution_cache_size(self) -> int:
        return len(self._cache)

    def register(
        self,
        *patterns: DispatchPattern | Trait | TraitSet | Callable[..., object],
        precedence: str | None = None,
        position: int | None = None,
    ) -> Callable[[_Implementation], _Implementation]:
        """Register a declarative signature or an opaque GAType predicate.

        A sole callable is an escape hatch whose arguments are the complete
        concrete GATypes, not Extensor values. Predicate registrations are
        always tried before declarative ones; ``position`` controls order only
        within that low-level tier.
        """

        if precedence not in {None, "declaration"}:
            raise ValueError("precedence must be None or 'declaration'")
        if len(patterns) == 1 and callable(patterns[0]):
            return self._register_predicate(patterns[0], position=position)
        if position is not None:
            raise TypeError("position is supported only for callable predicates")

        normalized = self._validate_registration_signature(patterns)
        for earlier in self._registrations:
            if _signature_equivalent(earlier.patterns, normalized):
                raise ValueError(
                    f"duplicate {self.name!r} GAType registration {normalized!r}"
                )
            if _signature_strictly_refines(normalized, earlier.patterns):
                raise ValueError(
                    f"{self.name!r} specialization {normalized!r} appears after "
                    f"the more-general registration {earlier.patterns!r}"
                )
            if (
                not _signature_strictly_refines(earlier.patterns, normalized)
                and _signatures_overlap(earlier.patterns, normalized)
                and precedence is None
            ):
                warnings.warn(
                    f"incomparable {self.name!r} registrations "
                    f"{earlier.patterns!r} and {normalized!r} overlap; "
                    "declaration order is the tie-breaker",
                    AmbiguousGATypeDispatchWarning,
                    stacklevel=2,
                )

        def decorate(implementation: _Implementation) -> _Implementation:
            if not callable(implementation):
                raise TypeError("a dispatch implementation must be callable")
            self._registrations.append(_Registration(normalized, implementation))
            self._cache.clear()
            return implementation

        return decorate

    def _register_predicate(
        self,
        predicate: Callable[..., object],
        *,
        position: int | None,
    ) -> Callable[[_Implementation], _Implementation]:
        if position is not None and (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
            or position > len(self._predicates)
        ):
            raise ValueError(
                "predicate position must be an insertion index in "
                f"[0, {len(self._predicates)}]"
            )

        def decorate(implementation: _Implementation) -> _Implementation:
            if not callable(implementation):
                raise TypeError("a dispatch implementation must be callable")
            entry = _PredicateRegistration(predicate, implementation)
            if position is None:
                self._predicates.append(entry)
            else:
                self._predicates.insert(position, entry)
            self._cache.clear()
            return implementation

        return decorate

    def resolve(self, *actual_gatypes: GAType) -> Callable[..., Any]:
        """Return the highest-priority implementation matching actual types."""

        try:
            return self._cache[actual_gatypes]
        except (KeyError, TypeError):
            return self._resolve_uncached(actual_gatypes)

    def _resolve_uncached(
        self,
        actual_gatypes: tuple[GAType, ...],
    ) -> Callable[..., Any]:
        """Validate and select once for a previously unseen signature."""

        actual = self._validate_actual_signature(actual_gatypes)

        for entry in self._predicates:
            if entry.predicate(*actual):
                self._cache[actual] = entry.implementation
                return entry.implementation

        matches = tuple(
            entry
            for entry in self._registrations
            if _signature_matches(actual, entry.patterns)
        )
        if not matches:
            raise LookupError(
                f"no {self.name!r} implementation matches GATypes {actual!r}"
            )

        maximal = tuple(
            candidate
            for candidate in matches
            if not any(
                other is not candidate
                and _signature_strictly_refines(
                    other.patterns,
                    candidate.patterns,
                )
                for other in matches
            )
        )
        implementation = maximal[0].implementation
        self._cache[actual] = implementation
        return implementation

    def __call__(self, *arguments: Any, **kwargs: Any) -> Any:
        operands = arguments[: self._operand_count]
        try:
            actual = tuple(operand.gatype for operand in operands)
            implementation = self._cache[actual]
        except (AttributeError, KeyError, TypeError):
            # Only valid, fixed-arity signatures enter the cache. A hit already
            # establishes these invariants without repeating type analysis.
            if len(arguments) < self._operand_count:
                raise TypeError(
                    f"{self.name!r} dispatch expects at least "
                    f"{self._operand_count} positional operands, got {len(arguments)}"
                ) from None
            actual = tuple(_operand_gatype(operand) for operand in operands)
            implementation = self._resolve_uncached(actual)
        # An implementation's own exceptions must never trigger resolution.
        return implementation(*arguments, **kwargs)

    def _validate_registration_signature(
        self,
        patterns: tuple[object, ...],
    ) -> tuple[DispatchPattern, ...]:
        if len(patterns) != self.operand_count:
            raise TypeError(
                f"{self.name!r} registration requires {self.operand_count} "
                f"patterns, got {len(patterns)}"
            )
        normalized = tuple(_normalize_pattern(pattern) for pattern in patterns)
        concrete = tuple(
            pattern for pattern in normalized if isinstance(pattern, GAType)
        )
        if self.algebra is not None and any(
            pattern.algebra is not self.algebra for pattern in concrete
        ):
            raise ValueError(
                f"every concrete {self.name!r} registration GAType must belong "
                "to its dispatcher algebra"
            )
        if concrete and any(
            pattern.algebra is not concrete[0].algebra for pattern in concrete[1:]
        ):
            raise ValueError(
                f"concrete {self.name!r} registration GATypes must belong to "
                "one algebra"
            )
        return normalized

    def _validate_actual_signature(
        self,
        actual: tuple[GAType, ...],
    ) -> tuple[GAType, ...]:
        if len(actual) != self.operand_count:
            raise TypeError(
                f"{self.name!r} actual requires {self.operand_count} GATypes, "
                f"got {len(actual)}"
            )
        if not all(isinstance(gatype, GAType) for gatype in actual):
            raise TypeError(f"every {self.name!r} actual entry must be a GAType")
        if self.algebra is not None and any(
            gatype.algebra is not self.algebra for gatype in actual
        ):
            raise ValueError(
                f"every {self.name!r} actual GAType must belong to its "
                "dispatcher algebra"
            )
        if actual and any(
            gatype.algebra is not actual[0].algebra for gatype in actual[1:]
        ):
            raise ValueError(
                f"every {self.name!r} actual GAType must belong to one algebra"
            )
        return actual


def _normalize_pattern(pattern: object) -> DispatchPattern:
    if isinstance(pattern, (GAType, GATypePattern)):
        return pattern
    if isinstance(pattern, Trait):
        traits = TraitSet((pattern,))
    elif isinstance(pattern, TraitSet):
        traits = pattern
    else:
        raise TypeError(
            "GAType registrations require a GAType, GATypePattern, Trait, "
            "TraitSet, or one callable predicate"
        )

    possible_arities: set[int] | None = None
    for trait in traits:
        if trait.valid_arities is None:
            raise TypeError(
                "a dynamic-arity Trait registration requires an explicit "
                "GATypePattern arity"
            )
        valid = set(trait.valid_arities)
        possible_arities = valid if possible_arities is None else possible_arities & valid
    if possible_arities is None or len(possible_arities) != 1:
        raise TypeError(
            "a Trait or TraitSet registration must imply one Extensor arity; "
            "use GATypePattern for an explicit arity"
        )
    return GATypePattern(possible_arities.pop(), traits)


def _operand_gatype(operand: object) -> GAType:
    gatype = getattr(operand, "gatype", None)
    if not isinstance(gatype, GAType):
        raise TypeError("GAType dispatch operands must expose one complete GAType")
    return gatype


def _signature_matches(
    actual: tuple[GAType, ...],
    pattern: tuple[DispatchPattern, ...],
) -> bool:
    return all(pattern_matches(left, right) for left, right in zip(actual, pattern))


def _signature_refines(
    left: tuple[DispatchPattern, ...],
    right: tuple[DispatchPattern, ...],
) -> bool:
    return all(pattern_refines(first, second) for first, second in zip(left, right))


def _signature_equivalent(
    left: tuple[DispatchPattern, ...],
    right: tuple[DispatchPattern, ...],
) -> bool:
    return _signature_refines(left, right) and _signature_refines(right, left)


def _signature_strictly_refines(
    left: tuple[DispatchPattern, ...],
    right: tuple[DispatchPattern, ...],
) -> bool:
    return _signature_refines(left, right) and not _signature_refines(right, left)


def _signatures_overlap(
    left: tuple[DispatchPattern, ...],
    right: tuple[DispatchPattern, ...],
) -> bool:
    return all(patterns_overlap(first, second) for first, second in zip(left, right))
