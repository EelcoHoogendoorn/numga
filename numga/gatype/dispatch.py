"""Specificity-aware dispatch over complete GAType values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar
import warnings

import numga

from .errors import GATypeError
from .gatype import GAType
from .orientation import Orientation
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
    staged: bool = False


@dataclass(frozen=True, slots=True)
class _PredicateRegistration:
    predicate: Callable[..., object]
    implementation: Callable[..., Any]
    staged: bool = False


class GATypeDispatch:
    """Declarative whole-GAType dispatch with an explicit low-level tier.

    A dispatcher may be scoped to one concrete algebra or constructed with
    ``algebra=None`` for cross-algebra semantic dispatch. Declarative patterns
    use true specificity. Opaque predicate registrations are presumed to be
    low-level specializations: matching predicates take priority over every
    declarative registration and use their own list order as the tie-breaker.

    A staged registration is called once per signature with the actual GATypes, and returns the
    implementation for values of those types, which the cache keeps. A declared input orientation
    is checked, and a declared output orientation applied, once per signature as well.
    """

    __slots__ = (
        "_name",
        "_takes",
        "_algebra",
        "_operand_count",
        "_registrations",
        "_predicates",
        "_field_predicates",
        "_cache",
        "_input_orientation",
        "_output_orientation",
    )

    def __init__(
        self,
        name: str,
        algebra: Algebra | None,
        operand_count: int,
        *,
        takes: str,
        input_orientation: Orientation | None = None,
        output_orientation: Orientation | None = None,
    ) -> None:
        """`takes` says what the operands of the operation must be, in words: what a call whose types
        match no case is told."""
        if not isinstance(name, str) or not name:
            raise TypeError("a dispatcher name must be a non-empty string")
        if (
            isinstance(operand_count, bool)
            or not isinstance(operand_count, int)
            or operand_count < 1
        ):
            raise TypeError("dispatcher operand_count must be a positive integer")
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_takes", takes)
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_operand_count", operand_count)
        object.__setattr__(self, "_registrations", [])
        object.__setattr__(self, "_predicates", [])
        object.__setattr__(self, "_field_predicates", [])
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_input_orientation", input_orientation)
        object.__setattr__(self, "_output_orientation", output_orientation)

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

    def position_of(self, implementation: Callable[..., Any]) -> int:
        """Where a predicate-registered implementation sits in the order predicates are tried."""
        return next(
            index for index, entry in enumerate(self._predicates)
            if entry.implementation is implementation
        )

    @property
    def resolution_cache_size(self) -> int:
        return len(self._cache)

    def register(
        self,
        *patterns: DispatchPattern | Trait | TraitSet | Callable[..., object],
        precedence: str | None = None,
        position: int | None = None,
        fields: bool = False,
        staged: bool = False,
    ) -> Callable[[_Implementation], _Implementation]:
        """Register a declarative signature or an opaque GAType predicate.

        A sole callable is an escape hatch whose arguments are the complete
        concrete GATypes, not Extensor values. Predicate registrations are
        always tried before declarative ones; ``position`` controls order only
        within that low-level tier.

        Types with field slots dispatch only to predicates registered with
        ``fields``. Without one, fields of values run the implementation over
        blades at every site.
        """

        if precedence not in {None, "declaration"}:
            raise ValueError("precedence must be None or 'declaration'")
        if fields:
            if not (len(patterns) == 1 and is_predicate(patterns[0])):
                raise TypeError("a field implementation registers a predicate")
            def decorate_fields(implementation: _Implementation) -> _Implementation:
                self._field_predicates.append(_PredicateRegistration(patterns[0], implementation, staged))
                self._cache.clear()
                return implementation
            return decorate_fields
        if len(patterns) == 1 and is_predicate(patterns[0]):
            return self._register_predicate(patterns[0], position=position, staged=staged)
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
            self._registrations.append(_Registration(normalized, implementation, staged))
            self._cache.clear()
            return implementation

        return decorate

    def _register_predicate(
        self,
        predicate: Callable[..., object],
        *,
        position: int | None,
        staged: bool = False,
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
            entry = _PredicateRegistration(predicate, implementation, staged)
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
        """Validate, select, stage and orient once for a previously unseen signature."""

        actual = self._validate_actual_signature(actual_gatypes)
        if self._input_orientation is not None and actual[0].orientation is not self._input_orientation:
            raise GATypeError(
                f"{self.name!r} takes {self._input_orientation.value} elements; got {actual[0].signature}"
            )
        implementation = self._oriented(self._staged(self._select(actual), actual))
        self._cache[actual] = implementation
        return implementation

    def _select(self, actual: tuple[GAType, ...]) -> _Registration | _PredicateRegistration:
        if any(gatype.has_fields for gatype in actual):
            return self._select_fields(actual)

        for entry in self._predicates:
            if entry.predicate(*actual):
                return entry

        matches = tuple(
            entry
            for entry in self._registrations
            if _signature_matches(actual, entry.patterns)
        )
        if not matches:
            raise GATypeError(self._refusal(actual))

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
        return maximal[0]

    @staticmethod
    def _staged(entry: _Registration | _PredicateRegistration, actual: tuple[GAType, ...]) -> Callable[..., Any]:
        return entry.implementation(*actual) if entry.staged else entry.implementation

    def _oriented(self, implementation: Callable[..., Any]) -> Callable[..., Any]:
        """The implementation with its result in the declared output orientation."""

        orientation = self._output_orientation
        if orientation is None:
            return implementation

        def oriented(*arguments: Any, **kwargs: Any) -> Any:
            result = implementation(*arguments, **kwargs)
            gatype = result.gatype
            if gatype.orientation is orientation:
                return result
            gatype = gatype.derive.with_orientations((orientation,) + gatype.input_orientations)
            return type(result)._from_prepared_kernel(result.context, gatype, result._kernel)

        return oriented

    def _refusal(self, actual: tuple[GAType, ...]) -> str:
        return f"{self.name!r} takes {self._takes}; got {' and '.join(gatype.signature for gatype in actual)}"

    def _select_fields(self, actual: tuple[GAType, ...]) -> _PredicateRegistration:
        for entry in self._field_predicates:
            if entry.predicate(*actual):
                return entry
        # A map with sites is one map into or out of a field, not a map per site: maps per site are a batch.
        if any(gatype.has_fields and gatype.arity for gatype in actual):
            raise GATypeError(f"{self._refusal(actual)}: a map with sites is one map into or out of a field")
        return _PredicateRegistration(_always, numga.extensor.fields.site_by_site(self, self._operand_count))

    def __call__(self, *arguments: Any, **kwargs: Any) -> Any:
        count = self._operand_count
        try:
            if count == 1:
                actual = (arguments[0].gatype,)
            elif count == 2:
                actual = (arguments[0].gatype, arguments[1].gatype)
            else:
                actual = tuple(operand.gatype for operand in arguments[:count])
            implementation = self._cache[actual]
        except (AttributeError, IndexError, KeyError, TypeError):
            operands = arguments[:count]
            # Only valid, fixed-arity signatures enter the cache. A hit already
            # establishes these invariants without repeating type analysis.
            if len(arguments) < self._operand_count:
                raise TypeError(
                    f"{self.name!r} dispatch expects at least "
                    f"{self._operand_count} positional operands, got {len(arguments)}"
                ) from None
            # A bare nullary type operand is its identity map.
            operands = tuple(
                operand._identity() if isinstance(operand, GAType) else operand
                for operand in operands
            )
            arguments = operands + arguments[self._operand_count:]
            actual = tuple(_operand_gatype(operand) for operand in operands)
            implementation = self._cache.get(actual) or self._resolve_uncached(actual)
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


def _always(*gatypes: GAType) -> bool:
    return True


def is_predicate(pattern: object) -> bool:
    """A callable registration pattern that is a condition, not a type: types are callable too, as
    their identity maps."""
    return callable(pattern) and not isinstance(pattern, (GAType, GATypePattern, Trait, TraitSet))


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
