"""Extensor descriptors backed by whole-GAType multiple dispatch."""

from __future__ import annotations

from inspect import Parameter, signature
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload as typing_overload

from numga.gatype import GAType, GATypeDispatch, GATypePattern, Trait, TraitSet

if TYPE_CHECKING:
    from numga.extensor import Extensor

_Implementation = TypeVar("_Implementation", bound=Callable[..., Any])


class ExtensionMethod:
    """A dynamically installable, whole-GAType-dispatched Extensor method.

    Portable registrations use algebra-independent :class:`GATypePattern`
    values (or unambiguous Trait/TraitSet shorthand). A concrete GAType binds
    one algebra and matches every operand whose support it contains, in any
    blade layout. A callable condition receives the complete GATypes, layout
    included, and is tried before every declarative registration.

    The name is explicit because assigning a descriptor to ``Extensor`` after
    class creation does not invoke the descriptor ``__set_name__`` hook.
    """

    __slots__ = ("_name", "_operand_count", "_dispatch", "_overloads")

    def __init__(self, name: str, *, operand_counts: tuple[int, ...] = ()) -> None:
        if not isinstance(name, str) or not name or name.startswith("_"):
            raise TypeError("an extension method name must be a public name")
        self._name = name
        self._operand_count: int | None = None
        self._dispatch: GATypeDispatch | None = None
        self._overloads = {count: ExtensionMethod(name) for count in operand_counts}

    def overload(self, operand_count: int) -> ExtensionMethod:
        """The independently registered dispatcher for a positional overload."""
        return self._overloads[operand_count]

    @property
    def name(self) -> str:
        return self._name

    @property
    def operand_count(self) -> int | None:
        return self._operand_count

    def register(
        self,
        *patterns: GAType | GATypePattern | Trait | TraitSet | Callable[..., object],
        precedence: str | None = None,
        position: int | None = None,
    ) -> Callable[[_Implementation], _Implementation]:
        """Register a type signature without mutating until decoration.

        The number of dispatched Extensor operands is the number of
        declarative patterns. For a sole predicate it is inferred from that
        callable's fixed positional signature, or checked against an arity
        already established by earlier registrations.
        """

        if not patterns:
            raise TypeError(f"{self.name!r} registration requires a pattern")
        if self._overloads:
            count = (
                _predicate_operand_count(patterns[0], established=None)
                if len(patterns) == 1 and callable(patterns[0]) else len(patterns)
            )
            return self._overloads[count].register(
                *patterns, precedence=precedence, position=position,
            )
        predicate = len(patterns) == 1 and callable(patterns[0])
        if predicate:
            operand_count = _predicate_operand_count(
                patterns[0],
                established=self._operand_count,
            )
        else:
            if not all(
                isinstance(pattern, (GAType, GATypePattern, Trait, TraitSet))
                for pattern in patterns
            ):
                raise TypeError(
                    f"every {self.name!r} registration entry must be a GAType, "
                    "GATypePattern, Trait, or TraitSet"
                )
            operand_count = len(patterns)

        if self._operand_count is not None and self._operand_count != operand_count:
            raise TypeError(
                f"extension method {self.name!r} dispatches "
                f"{self._operand_count} operands, not {operand_count}"
            )

        def decorate(implementation: _Implementation) -> _Implementation:
            if self._operand_count is not None and self._operand_count != operand_count:
                raise TypeError(
                    f"extension method {self.name!r} dispatches "
                    f"{self._operand_count} operands, not {operand_count}"
                )
            created = self._dispatch is None
            dispatch = (
                GATypeDispatch(self.name, None, operand_count)
                if created
                else self._dispatch
            )
            assert dispatch is not None
            registered = dispatch.register(
                *patterns,
                precedence=precedence,
                position=position,
            )(implementation)
            if created:
                self._dispatch = dispatch
            if self._operand_count is None:
                self._operand_count = operand_count
            return registered

        return decorate

    def __call__(self, *arguments: Any, **kwargs: Any) -> Any:
        if self._overloads:
            return self._overloads[len(arguments)](*arguments, **kwargs)
        dispatch = self._dispatch
        if dispatch is None:
            raise LookupError(
                f"extension method {self.name!r} has no registered implementations"
            )
        return dispatch(*arguments, **kwargs)

    @typing_overload
    def __get__(
        self,
        instance: None,
        owner: type[Extensor] | None = None,
    ) -> ExtensionMethod: ...

    @typing_overload
    def __get__(
        self,
        instance: Extensor,
        owner: type[Extensor] | None = None,
    ) -> MethodType: ...

    def __get__(
        self,
        instance: Extensor | None,
        owner: type[Extensor] | None = None,
    ) -> ExtensionMethod | MethodType:
        if instance is None:
            return self
        dispatch = self._dispatch
        return MethodType(self if dispatch is None else dispatch, instance)


def _predicate_operand_count(
    predicate: Callable[..., object],
    *,
    established: int | None,
) -> int:
    try:
        parameters = tuple(signature(predicate).parameters.values())
    except (TypeError, ValueError) as error:
        if established is not None:
            return established
        raise TypeError(
            "cannot infer the number of dispatched operands from this predicate; "
            "register a declarative pattern first"
        ) from error

    if established is not None:
        try:
            signature(predicate).bind(*(None for _ in range(established)))
        except TypeError as error:
            raise TypeError(
                f"predicate for this extension method must accept {established} "
                "positional GATypes"
            ) from error
        return established

    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )
    if any(parameter.kind is Parameter.VAR_POSITIONAL for parameter in parameters):
        raise TypeError(
            "cannot infer dispatch arity from a variadic predicate; register a "
            "declarative pattern first"
        )
    if any(
        parameter.kind is Parameter.KEYWORD_ONLY
        and parameter.default is Parameter.empty
        for parameter in parameters
    ):
        raise TypeError("a GAType predicate cannot require keyword-only arguments")
    if not positional or any(
        parameter.default is not Parameter.empty for parameter in positional
    ):
        raise TypeError(
            "a first predicate registration must have a fixed, non-optional "
            "positional signature"
        )
    return len(positional)
