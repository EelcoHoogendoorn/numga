"""`ExtensionMethod` attaches whole-`GAType` multiple dispatch to `Extensor` as a descriptor,
allowing mathematical domains (`norms`, `logexp`, `inverse`, `systems`, `spectra`, `forms`) to
register implementations in separate modules without crowding `Extensor` itself or subclassing it
per algebra or grade. Each descriptor takes its public name explicitly so methods can be installed
on or off the class body without relying on `__set_name__`.

Every `ExtensionMethod` declares a plain-English `takes` domain description used in `GATypeError`
refusal messages whenever no registration matches an operand signature. Methods whose mathematical
definition fixes input or output orientation across all implementations, such as `exp` requiring
extrinsic inputs or `norm` and `cos` always returning extrinsic scalars, enforce
`input_orientation` and `output_orientation` centrally in the dispatcher rather than repeating
checks in each kernel.

Single-signature methods fix their operand count on first registration from either the number of
declarative patterns or a predicate's positional parameters. Methods that accept optional trailing
operands, such as `eigh(form)` versus generalized `eigh(form, metric)` or `lstsq(operator, rhs)`
versus `lstsq(form, rhs, metric)`, declare `overloads` up front and route each call length to its
own independent `ExtensionMethod` table. Dispatch and specialization across types are described in
`docs/internals.md` (section 7), with orientation constraints in `docs/orientation.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, signature
from types import MethodType
from typing import TYPE_CHECKING, overload as typing_overload

from numga.gatype import GAType, GATypeDispatch, GATypePattern, Orientation, Trait, TraitSet
from numga.gatype.dispatch import is_predicate

if TYPE_CHECKING:
    from numga.extensor import Extensor


class ExtensionMethod:
    """Each public method attaches a whole-`GAType` dispatch table (or per-arity overload tables) to
    `Extensor`."""

    __slots__ = ("_name", "_takes", "_n_operands", "_dispatch", "_overloads", "_input_orientation", "_output_orientation")

    def __init__(
        self,
        name: str,
        *,
        takes: str,
        overloads: tuple[int, ...] = (),
        input_orientation: Orientation | None = None,
        output_orientation: Orientation | None = None,
    ) -> None:
        if not isinstance(name, str) or not name or name.startswith("_"):
            raise TypeError("an extension method name must be a public name")
        self._name = name
        self._takes = takes
        self._n_operands = None
        self._dispatch = None
        self._overloads = {n_operands: ExtensionMethod(name, takes=takes) for n_operands in overloads}
        self._input_orientation = input_orientation
        self._output_orientation = output_orientation

    def position_of(self, implementation: Callable) -> int:
        """Locating an implementation in the predicate tier lets a narrower specialization insert
        directly ahead of it via `position`."""
        return self._dispatch.position_of(implementation)

    def overload(self, n_operands: int) -> ExtensionMethod:
        """Calls with `n_operands` operands route to their own independent `ExtensionMethod`
        dispatcher."""
        return self._overloads[n_operands]

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_operands(self) -> int | None:
        return self._n_operands

    def register(
        self,
        *patterns: GAType | GATypePattern | Trait | TraitSet | Callable,
        position: int | None = None,
        fields: bool = False,
        staged: bool = False,
    ) -> Callable:
        """`n_operands` is fixed by the first registration, either from the number of patterns or
        from a predicate's positional signature."""

        if not patterns:
            raise TypeError(f"{self.name!r} registration requires a pattern")
        if self._overloads:
            n_operands = (
                _predicate_n_operands(patterns[0], established=None)
                if len(patterns) == 1 and is_predicate(patterns[0]) else len(patterns)
            )
            return self._overloads[n_operands].register(
                *patterns, position=position, fields=fields, staged=staged,
            )
        predicate = len(patterns) == 1 and is_predicate(patterns[0])
        if predicate:
            n_operands = _predicate_n_operands(
                patterns[0],
                established=self._n_operands,
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
            n_operands = len(patterns)

        if self._n_operands is not None and self._n_operands != n_operands:
            raise TypeError(
                f"extension method {self.name!r} dispatches "
                f"{self._n_operands} operands, not {n_operands}"
            )

        def decorate(implementation: Callable) -> Callable:
            if self._n_operands is not None and self._n_operands != n_operands:
                raise TypeError(
                    f"extension method {self.name!r} dispatches "
                    f"{self._n_operands} operands, not {n_operands}"
                )
            created = self._dispatch is None
            dispatch = (
                GATypeDispatch(
                    self.name, None, n_operands, takes=self._takes,
                    input_orientation=self._input_orientation, output_orientation=self._output_orientation,
                )
                if created
                else self._dispatch
            )
            assert dispatch is not None
            registered = dispatch.register(
                *patterns,
                position=position,
                fields=fields,
                staged=staged,
            )(implementation)
            if created:
                self._dispatch = dispatch
            if self._n_operands is None:
                self._n_operands = n_operands
            return registered

        return decorate

    def __call__(self, *arguments, **kwargs):
        if self._overloads:
            return self._overloads[len(arguments)](*arguments, **kwargs)
        dispatch = self._dispatch
        if dispatch is None:
            raise LookupError(
                f"extension method {self.name!r} has no registered implementations"
            )
        return dispatch(*arguments, **kwargs)

    @typing_overload
    def __get__(self, instance: None, owner: type[Extensor] | None = None) -> ExtensionMethod: ...

    @typing_overload
    def __get__(self, instance: Extensor, owner: type[Extensor] | None = None) -> MethodType: ...

    def __get__(
        self,
        instance: Extensor | None,
        owner: type[Extensor] | None = None,
    ) -> ExtensionMethod | MethodType:
        if instance is None:
            return self
        dispatch = self._dispatch
        if dispatch is None:
            return MethodType(self, instance)
        return MethodType(dispatch, instance)


def _predicate_n_operands(
    predicate: Callable,
    *,
    established: int | None = None,
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
