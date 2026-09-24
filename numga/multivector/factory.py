"""A context-bound construction namespace for nullary Extensors."""

from __future__ import annotations

import numpy as np

from functools import lru_cache
from inspect import Signature, signature
from typing import TYPE_CHECKING, Any, Callable

from numga.gatype import GAType, ROTOR_TRAITS, ReverseProductZero, TraitSet
from numga.subspace import SubSpace

if TYPE_CHECKING:
    import numpy as np

    from numga.algebra import Algebra
    from numga.backend import Context
    from numga.extensor import Extensor


_MISSING = object()
_ZERO_TRAITS = TraitSet((ReverseProductZero,))


class MultivectorFactory:
    """Thin context-bound facade over nullary ``Extensor`` construction.

    Named constructors obtain the complete type from ``algebra.gatype``.  This
    is important for semantic constructors: ``even`` creates a plain even
    value, while ``rotor`` (and its alias ``motor``) retains the certified rotor
    traits. Those traits are trusted, not checked: ``mv.rotor(values)`` asserts
    that the values are a unit versor, and methods such as ``inverse`` act on that
    assertion. Values of unknown provenance enter as ``mv.even(values)`` and are
    certified by ``.normalized()``.
    """

    __slots__ = ("_context", "__dict__")

    def __init__(self, context: "Context") -> None:
        if context is None:
            raise TypeError("a MultivectorFactory requires a Context")
        object.__setattr__(self, "_context", context)

    @property
    def context(self) -> "Context":
        return self._context

    @property
    def algebra(self) -> Algebra:
        return self.context.algebra

    @property
    def dtype(self) -> np.dtype:
        return self.context.dtype

    def blade(self, mask: int) -> Extensor:
        """The unit basis blade with this generator mask."""
        return self(self.algebra.subspace.blade(mask), [1.0])

    def basis(self) -> Extensor:
        """The unit basis vectors, as a batch along the first axis: x, y, z = mv.basis()."""
        return self.vector(np.eye(len(self.algebra.subspace.vector())))

    def __call__(
        self,
        gatype_or_subspace: GAType | SubSpace | str,
        coefficients: Any = _MISSING,
    ) -> Extensor:
        if isinstance(gatype_or_subspace, str):
            gatype_or_subspace = self.algebra.subspace(gatype_or_subspace)
        if isinstance(gatype_or_subspace, GAType):
            gatype = gatype_or_subspace
        elif isinstance(gatype_or_subspace, SubSpace):
            gatype = self.algebra.gatype(gatype_or_subspace)
        else:
            raise TypeError(
                "multivector construction requires a GAType, SubSpace, or blade string"
            )
        if gatype.arity != 0:
            raise ValueError(
                "multivector construction requires an arity-0 GAType; "
                f"got arity {gatype.arity}"
            )
        if coefficients is _MISSING:
            coefficients, known_traits = _projected_unit(gatype.output_subspace)
            try:
                complete_traits = TraitSet(gatype.traits.traits + known_traits.traits)
            except ValueError as error:
                raise ValueError(
                    "omitted coefficients produce a projected unit whose known "
                    f"facts contradict {gatype.traits!r}"
                ) from error
            gatype = self.algebra.gatype(gatype.subspaces, complete_traits)
        return self.context.extensor(gatype, coefficients)

    def __getattr__(self, name: str) -> Callable[..., Extensor] | Extensor:
        value = self._resolve_name(name)
        self.__dict__[name] = value
        return value

    def _resolve_name(self, name: str) -> Callable[..., Extensor] | Extensor:
        """Bind a declared GAType constructor, or a unit basis blade."""

        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )

        if name in self.algebra.gatype.constructor_names:
            gatype_constructor = getattr(self.algebra.gatype, name)
            constructor_signature = signature(gatype_constructor)

            if not constructor_signature.parameters:
                gatype = gatype_constructor()

                def construct_value(coefficients: Any = _MISSING) -> Extensor:
                    return self(gatype, coefficients)

                return construct_value

            def construct(
                *type_and_coefficients: object,
                coefficients: Any = _MISSING,
                **type_kwargs: object,
            ) -> Extensor:
                if coefficients is _MISSING:
                    type_args, value = _split_optional_coefficients(
                        constructor_signature,
                        type_and_coefficients,
                        type_kwargs,
                    )
                else:
                    type_args = type_and_coefficients
                    value = coefficients
                gatype = gatype_constructor(*type_args, **type_kwargs)
                return self(gatype, value)

            construct.__name__ = name
            construct.__qualname__ = f"{type(self).__name__}.{name}"
            return construct

        # Allow basis blade access like mv.x, mv.y, mv.xy, mv.xyz
        try:
            oriented = self.algebra.parse_blade(name)
        except (KeyError, ValueError):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )

        subspace = self.algebra.subspace.from_layout((oriented.mask,), (oriented.sign,))
        gatype = self.algebra.gatype(subspace)
        return self(gatype, [1.0])


    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__()) | set(self.algebra.gatype.constructor_names)
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("MultivectorFactory is immutable")

    def __repr__(self) -> str:
        return f"MultivectorFactory(context={self.context!r})"




def _split_optional_coefficients(
    constructor_signature: Signature,
    arguments: tuple[object, ...],
    keywords: dict[str, object],
) -> tuple[tuple[object, ...], object]:
    """Separate a final positional value from constructor arguments.

    A complete call to the GAType constructor means coefficients were omitted.
    Otherwise a final positional argument is coefficients exactly when removing
    it makes the type-constructor call complete. This keeps both
    ``vector(values)`` and ``k_vector(grade, values)`` while making
    ``k_vector(grade)`` the projected-unit default.
    """

    try:
        constructor_signature.bind(*arguments, **keywords)
    except TypeError:
        if arguments:
            type_arguments = arguments[:-1]
            try:
                constructor_signature.bind(*type_arguments, **keywords)
            except TypeError:
                pass
            else:
                return type_arguments, arguments[-1]
    else:
        return arguments, _MISSING

    # Preserve the constructor's own diagnostic for malformed type arguments.
    return arguments, _MISSING


@lru_cache(maxsize=None)
def _projected_unit(subspace: SubSpace) -> tuple[tuple[int, ...], TraitSet]:
    """Coordinates and exactly known facts of the projected scalar unit."""

    contains_scalar = 0 in subspace.masks
    coefficients = tuple(sign if mask == 0 else 0 for mask, sign in zip(subspace.masks, subspace.signs))
    traits = ROTOR_TRAITS if contains_scalar else _ZERO_TRAITS
    return coefficients, traits
