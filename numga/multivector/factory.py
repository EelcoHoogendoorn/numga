"""`MultivectorFactory` (customarily bound as `mv = context.multivector`) constructs nullary
`Extensor` values (`arity == 0`) inside a specific backend `Context` from array-like coordinates or
basis blade names. Every constructor declared in `algebra.gatype.constructor_names` forwards onto
`mv` with its full `GAType`, preserving orientation and semantic traits; `mv.antivector(coords)`
creates an intrinsic antivector, `mv.even(coords)` creates an uncertified even multivector, and
`mv.rotor(coords)` (or `mv.motor(coords)`) attaches certified unit-versor `ROTOR_TRAITS` trusted by
downstream operations such as `inverse` without runtime renormalization. Uncertified even elements
enter via `mv.even(coords)` and acquire rotor traits through `.normalized()`.

Omitting coordinates on any value constructor (`mv.rotor()`, `mv.even()`, `mv.k_vector(2)`, or
`mv(gatype)`) constructs the orthogonal projection of the scalar unit `1` onto that target
subspace, enriching the type's `TraitSet` with `ROTOR_TRAITS` when grade `0` lies in the subspace
and `ReverseProductZero` when it does not. Parameterized type constructors such as
`mv.k_vector(grade, coefficients)` separate type arguments from trailing coordinate arrays by
binding against the underlying `GATypeFactory` signature.

Attribute lookups not matching a named type constructor parse as exterior basis blades (`mv.x`,
`mv.yz`, `mv.xyz`), caching unit extrinsic single-blade values directly on the factory's instance
dictionary so repeated `mv.xy` accesses incur no parsing or allocation overhead. Direct calls
`mv(gatype, coefficients)` require an explicit `arity == 0` `GAType` and reject bare `SubSpace` or
string layouts with `UnorientedSubspace` so orientation is never guessed. Orientation choices at
construction and trusted rotor traits are discussed in `docs/orientation.md` and `docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from inspect import Signature, signature
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

from numga.gatype import GAType, Orientation, ROTOR_TRAITS, ReverseProductZero, TraitSet, UnorientedSubspace
from numga.subspace import SubSpace
from numga.units import UnitGrid, dim

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.backend import Context
    from numga.extensor import Extensor


_MISSING = object()
_ZERO_TRAITS = TraitSet((ReverseProductZero,))


class MultivectorFactory:
    """Context-bound constructors bind `algebra.gatype` types and unit basis blades to a single
    backend `Context`."""

    __slots__ = ("_context", "__dict__")

    def __init__(self, context: Context) -> None:
        if context is None:
            raise TypeError("a MultivectorFactory requires a Context")
        object.__setattr__(self, "_context", context)

    @property
    def context(self) -> Context:
        return self._context

    @property
    def algebra(self) -> Algebra:
        return self.context.algebra

    @property
    def dtype(self) -> np.dtype:
        return self.context.dtype

    def blade(self, mask: int) -> Extensor:
        """Constructing from canonical blade `mask` returns the unit extrinsic basis blade on its
        single-blade subspace."""
        unit = dim.none if (self.context.units or self.context.is_exact) else None
        return self(self.algebra.gatype(self.algebra.subspace.blade(mask), orientation=Orientation.extrinsic, unit=unit), [1.0])

    def basis(self) -> Extensor:
        """Stacking the algebra's unit basis vectors along a leading batch axis allows unpacking as
        `x, y, z = mv.basis()`."""
        return self.vector(np.eye(len(self.algebra.subspace.vector())))

    def zeros(self, gatype: GAType, shape: int | tuple[int, ...] = ()) -> Extensor:
        """Constructing a zero-filled batch of `gatype` without attaching `ReverseProductZero`
        gives a typed carry or accumulator of the exact requested type."""
        if isinstance(gatype, (str, SubSpace)):
            raise UnorientedSubspace()
        if not isinstance(gatype, GAType):
            raise TypeError("zeros requires a GAType")
        if isinstance(shape, int):
            shape = (shape,)
        if not gatype.has_units and self.context.units:
            gatype = gatype.derive.unit(UnitGrid.zero(tuple(len(space) for space in gatype.subspaces)))
        return self.context.extensor(
            gatype, self.context.xp.zeros(shape + gatype.structural_shape, dtype=self.dtype),
        )

    def __call__(
        self,
        gatype: GAType,
        coefficients: object = _MISSING,
    ) -> Extensor:
        """Constructing a value from an `arity == 0` `GAType` wraps `coefficients` in `context` or
        produces the projected scalar unit when `coefficients` is omitted."""
        if isinstance(gatype, (str, SubSpace)):
            raise UnorientedSubspace()
        if not isinstance(gatype, GAType):
            raise TypeError("multivector construction requires a GAType")
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
            if not gatype.has_units and (self.context.units or self.context.is_exact):
                entries = {
                    i: dim.none
                    for i, mask in enumerate(gatype.output_subspace.masks)
                    if mask == 0
                }
                projected_units = UnitGrid.from_entries((len(gatype.output_subspace),), entries)
            else:
                projected_units = gatype.units
            gatype = self.algebra.gatype.oriented(
                gatype.subspaces, gatype.orientations, complete_traits, units=projected_units,
            )
        elif gatype.is_scalar:
            if isinstance(coefficients, Number):
                coefficients = (coefficients,)
            elif getattr(coefficients, "ndim", None) == 0:
                coefficients = coefficients[None]
        return self.context.extensor(gatype, coefficients)

    def __getattr__(self, name: str) -> Callable | Extensor:
        value = self._resolve_name(name)
        self.__dict__[name] = value
        return value

    def _resolve_name(self, name: str) -> Callable | Extensor:
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )

        if name in self.algebra.gatype.constructor_names:
            gatype_constructor = getattr(self.algebra.gatype, name)
            constructor_signature = signature(gatype_constructor)

            if all(parameter.name == "unit" for parameter in constructor_signature.parameters.values()):
                gatype = gatype_constructor()

                def construct_value(coefficients: object = _MISSING, *, unit: object = None) -> Extensor:
                    target = gatype if unit is None else gatype.derive.unit(unit)
                    return self(target, coefficients)

                return construct_value

            def construct(
                *type_and_coefficients: object,
                coefficients: object = _MISSING,
                unit: object = None,
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
                gatype = gatype_constructor(*type_args, unit=unit, **type_kwargs)
                return self(gatype, value)

            construct.__name__ = name
            construct.__qualname__ = f"{type(self).__name__}.{name}"
            return construct

        try:
            oriented = self.algebra.parse_blade(name)
        except (KeyError, ValueError):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )

        # Basis blades represent products of basis planes and take extrinsic orientation.
        subspace = self.algebra.subspace.from_layout((oriented.mask,), (oriented.sign,))
        unit = dim.none if (self.context.units or self.context.is_exact) else None
        return self(self.algebra.gatype(subspace, orientation=Orientation.extrinsic, unit=unit), [1.0])

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
) -> tuple:
    # A complete bind means coefficients were omitted; otherwise a trailing argument is
    # coefficients when dropping it makes the GAType constructor bind succeed.
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

    return arguments, _MISSING


@lru_cache(maxsize=None)
def _projected_unit(subspace: SubSpace) -> tuple:
    contains_scalar = 0 in subspace.masks
    coefficients = tuple(sign if mask == 0 else 0 for mask, sign in zip(subspace.masks, subspace.signs))
    traits = ROTOR_TRAITS if contains_scalar else _ZERO_TRAITS
    return coefficients, traits
