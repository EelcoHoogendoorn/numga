"""Each algebra owns a flyweight factory (`ga.gatype`) that interns canonical `GAType` instances in
a strong pool keyed by output-first `SubSpace` axes, per-axis `Orientation`s, normalized `TraitSet`s,
and coupled field site counts. Because explicit traits already implied by the blade support are
stripped before pooling, equivalent structural and trait specifications resolve to the same `GAType`
object.

Named constructors on `ga.gatype` mirror `ga.subspace` while attaching canonical geometric
orientations: antivector-built spaces (`antiscalar`, `antivector`, `antibivector`, and complementary
builders) default to `Orientation.intrinsic`, whereas vector-built spaces (`scalar`, `vector`,
`bivector`, `even`, `odd`, `multivector`) default to `Orientation.extrinsic`. Semantic constructors
such as `rotor` (aliased as `motor`) further attach certified `Versor` and `ReverseProductOne`
traits to the even subalgebra.

Calling `ga.gatype(...)` directly builds nullary, unary, or multilinear types from output-first
nullary `GAType`s (which supply their own orientations), from bare `SubSpace`s paired with an
explicit `orientation=...` keyword, or from a symbolic `Extensor` expression such as
`ga.gatype(Point & Point)` whose output `GAType` records the support, orientation, and traits
produced by that expression.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache, wraps
from typing import TYPE_CHECKING

import numga
from numga.flyweight import FlyweightFactory
from numga.subspace import SubSpace
from numga.units import UnitGrid, coerce_unit_grid, dim

from .errors import GATypeError
from .gatype import GAType
from .orientation import Orientation
from .traits import (
    EMPTY_TRAITS,
    ROTOR_TRAITS,
    TraitSet,
    normalize_explicit_traits,
)

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.extensor import Extensor


class GATypeFactory(FlyweightFactory[GAType]):
    """A per-algebra flyweight namespace constructs and interns canonical `GAType` instances."""

    __slots__ = ("_algebra", "__dict__")

    # Semantic constructors attach algebraic traits on top of a structural SubSpace.
    semantic_constructor_names = ("rotor", "motor")

    def __init__(self, algebra: Algebra) -> None:
        if algebra is None:
            raise TypeError("a GATypeFactory requires an algebra")
        super().__init__()
        object.__setattr__(self, "_algebra", algebra)

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    @lru_cache(maxsize=None)
    def constructor_names(self) -> tuple[str, ...]:
        """Named constructors combine structural `SubSpace` builders with `rotor` and `motor`."""
        return tuple(
            dict.fromkeys(
                (
                    *self.algebra.subspace.constructor_names,
                    *self.semantic_constructor_names,
                )
            )
        )

    def __call__(
        self,
        subspaces: SubSpace | GAType | Iterable | Extensor,
        traits: TraitSet = EMPTY_TRAITS,
        fields: tuple = (),
        *,
        orientation: Orientation | None = None,
        unit: object = None,
    ) -> GAType:
        """Calling `ga.gatype` builds a canonical `GAType` from output-first nullary `GAType`s,
        from bare `SubSpace`s paired with an explicit `orientation`, or from a symbolic `Extensor`
        expression whose `output_gatype` is extracted."""
        if isinstance(subspaces, numga.extensor.Extensor):
            if traits != EMPTY_TRAITS or fields or orientation is not None:
                raise TypeError("an expression's type is what its construction gives: no traits, fields or orientation")
            if subspaces.algebra is not self.algebra:
                raise ValueError("the expression belongs to another algebra than this factory's")
            result = subspaces.gatype.derive.plain if subspaces.arity == 0 else subspaces.output_gatype
            return result if unit is None else result.derive.unit(unit)
        if isinstance(subspaces, (SubSpace, GAType)):
            raw_axes = (subspaces,)
        else:
            try:
                raw_axes = tuple(subspaces)
            except TypeError as error:
                raise TypeError(
                    "GAType subspaces must be a SubSpace, GAType, or output-first iterable"
                ) from error

        if any(isinstance(axis, SubSpace) for axis in raw_axes) and orientation is None:
            raise TypeError(
                "a subspace has no orientation: pass orientation=Orientation.extrinsic or "
                "Orientation.intrinsic, or build the type from a named constructor"
            )
        if orientation is not None and not any(isinstance(axis, SubSpace) for axis in raw_axes):
            raise TypeError("orientation applies to subspace axes; these GATypes carry their own")
        if not all(isinstance(axis, SubSpace) or (isinstance(axis, GAType) and axis.arity == 0) for axis in raw_axes):
            raise TypeError("each axis must be a SubSpace or a nullary GAType")
        axes = tuple(axis if isinstance(axis, SubSpace) else axis.output_subspace for axis in raw_axes)
        orientations = tuple(orientation if isinstance(axis, SubSpace) else axis.orientation for axis in raw_axes)
        if unit is not None:
            resolved_units: UnitGrid | None = coerce_unit_grid(unit, axes)
        elif len(raw_axes) == 1 and isinstance(raw_axes[0], GAType):
            resolved_units = raw_axes[0].units
        elif any(isinstance(axis, GAType) and axis.has_units for axis in raw_axes):
            axis_grids = tuple(
                axis.units if isinstance(axis, GAType) and axis.has_units
                else UnitGrid.uniform_grid((len(space),), dim.none)
                for axis, space in zip(raw_axes, axes)
            )
            resolved_units = UnitGrid.outer_quotient(axis_grids[0], axis_grids[1:])
        else:
            resolved_units = None
        return self.oriented(axes, orientations, traits, fields, units=resolved_units)

    def oriented(
        self,
        subspaces: Iterable[SubSpace],
        orientations: Iterable[Orientation],
        traits: TraitSet = EMPTY_TRAITS,
        fields: tuple = (),
        units: UnitGrid | None = None,
    ) -> GAType:
        """Explicit output-first subspaces and per-axis orientations pool a canonical `GAType` after
        stripping structurally implied traits."""
        axes = tuple(subspaces)
        orientations = tuple(orientations)

        # Reject foreign SubSpaces before pooling while leaving detailed arity and trait checks to GAType.
        if axes and any(
            not isinstance(axis, SubSpace) or axis.algebra is not self.algebra
            for axis in axes
        ):
            raise ValueError("every GAType axis must belong to this factory's algebra")
        blade_shape = tuple(len(axis) for axis in axes)
        if units is None:
            resolved_units = UnitGrid.unchecked(blade_shape)
        elif not isinstance(units, UnitGrid):
            raise TypeError(f"GAType units must be a UnitGrid; got {units!r}")
        elif units.shape != blade_shape:
            raise GATypeError(f"UnitGrid shape {units.shape} does not match type shape {blade_shape}")
        else:
            resolved_units = units

        normalized_traits = normalize_explicit_traits(axes, traits)

        fields = tuple(sorted(fields))
        key = (axes, orientations, normalized_traits, fields, resolved_units)
        return self.factory_construct(
            key,
            lambda: GAType(axes, orientations, normalized_traits, fields, resolved_units),
        )

    def rotor(self, *, unit: object = None) -> GAType:
        """Rotors span the extrinsic even subalgebra with certified `Versor` and `ReverseProductOne`
        traits."""
        space = self.algebra.subspace.even()
        resolved = None if unit is None else coerce_unit_grid(unit, (space,))
        return self.oriented((space,), (Orientation.extrinsic,), ROTOR_TRAITS, units=resolved)

    motor = rotor

    def __getattr__(self, name: str) -> Callable:
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )
        subspace_factory = self.algebra.subspace
        if name not in subspace_factory.constructor_names:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )
        constructor = getattr(subspace_factory, name)
        intrinsic = name in subspace_factory.intrinsic_constructor_names
        orientation = Orientation.intrinsic if intrinsic else Orientation.extrinsic

        @wraps(constructor)
        def constructed(*args: object, unit: object = None, **kwargs: object) -> GAType:
            space = constructor(*args, **kwargs)
            resolved = None if unit is None else coerce_unit_grid(unit, (space,))
            return self.oriented((space,), (orientation,), units=resolved)

        self.__dict__[name] = constructed
        return constructed

    def __dir__(self) -> list[str]:
        forwarded = set(self.constructor_names)
        return sorted(set(super().__dir__()) | forwarded)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("GATypeFactory is immutable")

    def __repr__(self) -> str:
        return f"GATypeFactory(algebra={self.algebra!r})"
