"""Per-algebra canonical construction of whole-extensor types."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache, wraps
from typing import TYPE_CHECKING

from numga.flyweight import FlyweightFactory
from numga.subspace import SubSpace

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


class GATypeFactory(FlyweightFactory[GAType]):
    """Strong flyweight pool scoped to one Algebra instance."""

    __slots__ = ("_algebra", "__dict__")

    # Public constructor protocol for namespaces which lift complete GATypes.
    # Structural names come from the selected SubSpace factory; these names
    # add semantic refinements rather than merely selecting support.
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
        subspaces: SubSpace | GAType | Iterable[SubSpace | GAType],
        traits: TraitSet = EMPTY_TRAITS,
        fields: tuple[tuple[int, int], ...] = (),
        *,
        orientation: Orientation | None = None,
    ) -> GAType:
        """The type over these axes, output first. A nullary GAType axis carries its orientation; bare
        subspace axes take `orientation`, which they require: a subspace is a layout and has none."""
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
        return self.oriented(axes, orientations, traits, fields)

    def oriented(
        self,
        subspaces: Iterable[SubSpace],
        orientations: Iterable[Orientation],
        traits: TraitSet = EMPTY_TRAITS,
        fields: tuple[tuple[int, int], ...] = (),
    ) -> GAType:
        """The GAType over these subspaces with these orientations, one per axis, output first."""

        axes = tuple(subspaces)
        orientations = tuple(orientations)

        # Let the value object own detailed diagnostics and structural
        # validation. Checking ownership here keeps a foreign type out of this
        # algebra's pool without duplicating the full constructor.
        if axes and any(
            not isinstance(axis, SubSpace) or axis.algebra is not self.algebra
            for axis in axes
        ):
            raise ValueError("every GAType axis must belong to this factory's algebra")

        normalized_traits = normalize_explicit_traits(axes, traits)

        fields = tuple(sorted(fields))
        key = (axes, orientations, normalized_traits, fields)
        return self.factory_construct(
            key,
            lambda: GAType(axes, orientations, normalized_traits, fields),
        )

    @lru_cache(maxsize=None)
    def rotor(self) -> GAType:
        """Return the certified unit-versor refinement of even support."""

        return self.oriented((self.algebra.subspace.even(),), (Orientation.extrinsic,), ROTOR_TRAITS)

    motor = rotor

    def __getattr__(self, name: str) -> Callable[..., GAType]:
        """Lift a declared SubSpace constructor into this GAType namespace."""

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
        def lifted_constructor(*args: object, **kwargs: object) -> GAType:
            return self.oriented((constructor(*args, **kwargs),), (orientation,))

        self.__dict__[name] = lifted_constructor
        return lifted_constructor

    def __dir__(self) -> list[str]:
        forwarded = set(self.constructor_names)
        return sorted(set(super().__dir__()) | forwarded)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("GATypeFactory is immutable")

    def __repr__(self) -> str:
        return f"GATypeFactory(algebra={self.algebra!r})"
