"""Per-algebra canonical construction of whole-extensor types."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache, wraps
from typing import TYPE_CHECKING

from numga.flyweight import FlyweightFactory
from numga.subspace import SubSpace

from .gatype import GAType
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
    semantic_constructor_names = ("rotor",)

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
    ) -> GAType:
        if isinstance(subspaces, (SubSpace, GAType)):
            raw_axes = (subspaces,)
        else:
            try:
                raw_axes = tuple(subspaces)
            except TypeError as error:
                raise TypeError(
                    "GAType subspaces must be a SubSpace, GAType, or output-first iterable"
                ) from error

        def _to_subspace(axis: SubSpace | GAType) -> SubSpace:
            if isinstance(axis, SubSpace):
                return axis
            if isinstance(axis, GAType) and axis.arity == 0:
                return axis.output_subspace
            raise TypeError(
                f"each axis must be a SubSpace or nullary GAType; got {type(axis).__name__}"
            )

        axes = tuple(_to_subspace(axis) for axis in raw_axes)

        # Let the value object own detailed diagnostics and structural
        # validation. Checking ownership here keeps a foreign type out of this
        # algebra's pool without duplicating the full constructor.
        if axes and any(
            not isinstance(axis, SubSpace) or axis.algebra is not self.algebra
            for axis in axes
        ):
            raise ValueError("every GAType axis must belong to this factory's algebra")

        normalized_traits = normalize_explicit_traits(axes, traits)

        key = (axes, normalized_traits)
        return self.factory_construct(
            key,
            lambda: GAType(axes, normalized_traits),
        )

    @lru_cache(maxsize=None)
    def lift(self, subspace: SubSpace) -> GAType:
        if not isinstance(subspace, SubSpace):
            raise TypeError(
                "GATypeFactory.lift requires one SubSpace; "
                f"got {type(subspace).__name__}"
            )
        if subspace.algebra is not self.algebra:
            raise ValueError("cannot lift a SubSpace from another algebra")
        return self(subspace)

    @lru_cache(maxsize=None)
    def rotor(self) -> GAType:
        """Return the certified unit-versor refinement of even support."""

        return self(self.algebra.subspace.even(), ROTOR_TRAITS)

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

        @wraps(constructor)
        def lifted_constructor(*args: object, **kwargs: object) -> GAType:
            return self.lift(constructor(*args, **kwargs))

        self.__dict__[name] = lifted_constructor
        return lifted_constructor

    def __dir__(self) -> list[str]:
        forwarded = set(self.constructor_names)
        return sorted(set(super().__dir__()) | forwarded)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("GATypeFactory is immutable")

    def __repr__(self) -> str:
        return f"GATypeFactory(algebra={self.algebra!r})"
