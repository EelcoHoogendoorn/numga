"""The minimal default SubSpace construction namespace."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from operator import index
from typing import TYPE_CHECKING, Iterable

from numga.flyweight import FlyweightFactory

from .subspace import SubSpace, _normalize_masks

if TYPE_CHECKING:
    from numga.algebra import Algebra


class SubSpaceFactory(FlyweightFactory[SubSpace]):
    """Per-algebra flyweights for default supports and explicitly signed layouts."""

    __slots__ = ("_algebra", "_blade_masks", "_default")

    # The public constructor protocol mirrored by GATypeFactory. Custom
    # SubSpace factories can extend this tuple together with their methods;
    # implementation helpers are deliberately absent.
    constructor_names = (
        "from_masks",
        "from_blades",
        "from_layout",
        "from_grades",
        "empty",
        "k_vector",
        "scalar",
        "vector",
        "bivector",
        "trivector",
        "pseudoscalar",
        "antivector",
        "antibivector",
        "full",
        "even",
        "multivector",
        "even_grade",
    )

    def __init__(self, algebra: Algebra, default: str | None = None) -> None:
        dimension = algebra.dimension
        super().__init__()
        blade_masks = getattr(algebra, "blade_masks", range(1 << dimension))
        # Algebra's native ``range`` remains lazy, so merely asking for a
        # construction namespace is not exponential in the dimension.
        frozen_blade_masks = (
            blade_masks if isinstance(blade_masks, range) else tuple(blade_masks)
        )
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_blade_masks", frozen_blade_masks)
        layout = self.from_blades(default) if default is not None else None
        if layout is not None and len(layout) != 1 << dimension:
            raise ValueError("a default layout must specify every basis blade")
        object.__setattr__(self, "_default", layout)

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    def from_masks(self, masks: Iterable[int]) -> SubSpace:
        normalized = _normalize_masks(self.algebra, masks)
        if self._default is not None:
            return self._default.restrict(normalized)
        return self.from_layout(normalized, (1,) * len(normalized))

    def from_layout(self, masks: Iterable[int], signs: Iterable[int]) -> SubSpace:
        """Intern exact ordered masks and signs; no default ordering is applied."""

        masks, signs = tuple(masks), tuple(signs)
        return self.factory_construct(
            (masks, signs),
            lambda: SubSpace(self.algebra, masks, signs),
        )

    def from_blades(self, blades: str | Iterable[str | tuple[str, ...]]) -> SubSpace:
        """Preserve a whitespace-separated spelling, or a sequence of blade tokens."""

        blades = blades.split() if isinstance(blades, str) else blades
        masks, signs = [], []
        for blade in blades:
            sign = 1
            if isinstance(blade, str) and blade.startswith(("-", "+")):
                sign = -1 if blade[0] == "-" else 1
                blade = blade[1:]
            oriented = self.algebra.parse_blade(blade)
            masks.append(oriented.mask)
            signs.append(sign * oriented.sign)
        return self.from_layout(masks, signs)

    __call__ = from_blades

    def empty(self) -> SubSpace:
        return self.from_masks(())

    def from_grades(self, grades: Iterable[int]) -> SubSpace:
        """Construct the union of the requested grades."""

        return self.from_masks({
            mask for grade in grades for mask in self.k_vector(grade).masks
        })

    @lru_cache(maxsize=None)
    def k_vector(self, grade: int) -> SubSpace:
        normalized_grade = index(grade)
        dimension = self.algebra.dimension
        if normalized_grade < 0 or normalized_grade > dimension:
            raise ValueError(
                f"grade {normalized_grade} is outside the valid range "
                f"[0, {dimension}]"
            )
        return self.from_masks(
            sum(1 << generator for generator in generators)
            for generators in combinations(range(dimension), normalized_grade)
        )

    def scalar(self) -> SubSpace:
        return self.k_vector(0)

    def vector(self) -> SubSpace:
        return self.k_vector(1)

    def bivector(self) -> SubSpace:
        return self.k_vector(2)

    def trivector(self) -> SubSpace:
        return self.k_vector(3)

    def pseudoscalar(self) -> SubSpace:
        return self.k_vector(self.algebra.dimension)

    def antivector(self) -> SubSpace:
        return self.k_vector(self.algebra.dimension - 1)

    def antibivector(self) -> SubSpace:
        return self.k_vector(self.algebra.dimension - 2)

    @lru_cache(maxsize=None)
    def full(self) -> SubSpace:
        return self.from_masks(self._blade_masks)

    @lru_cache(maxsize=None)
    def even(self) -> SubSpace:
        return self.from_masks(
            mask
            for mask in self._blade_masks
            if index(self.algebra.grade(mask)) % 2 == 0
        )

    # Descriptive aliases retained on the new surface; they do not introduce a
    # second construction path.
    multivector = full
    even_grade = even

    @lru_cache(maxsize=None)
    def __getattr__(self, name: str) -> SubSpace:
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )
        try:
            return self.from_blades(name.replace("_", " "))
        except (KeyError, ValueError):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("SubSpaceFactory is immutable")

    def __repr__(self) -> str:
        return f"SubSpaceFactory(algebra={self.algebra!r})"
