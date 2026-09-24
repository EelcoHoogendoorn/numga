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
        "odd",
        "odd_grade",
        "quadvector",
        "antitrivector",
        "antiquadvector",
        "scalar_pseudoscalar",
        "nonscalar",
        "self_reverse",
        "mod4",
        "k_reflection",
        "reflection",
        "bireflection",
        "trireflection",
        "quadreflection",
        "degenerate",
        "nondegenerate",
        "scalar_degenerate",
        "translator",
        "blade",
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

    @lru_cache(maxsize=None)
    def odd(self) -> SubSpace:
        return self.from_grades(range(1, self.algebra.dimension + 1, 2))

    def quadvector(self) -> SubSpace:
        return self.k_vector(4)

    def antitrivector(self) -> SubSpace:
        return self.k_vector(self.algebra.dimension - 3)

    def antiquadvector(self) -> SubSpace:
        return self.k_vector(self.algebra.dimension - 4)

    def scalar_pseudoscalar(self) -> SubSpace:
        return self.from_grades((0, self.algebra.dimension))

    def nonscalar(self) -> SubSpace:
        return self.from_grades(range(1, self.algebra.dimension + 1))

    @lru_cache(maxsize=None)
    def self_reverse(self) -> SubSpace:
        """The grades a reverse leaves unchanged: 0, 1, 4, 5, 8, ..."""
        return self.from_grades(k for k in range(self.algebra.dimension + 1) if k // 2 % 2 == 0)

    @lru_cache(maxsize=None)
    def mod4(self) -> SubSpace:
        """Grades 0, 4, 8, ...: where a motor times its reverse lands."""
        return self.from_grades(range(0, self.algebra.dimension + 1, 4))

    @lru_cache(maxsize=None)
    def k_reflection(self, k: int) -> SubSpace:
        """Where a product of k vectors lives: grades k, k - 2, ... down to 0 or 1."""
        return self.from_grades(range(k % 2, min(k, self.algebra.dimension) + 1, 2))

    def reflection(self) -> SubSpace:
        return self.k_reflection(1)

    def bireflection(self) -> SubSpace:
        return self.k_reflection(2)

    def trireflection(self) -> SubSpace:
        return self.k_reflection(3)

    def quadreflection(self) -> SubSpace:
        return self.k_reflection(4)

    @lru_cache(maxsize=None)
    def degenerate(self) -> SubSpace:
        """Every blade containing a null generator."""
        return self.full().degenerate()

    @lru_cache(maxsize=None)
    def nondegenerate(self) -> SubSpace:
        """Every blade free of null generators."""
        return self.full().nondegenerate()

    def scalar_degenerate(self) -> SubSpace:
        return self.scalar().union(self.degenerate())

    @lru_cache(maxsize=None)
    def translator(self) -> SubSpace:
        """The scalar and the bivectors containing a null generator: where translators live."""
        return self.bireflection().difference(self.bivector().nondegenerate())

    def blade(self, mask: int) -> SubSpace:
        return self.from_masks((mask,))

    # Other names for the even and odd subspaces.
    odd_grade = odd
    motor = even

    @lru_cache(maxsize=None)
    def named(self) -> tuple[tuple[str, SubSpace], ...]:
        """The named subspaces of this algebra, in order of precedence where two coincide."""
        n = self.algebra.dimension
        grades = (("scalar", 0), ("pseudoscalar", n), ("vector", 1), ("bivector", 2),
                  ("antivector", n - 1), ("antibivector", n - 2), ("trivector", 3))
        return (
            (("empty", self.empty()),)
            + tuple((name, self.k_vector(k)) for name, k in grades if 0 <= k <= n)
            + (("even", self.even()), ("odd", self.odd()), ("full", self.full()))
        )

    # Descriptive aliases; they do not introduce a second construction path.
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
