"""Each algebra owns a flyweight factory that constructs and strongly interns `SubSpace` instances.
When an algebra supplies a `default` full-basis layout string (such as
`"1 x y z w yz zx xy wx wy wz wzy wxz wyx xyz wxyz"` on `PGA3D`), mask-based constructors
(`from_masks`, `from_grades`, `k_vector`, `even`, `odd`, `full`) restrict that layout so standard
subspaces inherit the algebra's chosen coordinate ordering and signs; without `default`, masks sort
by ascending grade and integer bit pattern with `+1` signs.

Explicit spellings via `from_blades` (also callable as `algebra.subspace("yz zx xy")` or accessed
as `algebra.subspace.yz_zx_xy`) and `from_layout` preserve the caller's exact blade order and signs
without consulting `default`, pooling each `(masks, signs)` pair once in the factory's flyweight
table. Keeping `algebra.blade_masks` as a lazy `range` makes constructing the factory `O(1)` in
high dimensions until `full()` or `even()` is requested.

`constructor_names` and `intrinsic_constructor_names` declare the public constructor protocol
mirrored onto `GATypeFactory` and `MultivectorFactory`. Antigrade constructors (`antiscalar` down
through `antipseudoscalar`) list under `intrinsic_constructor_names` because in plane-based
algebras they represent points and joins of points measured against the volume element, whereas
`scalar`, `vector`, `bivector`, `pseudoscalar`, and versor subspaces default to extrinsic
orientation. Subclasses such as `SphericalSubSpaceFactory` extend both tuples alongside custom
methods so new geometric constructors propagate to types and multivectors.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from itertools import combinations
from operator import index
from typing import TYPE_CHECKING

from numga.flyweight import FlyweightFactory

from .subspace import SubSpace, _normalize_masks

if TYPE_CHECKING:
    from numga.algebra import Algebra


class SubSpaceFactory(FlyweightFactory[SubSpace]):
    """An immutable per-algebra namespace pools named grade supports and custom signed blade
    layouts."""

    __slots__ = ("_algebra", "_blade_masks", "_default")

    # Public constructor names mirrored onto GATypeFactory and MultivectorFactory.
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
        "antiscalar",
        "antivector",
        "antibivector",
        "antipseudoscalar",
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
    # Antigrade constructors default to intrinsic orientation on GATypeFactory.
    intrinsic_constructor_names = ("antiscalar", "antivector", "antibivector", "antitrivector", "antiquadvector",
                                   "antipseudoscalar")

    def __init__(self, algebra: Algebra, default: str | None = None) -> None:
        dimension = algebra.dimension
        super().__init__()
        blade_masks = getattr(algebra, "blade_masks", range(1 << dimension))
        # Keeping range objects lazy avoids O(2**dimension) allocation when building the factory.
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
        """Canonical masks assemble into a `SubSpace` ordered and signed according to the algebra's
        default layout, or ascending `(grade, mask)` when no default is set."""
        normalized = _normalize_masks(self.algebra, masks)
        if self._default is not None:
            return self._default.restrict(normalized)
        return self.from_layout(normalized, (1,) * len(normalized))

    def from_layout(self, masks: Iterable[int], signs: Iterable[int]) -> SubSpace:
        """Interning explicit masks and `+1`/`-1` signs preserves the caller's exact coordinate
        order without applying the default layout."""
        masks, signs = tuple(masks), tuple(signs)
        return self.factory_construct(
            (masks, signs),
            lambda: SubSpace(self.algebra, masks, signs),
        )

    def from_blades(self, blades: str | Iterable) -> SubSpace:
        """Whitespace-separated blade strings such as `"yz zx xy"` or sequences of blade tokens
        construct a `SubSpace` in the given order and orientation signs."""
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
        """Grades in `grades` combine into a single `SubSpace` in the algebra's default blade
        layout."""
        return self.from_masks({
            mask for grade in grades for mask in self.k_vector(grade).masks
        })

    @lru_cache(maxsize=None)
    def k_vector(self, grade: int) -> SubSpace:
        """All basis blades of exact `grade` form the `grade`-vector subspace."""
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

    def antiscalar(self) -> SubSpace:
        return self.k_vector(self.algebra.dimension)

    def antivector(self) -> SubSpace:
        return self.k_vector(self.algebra.dimension - 1)

    def antibivector(self) -> SubSpace:
        return self.k_vector(self.algebra.dimension - 2)

    def antipseudoscalar(self) -> SubSpace:
        """Complementing the pseudoscalar gives grade `0`, which as a `GAType` carries intrinsic
        orientation to represent oriented volume joins."""
        return self.k_vector(0)

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
        """Reversion leaves grades `k` with `k // 2` even (`0, 1, 4, 5, ...`) unchanged in sign."""
        return self.from_grades(k for k in range(self.algebra.dimension + 1) if k // 2 % 2 == 0)

    @lru_cache(maxsize=None)
    def mod4(self) -> SubSpace:
        """Grades divisible by four (`0, 4, 8, ...`) span the support of an even element multiplied
        by its reverse."""
        return self.from_grades(range(0, self.algebra.dimension + 1, 4))

    @lru_cache(maxsize=None)
    def k_reflection(self, n_reflections: int) -> SubSpace:
        """Multiplying `n_reflections` vectors spans grades of matching parity up to `n_reflections`
        and bounded by the algebra dimension."""
        return self.from_grades(
            range(n_reflections % 2, min(n_reflections, self.algebra.dimension) + 1, 2)
        )

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
        """Blades containing at least one null generator span the degenerate ideal of the algebra."""
        return self.full().degenerate()

    @lru_cache(maxsize=None)
    def nondegenerate(self) -> SubSpace:
        """Blades containing no null generator span the non-degenerate complement of the radical."""
        return self.full().nondegenerate()

    def scalar_degenerate(self) -> SubSpace:
        return self.scalar().union(self.degenerate())

    @lru_cache(maxsize=None)
    def translator(self) -> SubSpace:
        """Scalars together with bivectors touching a null generator span the support of projective
        translators."""
        return self.bireflection().difference(self.bivector().nondegenerate())

    def blade(self, mask: int) -> SubSpace:
        return self.from_masks((mask,))

    odd_grade = odd
    motor = even

    @lru_cache(maxsize=None)
    def named(self) -> tuple:
        """Standard named subspaces appear in display-precedence order so low-dimensional
        coincidences resolve deterministically."""
        n = self.algebra.dimension
        grades = (("scalar", 0), ("pseudoscalar", n), ("vector", 1), ("bivector", 2),
                  ("antivector", n - 1), ("antibivector", n - 2), ("trivector", 3))
        return (
            (("empty", self.empty()),)
            + tuple((name, self.k_vector(k)) for name, k in grades if 0 <= k <= n)
            + (("even", self.even()), ("odd", self.odd()), ("full", self.full()))
        )

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
