from __future__ import annotations

from functools import partial

from typing import Any

from numga.algebra import Algebra
from numga.extensor import Extensor
from numga.multivector import MultivectorFactory
from numga.subspace import SubSpace, SubSpaceFactory


class ProjectiveMultivectorFactory(MultivectorFactory):
    """Value constructors for a projective algebra, one with a single degenerate basis vector, read
    plane-based: points and directions from Euclidean coordinates."""

    def point(self, coordinates: Any) -> Extensor:
        """Points of weight one at these coordinates `[..., d]`: the dual of the plane through the origin
        with these coordinates, plus the degenerate basis vector. Intrinsic, as points are."""
        if self.algebra.degenerate_mask.bit_count() != 1:
            raise TypeError(f"points of weight one need a single degenerate basis vector; {self.algebra} has none or several")
        planes = self(self.algebra.gatype.vector().derive.nondegenerate, coordinates)
        return (planes + self.blade(self.algebra.degenerate_mask)).dual()

    def direction(self, coordinates: Any) -> Extensor:
        """Points at infinity in these directions `[..., d]`: the dual of the plane through the origin
        with these coordinates. Intrinsic, as points are."""
        return self(self.algebra.gatype.vector().derive.nondegenerate, coordinates).dual()


PGA2D = Algebra(
    "x+y+w0",
    subspace_factory=partial(
        SubSpaceFactory,
        default="1 x y w yw wx xy xyw",
    ),
    multivector_factory=ProjectiveMultivectorFactory,
)


PGA3D = Algebra(
    "w0x+y+z+",
    subspace_factory=partial(
        SubSpaceFactory,
        default="1 x y z w yz zx xy wx wy wz wzy wxz wyx xyz wxyz",
    ),
    multivector_factory=ProjectiveMultivectorFactory,
)


class SphericalSubSpaceFactory(SubSpaceFactory):
    """Subspace factory for spherical geometry on the sphere, in the algebra `x+y+z+`.

    Maps cutting planes / great circles to Grade 1 vectors (x, y, z) and
    points on the sphere to Grade 2 antivectors with right-handed cyclic basis (yz, zx, xy).
    """

    constructor_names = SubSpaceFactory.constructor_names + (
        "plane",
        "point",
        "line",
    )
    intrinsic_constructor_names = SubSpaceFactory.intrinsic_constructor_names + ("point",)

    def plane(self) -> SubSpace:
        """Grade 1 cutting planes / great circles on the sphere."""
        return self.vector()

    def point(self) -> SubSpace:
        """Grade 2 antivectors representing points on the sphere."""
        return self.antivector()

    def line(self) -> SubSpace:
        """Great-circle geodesic lines on the sphere (identical to cutting planes)."""
        return self.plane()


Spherical3D = Algebra(
    "x+y+z+",
    subspace_factory=partial(
        SphericalSubSpaceFactory,
        default="1 x y z yz zx xy xyz",
    ),
)


STA = Algebra(
    "t+x-y-z-",
    subspace_factory=partial(
        SubSpaceFactory,
        default="1 t x y z xt yt zt yz zx xy xyz tyz tzx txy txyz",
    ),
)


VGA3D = Algebra(
    "x+y+z+",
    subspace_factory=partial(
        SubSpaceFactory,
        default="1 x y z yz zx xy xyz",
    ),
)

