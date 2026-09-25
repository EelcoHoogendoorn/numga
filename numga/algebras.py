from __future__ import annotations

from functools import partial

from numga.algebra import Algebra
from numga.subspace import SubSpace, SubSpaceFactory


PGA2D = Algebra(
    "x+y+w0",
    subspace_factory=partial(
        SubSpaceFactory,
        default="1 x y w yw wx xy xyw",
    ),
)


PGA3D = Algebra(
    "x+y+z+w0",
    subspace_factory=partial(
        SubSpaceFactory,
        default="1 x y z w yz zx xy xw yw zw yzw zxw xyw zyx xyzw",
    ),
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


STA = Algebra("t+x-y-z-")


VGA3D = Algebra("x+y+z+")

