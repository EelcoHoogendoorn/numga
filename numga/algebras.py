"""Stock geometric algebras pair standard metric signatures with convention-matched basis layouts
and geometry-specific constructors. Right-handed cyclic bivector bases (`yz`, `zx`, `xy`) and
homogeneous projective bases (`x+y+w0` for `PGA2D`, `w0x+y+z+` for `PGA3D`) come preconfigured on
`VGA3D`, `STA`, `PGA2D`, `PGA3D`, and `Spherical3D`, alongside planar vector algebra `VGA2D`
(`x+y+`), planar spacetime `STA2D` (`x+y+t-`), and 3D conformal space `CGA3D` (`x+y+z+w+e-`).
Because `Algebra` interns by both description and factory callables, stock algebras carrying custom
layouts or constructors remain distinct from plain `Algebra(specification)` instances of the same
signature.

Plane-based projective algebras with a single null generator use `ProjectiveMultivectorFactory` to
construct intrinsic points and ideal directions directly from Euclidean coordinates by dualizing
offset or origin planes. On the unit sphere in `x+y+z+`, `SphericalSubSpaceFactory` adds `plane`,
`line`, and intrinsic `point` subspaces so great circles take grade-1 cutting planes and spherical
points take grade-2 antivectors.
"""

from __future__ import annotations

from functools import partial

from numga.algebra import Algebra
from numga.extensor import Extensor
from numga.multivector import MultivectorFactory
from numga.subspace import SubSpace, SubSpaceFactory


class ProjectiveMultivectorFactory(MultivectorFactory):
    """Euclidean coordinate helpers construct intrinsic points and ideal directions in plane-based
    projective algebras with a single null generator."""

    def point(self, coordinates) -> Extensor:
        """Dualizing a non-degenerate plane offset by the unit null generator maps Euclidean
        `coordinates` of shape `[..., d]` to intrinsic points of unit weight."""
        if self.algebra.degenerate_mask.bit_count() != 1:
            raise TypeError(f"points of weight one need a single degenerate basis vector; {self.algebra} has none or several")
        planes = self(self.algebra.gatype.vector().derive.nondegenerate, coordinates)
        return (planes + self.blade(self.algebra.degenerate_mask)).dual()

    def direction(self, coordinates) -> Extensor:
        """Dualizing a non-degenerate plane through the origin maps Euclidean direction `coordinates`
        of shape `[..., d]` to intrinsic ideal points at infinity."""
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
    """On the unit sphere in `x+y+z+`, grade-1 vectors represent cutting planes and great-circle
    lines while intrinsic grade-2 antivectors `(yz, zx, xy)` represent points."""

    constructor_names = SubSpaceFactory.constructor_names + (
        "plane",
        "point",
        "line",
    )
    intrinsic_constructor_names = SubSpaceFactory.intrinsic_constructor_names + ("point",)

    def plane(self) -> SubSpace:
        """Great circles on the sphere correspond to grade-1 cutting planes through the origin."""
        return self.vector()

    def point(self) -> SubSpace:
        """Points on the sphere occupy the grade-2 antivector subspace `(yz, zx, xy)`."""
        return self.antivector()

    def line(self) -> SubSpace:
        """Geodesic lines on the sphere coincide with great-circle cutting planes."""
        return self.plane()


Spherical3D = Algebra(
    "x+y+z+",
    subspace_factory=partial(
        SphericalSubSpaceFactory,
        default="1 x y z yz zx xy xyz",
    ),
)


STA = Algebra(
    "x+y+z+t-",
    subspace_factory=partial(
        SubSpaceFactory,
        default="1 x y z t yz zx xy xt yt zt xyz yzt zxt xyt xyzt",
    ),
)


VGA3D = Algebra(
    "x+y+z+",
    subspace_factory=partial(
        SubSpaceFactory,
        default="1 x y z yz zx xy xyz",
    ),
)


VGA2D = Algebra("x+y+")

CGA3D = Algebra("x+y+z+w+e-")

STA2D = Algebra("x+y+t-")
