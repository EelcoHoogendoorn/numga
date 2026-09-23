"""Rendering quadrics on the 3-sphere, Cl(4), by projection: a quadric projects to a conic.

A body is an ellipsoid, a dual quadric placed by a motor, exactly as in the spherical quadric physics one
dimension down. Seen from an eye E, its outline on the image sphere a quarter turn ahead is
the body projected from E: the dual quadric pushed through the central projection
M = (E ∨ Point) ∧ image, the same pushforward the lens camera applies to its aperture. Pulled
into the eye's frame that is a conic in the pixel chart (1, u, v), and a pixel is inside the
outline where the conic's form is negative. Together with the eye's projected polar plane,
that conic gives a depth proportional to cot t, with t the angle travelled along the great
circle. The largest depth selects the body per pixel; only that hit is reconstructed and
shaded with its polar plane. The check is that a pixel passes the 2D test exactly when its
great circle through the body has real roots.

The spherical signature shows in one thing: identical ellipsoids placed further and further away
along a geodesic do not keep shrinking. Past a quarter turn the great circles reconverge on
the antipode of the eye, and the farthest ellipsoid looms larger than the middle one.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from numga import Algebra, NumpyContext
from numga.gatype.traits import Versor
from numga.subspace import SubSpaceFactory

ga = Algebra("x+y+z+w+", subspace_factory=partial(SubSpaceFactory, default="1 x y z w yz zx xy xw yw zw yzw zxw xyw zyx xyzw"))
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
ScreenPoint = ga.gatype.from_blades("yzw zxw xyw")
ScreenConic = ga.gatype((ga.gatype.scalar(), ScreenPoint, ScreenPoint))
ScreenPolar = ga.gatype((ga.gatype.scalar(), ScreenPoint))
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Quadric = ga.gatype((Plane, Point))           # primal quadric: polar plane <= point
DualQuadric = ga.gatype((Point, Plane))       # dual quadric: pole <= plane


def unit(points: Point) -> Point:
    """The representative of each point with positive unit weight; a unit point is a versor."""
    return (points / (mv.w & points)).normalized().with_traits(Versor)


origin = unit(mv.zyx)


def direction(coords: np.ndarray) -> Point:
    """A point a quarter turn from the origin: the direction (x, y, z) seen from it."""
    return (mv.yzw * coords[..., 0] + mv.zxw * coords[..., 1] + mv.xyw * coords[..., 2]).normalized()


def pixel_chart(fov: float, shape: tuple[int, int]) -> ScreenPoint:
    """Homogeneous screen points (1, u, v) of a pinhole grid looking along +x from the origin."""
    v, u = np.meshgrid(np.linspace(1.0, -1.0, shape[0]) * shape[0] / shape[1], np.linspace(-1.0, 1.0, shape[1]), indexing="ij")
    return mv.yzw + mv.zxw * (np.tan(fov / 2) * u.ravel()) + mv.xyw * (np.tan(fov / 2) * v.ravel())


# --- math -----------------------------------------------------------------------------
def outlines(eye_frame: Motor, surfaces: Quadric) -> Quadric:
    """The cone of rays from the eye tangent to each body: the eye's polar plane squared, less the
    form scaled by the eye's own value. Non-negative on the directions whose ray meets the body."""
    eye = eye_frame >> origin
    polar = surfaces(eye)
    return polar * (Point & polar) - surfaces * (eye & polar)


def inside(cone: Quadric, rays: ScreenPoint) -> np.ndarray:
    """Hit test of ray directions against the bodies' outline cones."""
    return (rays & cone.reshape(-1, 1)(rays)) >= 0.0


def project(eye_frame: Motor, surfaces: Quadric) -> tuple[ScreenConic, ScreenPolar]:
    """Project each quadric to its screen conic and eye-polar linear form; the eye must be off-surface."""
    eye = eye_frame >> origin
    screen = eye_frame >> ScreenPoint
    normalized = surfaces * (eye & surfaces(eye)).inverse()
    polar = screen & normalized(eye)
    return (screen & normalized(screen)) - polar * polar, polar


def reproject(conic: ScreenConic, polar: ScreenPolar, pixels: ScreenPoint) -> Scalar:
    """First-hit screen depth, proportional to cot(angle): larger is nearer; NaN means a miss."""
    with np.errstate(invalid="ignore"):
        return -polar(pixels) + (-conic(pixels, pixels)).square_root()
