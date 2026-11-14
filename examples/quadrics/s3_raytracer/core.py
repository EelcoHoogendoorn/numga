"""Projection of quadric ellipsoids on the 3-sphere to screen conics and cotangent depths.

Points of the 3-sphere are intrinsic antivectors, great spheres are extrinsic planes, and an
ellipsoid is modelled as a dual quadric placed by a motor. Seen from an eye, a body's outline on the
image sphere a quarter turn ahead is the cone of tangent rays pushed into the eye's screen chart,
where it forms a conic negative on covered pixels. Paired with the eye's polar plane under the body,
that conic gives each hit's depth as the cotangent of the arc travelled along the pixel's great circle.

Along a geodesic line of sight, identical ellipsoids do not keep shrinking with distance. Past a
quarter turn, the great circles leaving the eye reconverge toward its antipode, so an ellipsoid past
the equator subtends a wider visual angle than one nearer the equator.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.gatype.traits import Versor

ga = Algebra("x+y+z+w+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Point = ga.gatype.antivector()                                                 # Antivector-
Plane = ga.gatype.vector()                                                     # Vector+
# Directions from the origin correspond to duals of planes through it, spanned by x, y, and z.
ScreenPlane = Plane.derive.within(Algebra("x+y+z+"))                           # Vector+
ScreenPoint = ScreenPlane.derive.dual                                          # Antivector-
ScreenConic = ga.gatype((Scalar, ScreenPoint, ScreenPoint))                    # Scalar+ <- (ScreenPoint-, ScreenPoint-)
ScreenPolar = ga.gatype((Scalar, ScreenPoint))                                 # Scalar+ <- ScreenPoint-
Motor = ga.gatype.rotor()                                                      # Rotor+
# A primal quadric maps a point to its polar plane.
Quadric = ga.gatype((Plane, Point))                                            # Plane+ <- Point-


def unit(points: Point) -> Point:
    """Dividing `points` (`[...] Point-`) by their `w` weight and normalizing gives the positive-weight unit versor representatives."""
    return (points / (mv.w & points)).normalized().with_traits(Versor)         # [...] Point-


origin = unit(mv.w.dual())                                                     # [] Point-


def direction(coordinates: np.ndarray) -> Point:
    """Dualizing and normalizing `coordinates` (`[..., 3]`) gives unit equatorial points (`[...] Point-`) a quarter turn from `origin`."""
    return mv(ScreenPlane, coordinates).dual().normalized()                    # [...] Point-


# --- math -----------------------------------------------------------------------------
def placed_ellipsoids(
    half_widths: np.ndarray,
    distances: np.ndarray,
    sideways: np.ndarray,
    upward: np.ndarray,
    turn: float,
) -> Quadric:
    """Placing a central ellipsoid of angular `half_widths` (`[3]`) by geodesic offsets (`[n_bodies]`) and `turn` gives primal quadrics (`[n_bodies] Plane+ <- Point-`)."""
    # Principal directions a quarter turn from `origin` weighted by squared tangent half-widths minus the origin dyad close the dual ellipsoid.
    axes = direction(np.eye(3))                                                # [3] Point-
    ellipsoid = (axes * (axes & Plane) * np.tan(half_widths) ** 2).sum(axis=0) - origin * (origin & Plane)  # [] Point- <- Plane+
    to_distance = (mv.xw * (distances / 2)).exp()                              # [...] Motor+
    to_side = (mv.yw * (sideways / 2)).exp()                                   # [...] Motor+
    to_height = (mv.zw * (upward / 2)).exp()                                   # [...] Motor+
    spin = (mv.xy * (turn / 2)).exp()                                          # [] Motor+
    placements = to_distance * to_side * to_height * spin                      # [...] Motor+
    bodies = placements >> ellipsoid(placements << Plane)                      # [...] Point- <- Plane+
    return bodies.inverse()                                                    # [...] Plane+ <- Point-


def walk_frames(walk_distance: float, n_frames: int) -> Iterator[Motor]:
    """Yields `n_frames` eye-frame motors (`[] Motor+`) advancing by `walk_distance` along the forward geodesic."""
    for step in np.linspace(0, walk_distance, n_frames, endpoint=False):
        yield (mv.xw * (step / 2)).exp()                                       # [] Motor+


def outlines(eye_frame: Motor, surfaces: Quadric) -> Quadric:
    """Subtracting each quadric scaled by the eye's own value from the squared polar plane of the eye gives the tangent outline cones (`[...] Plane+ <- Point-`)."""
    eye = eye_frame >> origin                                                  # [] Point-
    polar = surfaces(eye)                                                      # [...] Plane+
    return polar * (Point & polar) - surfaces * (eye & polar)                  # [...] Plane+ <- Point-


def project(eye_frame: Motor, surfaces: Quadric) -> tuple[ScreenConic, ScreenPolar]:
    """Pulling `surfaces` (`[...] Plane+ <- Point-`) back to the screen of `eye_frame` gives the outline conics and eye-polar linear forms."""
    eye = eye_frame >> origin                                                  # [] Point-
    screen = eye_frame >> ScreenPoint                                          # [] Point- <- ScreenPoint-
    normalized = surfaces / (eye & surfaces(eye))                              # [...] Plane+ <- Point-
    polar = screen & normalized(eye)                                           # [...] Scalar+ <- ScreenPoint-
    return (screen & normalized(screen)) - polar * polar, polar                # [...] Scalar+ <- (ScreenPoint-, ScreenPoint-), [...] Scalar+ <- ScreenPoint-


def reproject(conic: ScreenConic, polar: ScreenPolar, pixels: ScreenPoint) -> Scalar:
    """The larger root of the screen quadratic from `conic` and `polar` gives each pixel's first-hit cotangent depth (`[...] Scalar+`, `NaN` on misses)."""
    with np.errstate(invalid="ignore"):
        return -polar(pixels) + (-conic(pixels, pixels)).square_root()         # [...] Scalar+


# --- plumbing -------------------------------------------------------------------------
def pixel_chart(field_of_view: float, shape: tuple[int, int]) -> ScreenPoint:
    """Pinhole grid coordinates of `shape` (`(n_rows, n_columns)`) and `field_of_view` form homogeneous screen points (`[n_rows * n_columns] ScreenPoint-`)."""
    n_rows, n_columns = shape
    vertical, horizontal = np.meshgrid(
        np.linspace(1, -1, n_rows) * n_rows / n_columns,
        np.linspace(-1, 1, n_columns),
        indexing="ij",
    )
    half_width = np.tan(field_of_view / 2)
    axes = mv(ScreenPlane, np.eye(3)).dual()                                   # [3] ScreenPoint-
    return axes[0] + axes[1] * (half_width * horizontal.ravel()) + axes[2] * (half_width * vertical.ravel())  # [...] ScreenPoint-
