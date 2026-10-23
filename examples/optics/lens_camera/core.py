"""A zoom camera with depth of field: cones pulled through non-rigid maps, in flat or spherical space.

An ideal thin lens is a projective collineation of space, `Point + origin * (home & Point) / focal`,
and its action on lines, `Line - (origin & (Line ^ home)) / focal`, is the join of the images: the
train composes either way. The rays a scene point sends through the aperture form a cone, the
pullback of a ball through the central projection from the point onto the aperture plane.
The train carries that cone to the image cone, a pullback through the inverse of its point
map, and the sensor cuts the image cone in the point's blur conic. Nothing asks where the
point focuses; the cone's vertex is wherever the collineation put it.

The core logic never names the metric: with w squaring to 1 instead of 0 the same lens maps,
quadrics, pullbacks and checks run on the 3-sphere, where translators are rotations toward the
pole and a ball of "radius" r is a ball of angular radius `np.arctan(r)`. Only the point constructor, which
motors a chosen origin, and the chart readouts for drawing know which space they are in.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D
from numga.gatype.traits import Versor


ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
SensorPoint = Point.derive.containing("x")
SensorPlane = Plane.derive.within("y z w")
# Lines are joins of points.
Line = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
# A collineation maps points to points.
PointMap = ga.gatype((Point, Point))          # Point <- Point
LineMap = ga.gatype((Line, Line))
# A primal quadric maps a point to its polar plane.
Quadric = ga.gatype((Plane, Point))           # Plane <- Point
# A dual quadric maps a plane to its pole.
DualQuadric = ga.gatype((Point, Plane))       # Point <- Plane
# A camera maps a scene point and a pupil point to a sensor point.
Camera = ga.gatype((Point, Point, Point))     # Point <- (Point, Point)


# --- plumbing -------------------------------------------------------------------------
def unit(points: Point) -> Point:
    """The representative of each point with positive unit weight; a unit point is a versor."""
    return (points / (mv.w & points)).normalized().with_traits(Versor)


# Every element is built in its home frame: centred on the origin, in the plane x == 0.
origin = unit(mv.w.dual())                                                      # Point
home = mv.x                                                                     # Plane


def point(coords: np.ndarray) -> Point:
    """The origin carried by the translator with the given displacement."""
    return ((mv.xw * coords[..., 0] + mv.yw * coords[..., 1] + mv.zw * coords[..., 2]) * 0.5).exp() >> origin


# --- math -----------------------------------------------------------------------------
def thin_lens(focal: np.ndarray) -> tuple[PointMap, LineMap]:
    """Thin lenses of the given focal lengths in the home plane, as collineations of points and
    as the induced maps on lines."""
    points = Point + origin * (home & Point) / focal                            # PointMap
    lines = Line - (origin & (Line ^ home)) / focal                             # LineMap
    return points, lines


def ball(radius: float) -> Quadric:
    """A ball of the given radius about the origin; its section with the home plane is the aperture rim."""
    return mv.x * (mv.x & Point) + mv.y * (mv.y & Point) + mv.z * (mv.z & Point) - mv.w * (mv.w & Point) * radius**2


def expose(
    scene: Point, subject: Point, focal: np.ndarray, placements: Motor, focus: Point, tilt: Motor, radius: float, rim: Point,
):
    """Place the lenses, focus the sensor, and carry each scene point's aperture cone to the sensor side.

    Returns the lens train's collineation, the camera map, the sensor frame, the image cone of
    every scene point, the element planes [front, rear, sensor], and a fan of rays from the
    subject through the aperture rim, as its points on successive planes [plane + 1, ray].
    """
    # Place the lenses [front, rear], then compose their point maps and their line maps.
    lens_points, lens_lines = thin_lens(focal)
    points = placements >> lens_points(placements << Point)
    lines = placements >> lens_lines(placements << Line)
    front, rear = lines[0], lines[1]
    front_plane, rear_plane = placements[0] >> home, placements[1] >> home
    collineation = points[1](points[0])                                         # PointMap
    train = rear(front)
    pupil_ball = placements[0] >> ball(radius)(placements[0] << Point)          # Quadric

    # The focus point's image places the sensor; tilt turns it about that image.
    image = unit(collineation(focus))                                           # Point
    frame = (image / origin).square_root() * tilt                               # Motor
    sensor = frame >> home                                                      # Plane
    camera = train(Point & Point) ^ sensor                                      # Camera

    # Project through each subject onto the pupil, then pull back its quadric.
    # Pull back once more through the inverse lens train to get the image cones.
    project = (scene & Point) ^ front_plane
    cone = project.adjugate()(pupil_ball(project))                              # Quadric
    back = collineation.inverse()                                               # PointMap
    cones = back.adjugate()(cone(back))                                         # Quadric

    rays = subject & (placements[0] >> rim)                                     # Line
    legs = stack([subject.broadcast_to(rays.shape), rays ^ front_plane,         # Point
                         front(rays) ^ rear_plane, rear(front(rays)) ^ sensor])
    return collineation, camera, frame, cones, stack([front_plane, rear_plane, sensor]), legs


def section(cone: Quadric, start: Point, frame: Motor, samples: int) -> Point:
    """Boundary of the sensor's section of a cone, traced from a point inside it along the sensor plane."""
    theta = np.linspace(0.0, 2 * np.pi, samples)
    across = frame >> ((mv.yz * (-theta / 2)).exp() >> mv.y.dual())
    a, b, c = across & cone(across), across & cone(start), start & cone(start)
    # Rounding-level negative at the vertex:
    root = (b * b - a * c).clip(0.0, np.inf).square_root()
    return start + across * ((root - b) / a)
