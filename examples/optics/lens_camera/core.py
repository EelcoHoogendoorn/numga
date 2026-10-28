"""This module models a zoom camera with depth of field by pulling cones through non-rigid maps, in flat or spherical space.

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
Point = ga.gatype.antivector()                                                  # Antivector-
Plane = ga.gatype.vector()                                                      # Vector+
SensorPoint = Point.derive.containing("x")                                      # Antivector-
SensorPlane = Plane.derive.within("y z w")                                      # Vector+
# Lines are joins of points.
Line = ga.gatype.antibivector()                                                 # Antibivector-
Motor = ga.gatype.rotor()                                                       # Rotor+
# A collineation maps points to points.
PointMap = ga.gatype((Point, Point))                                            # Point- <- Point-
LineMap = ga.gatype((Line, Line))                                               # Line- <- Line-
# A primal quadric maps a point to its polar plane.
Quadric = ga.gatype((Plane, Point))                                             # Plane+ <- Point-
# A camera maps a scene point and a pupil point to a sensor point.
Camera = ga.gatype((Point, Point, Point))                                       # Point- <- (Point-, Point-)


# --- plumbing -------------------------------------------------------------------------
def unit(points: Point) -> Point:
    """The representative of each point with positive unit weight; a unit point is a versor."""
    return (points / (mv.w & points)).normalized().with_traits(Versor)


# Every element is built in its home frame: centred on the origin, in the plane x == 0.
origin = unit(mv.w.dual())                                                      # [] Point-
home = mv.x                                                                     # [] Plane+


def point(coords: np.ndarray) -> Point:
    """The origin carried by the translator with the given displacement."""
    translators = ((stack((mv.xw, mv.yw, mv.zw)) * coords).sum(-1) / 2).exp()    # [...] Motor+
    return translators >> origin


# --- math -----------------------------------------------------------------------------
def thin_lens(focal: np.ndarray) -> tuple[PointMap, LineMap]:
    """Thin lenses of the given focal lengths in the home plane, as collineations of points and
    as the induced maps on lines."""
    points = Point + origin * (home & Point) / focal                            # [lenses] Point- <- Point-
    lines = Line - (origin & (Line ^ home)) / focal                             # [lenses] Line- <- Line-
    return points, lines


def ball(radius: float) -> Quadric:
    """A ball of the given radius about the origin; its section with the home plane is the aperture rim."""
    coordinate_planes = stack((mv.x, mv.y, mv.z))                               # [3] Plane+
    return (coordinate_planes * (coordinate_planes & Point)).sum(-1) - mv.w * (mv.w & Point) * radius**2


def expose(
    scene: Point, subject: Point, focal: np.ndarray, placements: Motor, focus: Point, tilt: Motor, radius: float, rim: Point,
) -> tuple[PointMap, Camera, Motor, Quadric, Plane, Point]:
    """The scene seen through the lenses at their placements, with the sensor focused on `focus` and tilted: each
    scene point's aperture cone carried to the sensor side.

    The result is the lens train's collineation, the camera map, the sensor frame, the image cone of
    every scene point, the element planes [front, rear, sensor], and a fan of rays from the
    subject through the aperture rim, as its points on successive planes [planes + 1, rays].
    """
    # Place the lenses [front, rear], then compose their point maps and their line maps.
    lens_points, lens_lines = thin_lens(focal)                                  # [lenses] Point- <- Point-, Line- <- Line-
    point_maps = placements >> lens_points(placements << Point)                 # [lenses] Point- <- Point-
    line_maps = placements >> lens_lines(placements << Line)                    # [lenses] Line- <- Line-
    front, rear = line_maps[0], line_maps[1]                                    # [] Line- <- Line-, Line- <- Line-
    front_plane, rear_plane = placements[0] >> home, placements[1] >> home      # [] Plane+, Plane+
    collineation = point_maps[1](point_maps[0])                                 # [] Point- <- Point-
    train = rear(front)                                                         # [] Line- <- Line-
    pupil_ball = placements[0] >> ball(radius)(placements[0] << Point)          # [] Plane+ <- Point-

    # The focus point's image places the sensor; tilt turns it about that image.
    image = unit(collineation(focus))                                           # [] Point-
    frame = (image / origin).square_root() * tilt                               # [] Motor+
    sensor = frame >> home                                                      # [] Plane+
    camera = train(Point & Point) ^ sensor                                      # [] Point- <- (Point-, Point-)

    # Project through each scene point onto the pupil, then pull back its quadric.
    # Pull back once more through the inverse lens train to get the image cones.
    projection = (scene & Point) ^ front_plane                                  # [depths, heights, widths] Point- <- Point-
    aperture_cones = projection.adjugate()(pupil_ball(projection))              # [depths, heights, widths] Plane+ <- Point-
    inverse_collineation = collineation.inverse()                               # [] Point- <- Point-
    image_cones = inverse_collineation.adjugate()(aperture_cones(inverse_collineation))   # [depths, heights, widths] Plane+ <- Point-

    rays = subject & (placements[0] >> rim)                                     # [rays] Line-
    legs = stack([subject.broadcast_to(rays.shape), rays ^ front_plane,         # [planes + 1, rays] Point-
                  front(rays) ^ rear_plane, rear(front(rays)) ^ sensor])
    return collineation, camera, frame, image_cones, stack([front_plane, rear_plane, sensor]), legs


def section(cone: Quadric, start: Point, frame: Motor, n_samples: int) -> Point:
    """The boundary of the sensor's section of a cone, traced from a point inside it along the sensor plane."""
    angles = np.linspace(0.0, 2 * np.pi, n_samples)
    turns = (mv.yz * (-angles / 2)).exp()                                       # [samples] Motor+
    sensor_directions = frame >> (turns >> mv.y.dual())                         # [samples] Point-
    # The cone's form on `start + sensor_directions * step` is quadratic in the step, with these coefficients.
    a, b, c = sensor_directions & cone(sensor_directions), sensor_directions & cone(start), start & cone(start)
    # The discriminant is negative by rounding at the cone's vertex, where the section is a single point.
    root = (b * b - a * c).clip(0.0, np.inf).square_root()
    return start + sensor_directions * ((root - b) / a)
