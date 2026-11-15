"""Projective collineations pull aperture cones through a two-lens zoom camera onto a tilted sensor in three dimensions.

An ideal thin lens is modelled as a projective collineation of space, `Point + origin * (home & Point) / focal`, whose induced action on lines `Line - (origin & (Line ^ home)) / focal` equals the join of the point images, so an optical train composes identically on points and on lines. The rays that a scene point sends through the aperture form a quadric cone obtained by pulling the aperture ball back through central projection from the scene point onto the aperture plane. Pulling that aperture cone forward through the inverse point collineation of the lens train gives the image cone on the sensor side, and intersecting with the sensor plane cuts out the blur conic directly without solving for an intermediate focus distance.

Because the cone pullbacks and lens maps never invoke a Euclidean metric, changing the weight generator's square from zero to one runs the same collineations, quadrics, and pullbacks on the 3-sphere, where translations become rotations from the origin pole.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import PGA3D
from numga.gatype.traits import Versor

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()                                                  # Antivector-
Plane = ga.gatype.vector()                                                      # Vector+
SensorPlane = Plane.derive.within(Algebra("w0y+z+"))                            # Vector+
SensorPoint = SensorPlane.derive.dual                                           # Antivector-
# Lines are joins of points.
Line = ga.gatype.antibivector()                                                 # Antibivector-
Motor = ga.gatype.rotor()                                                       # Rotor+
# A collineation maps points to points.
PointMap = ga.gatype((Point, Point))                                            # Point- <- Point-
LineMap = ga.gatype((Line, Line))                                               # Line- <- Line-
# A primal quadric maps each point to its polar plane.
Quadric = ga.gatype((Plane, Point))                                             # Plane+ <- Point-
# A camera maps a scene point and a pupil point to a sensor point.
Camera = ga.gatype((Point, Point, Point))                                       # Point- <- (Point-, Point-)


# --- plumbing -------------------------------------------------------------------------
def unit_point(points: Point) -> Point:
    """Dividing by point weight and normalizing gives the positive unit-weight versor representative (`[...] Point-`)."""
    return (points / (mv.w & points)).normalized().with_traits(Versor)


# Every optical element is defined in its home frame, centred on the origin in the plane x == 0.
origin = unit_point(mv.w.dual())                                                # [] Point-
home = mv.x                                                                     # [] Plane+


def scene_point(coords: np.ndarray) -> Point:
    """Translating `origin` by `coords` (`[..., 3]`) constructs unit-weight points (`[...] Point-`)."""
    translators = ((stack((mv.xw, mv.yw, mv.zw)) * coords).sum(-1) / 2).exp()   # [...] Rotor+
    return translators >> origin


# --- math -----------------------------------------------------------------------------
def thin_lens_maps(focal: np.ndarray) -> tuple[PointMap, LineMap]:
    """Focal lengths `focal` (`[n_lenses]`) in the home plane define point collineations (`[n_lenses] Point- <- Point-`) and induced line maps (`[n_lenses] Line- <- Line-`)."""
    point_maps = Point + origin * (home & Point) / focal                        # [...] Point- <- Point-
    line_maps = Line - (origin & (Line ^ home)) / focal                         # [...] Line- <- Line-
    return point_maps, line_maps


def aperture_ball(radius: float) -> Quadric:
    """Radius `radius` around `origin` defines a spherical primal quadric (`[] Plane+ <- Point-`) whose home-plane section bounds the aperture."""
    coordinate_planes = stack((mv.x, mv.y, mv.z))                               # [3] Plane+
    return (coordinate_planes * (coordinate_planes & Point)).sum(-1) - mv.w * (mv.w & Point) * radius**2


def camera_exposure(
    scene: Point,
    subject: Point,
    focal: np.ndarray,
    placements: Motor,
    focus: Point,
    tilt: Motor,
    radius: float,
    rim: Point,
) -> tuple[PointMap, Camera, Motor, Quadric, Plane, Point]:
    """Exposing `scene` (`[...] Point-`) through the two lenses at `placements` (`[2] Rotor+`) with aperture `radius`, focus target `focus`, and sensor `tilt` produces the train collineation, camera map, sensor frame, image cones (`[...] Plane+ <- Point-`), element planes (`[3] Plane+`), and traced ray vertices (`[4, ...] Point-`)."""
    # Place the front and rear lenses, then compose their point collineations and line maps.
    lens_points, lens_lines = thin_lens_maps(focal)                             # [2] Point- <- Point-, Line- <- Line-
    point_maps = placements >> lens_points(placements << Point)                 # [2] Point- <- Point-
    line_maps = placements >> lens_lines(placements << Line)                    # [2] Line- <- Line-
    front, rear = line_maps[0], line_maps[1]                                    # [] Line- <- Line-, Line- <- Line-
    front_plane, rear_plane = placements[0] >> home, placements[1] >> home      # [] Plane+, Plane+
    collineation = point_maps[1](point_maps[0])                                 # [] Point- <- Point-
    train = rear(front)                                                         # [] Line- <- Line-
    pupil_ball = placements[0] >> aperture_ball(radius)(placements[0] << Point)  # [] Plane+ <- Point-

    # The image of the focus target places the sensor centre, and `tilt` turns the sensor plane about that image.
    image = unit_point(collineation(focus))                                     # [] Point-
    frame = (image / origin).square_root() * tilt                               # [] Rotor+
    sensor = frame >> home                                                      # [] Plane+
    camera = train(Point & Point) ^ sensor                                      # [] Point- <- (Point-, Point-)

    # Pull the pupil ball back through central projection from each scene point, then push forward through the lens train.
    projection = (scene & Point) ^ front_plane                                  # [...] Point- <- Point-
    aperture_cones = projection.adjugate()(pupil_ball(projection))              # [...] Plane+ <- Point-
    inverse_collineation = collineation.inverse()                               # [] Point- <- Point-
    image_cones = inverse_collineation.adjugate()(aperture_cones(inverse_collineation))  # [...] Plane+ <- Point-

    rays = subject & (placements[0] >> rim)                                     # [...] Line-
    ray_vertices = stack([
        subject.broadcast_to(rays.shape),
        rays ^ front_plane,
        front(rays) ^ rear_plane,
        rear(front(rays)) ^ sensor,
    ])                                                                          # [4, ...] Point-
    return collineation, camera, frame, image_cones, stack([front_plane, rear_plane, sensor]), ray_vertices


def cone_section(cone: Quadric, start: Point, frame: Motor, n_samples: int) -> Point:
    """Tracing rays outward from interior point `start` along `n_samples` sensor-plane directions samples the conic boundary (`[n_samples] Point-`) where `cone` meets the sensor."""
    angles = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
    turns = (mv.yz * (-angles / 2)).exp()                                       # [n_samples] Rotor+
    sensor_directions = frame >> (turns >> mv.y.dual())                         # [n_samples] Point-
    # Evaluating the quadric form along `start + sensor_directions * step` yields a scalar quadratic in `step`.
    a = cone(sensor_directions) & sensor_directions                             # [n_samples] Scalar+
    b = cone(start) & sensor_directions                                         # [n_samples] Scalar+
    c = cone(start) & start                                                     # [] Scalar+
    # Clamping suppresses tiny negative rounding residuals at a sharp focus vertex where the section collapses to `start`.
    root = (b * b - a * c).clip(0.0, np.inf).square_root()                      # [n_samples] Scalar+
    return start + sensor_directions * ((root - b) / a)                         # [n_samples] Point-
