"""Top-down drawing and convergence animation for multi-camera reconstruction in PGA2D, drawn on the GPU by `planar`.

Every sample of a pixel is a point of the plane: a sight cone or a fused splat covers it where its quadric's value
there, `quadric(p) & p`, lies under the level it is drawn at, and the cameras' wedges, sight lines, sensors and axes,
and the poses' covariances, cover it by the pairings of their lines with it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga.algebras import PGA2D

from examples import instantiate
from examples.util.wgsl import image_figure
from examples.util.wgsl.planar import SOLID, Cap, Conics, Polygons, Regions, Strokes, draw, window

# The figures are top-down views of the plane: they draw PGA2D geometry.
core = instantiate("examples.estimation.multiview.core", PGA2D)
Motor, Point, Quadric, Scalar = core.Motor, core.Point, core.Quadric, core.Scalar
mv = core.mv

CAMERA_COLORS = ["#0284c7", "#ec4899", "#8b5cf6"]
SPLAT_COLOR = np.array([0.98, 0.48, 0.04])
X_RANGE = (-1.25, 1.25)
Y_RANGE = (-0.30, 2.95)
# README picture: the rows of pixels down the scene set plots/multiview_convergence.gif, about 300 pixels a side, tuned as
# a visual for the README; to experiment, copy them.
PIXELS = 300
# The room an animation leaves around its first state's cameras, in metres, where they lie outside the scene drawn.
CAMERA_ROOM = 0.35
# Angular half-width of a pixel, the cone radius per unit depth.
PIXEL_ANGLE = 0.035         # rad
# The cameras' wedges, how far in front of the pinhole their sensors lie and how wide they open, and how far their axes
# reach past the sensor.
WEDGE_DEPTH, HALF_FIELD, AXIS_REACH = 0.28, np.radians(38.0), 1.15
# Pixels a point, a stroke's width at 60 dots per inch. The widths of the strokes: the edge of a wedge, the sight lines
# bounding it, the sensor, the dashed axis; a covariance's dashed outline, its dotted arc of headings and the dotted
# lines out to the arc's ends; and the arc's radius.
POINT = 60 / 72
EDGE_WIDTH, RAY_WIDTH, SENSOR_WIDTH, AXIS_WIDTH = 1.0 * POINT, 1.1 * POINT, 2.0 * POINT, 1.2 * POINT
OUTLINE_WIDTH, ARC_WIDTH, SPOKE_WIDTH = 1.6 * POINT, 1.3 * POINT, 1.1 * POINT
ARC_RADIUS = 0.38
# Dashes and dots, on and off, in widths of their stroke; and the segments a curve is drawn with.
DASHED, DOTTED = (3.7, 1.6), (1.0, 1.65)
N_CURVE_SEGMENTS = 96


# --- read-out -----------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """Cartesian (x, y) of points, read in an explicit blade layout."""
    k = points.cast(core.ga.subspace("yw wx xy")).kernel
    return k[..., :2] / k[..., 2:]


# --- layers -------------------------------------------------------------------------------
def colours(motors: Motor) -> np.ndarray:
    """The cameras' colours `[cameras, 3]`."""
    return np.array([mcolors.to_rgb(colour) for colour in CAMERA_COLORS[:motors.shape[0]]])


def cones(motors: Motor, world_cones: Quadric) -> list[Conics]:
    """Each camera's sight cones in its colour, as wide as a pixel at each depth: about the depth in front of the
    camera's focal line times the pixel's angle, at least as wide as at 5 cm, and only from 2 cm in front of that line."""
    focal = motors >> mv.y                                                       # [cameras] Plane
    return [Conics(world_cones[:, camera], np.zeros(1), focal[camera] * PIXEL_ANGLE, np.full(1, PIXEL_ANGLE * 0.05),
                   (focal[camera] - mv.w * 0.02)[None], colour, 0.30)
            for camera, colour in enumerate(colours(motors))]


def splats(motors: Motor, fused: Quadric, points: Point) -> list[Conics]:
    """The fused quadrics around their points, less their value there, about the pixel's width at the points' depth
    and that value, wherever some camera has the point in front of it."""
    floor = np.maximum((fused(points) & points).to_array(), 0.0)                  # [points]
    depth = ((motors >> mv.y)[None, :] & points[:, None]).to_array().mean(axis=1)  # [points]
    radius = np.sqrt((PIXEL_ANGLE * np.maximum(depth, 0.2)) ** 2 + floor)          # [points]
    in_front = ((motors >> mv.y) - mv.w * 0.02)[None]                              # [1, cameras] Plane
    return [Conics(fused, floor, mv.w * radius, np.zeros(1), in_front, SPLAT_COLOR, 0.95)]


def cameras(motors: Motor) -> list:
    """Each camera's field-of-view wedge, with its edge, and the sight lines bounding it, then each camera's sensor line
    and dashed optical axis: the pinhole, the ends of the sensor and the axis's tip, carried from the camera's frame by
    its pose. Solid lines reach half their width past their ends; dashes end square."""
    reach = WEDGE_DEPTH * np.tan(HALF_FIELD)
    local = mv.point([[0.0, 0.0], [reach, WEDGE_DEPTH], [-reach, WEDGE_DEPTH], [0.0, WEDGE_DEPTH * AXIS_REACH]])
    marks = motors[:, None] >> local                                             # [cameras, 4] Point
    wedges = [[Polygons(marks[camera:camera + 1, :3], colour, 0.18),
               Strokes(marks[camera:camera + 1, [0, 1, 2, 0]], EDGE_WIDTH, colour, 0.18, SOLID, Cap.PROJECTING),
               Strokes(marks[camera][np.array([[0, 1], [0, 2]])], RAY_WIDTH, colour, 0.6, SOLID, Cap.PROJECTING)]
              for camera, colour in enumerate(colours(motors))]
    sensors = [[Strokes(marks[camera:camera + 1, [1, 2]], SENSOR_WIDTH, colour, 1.0, SOLID, Cap.PROJECTING),
                Strokes(marks[camera:camera + 1, [0, 3]], AXIS_WIDTH, colour, 1.0, dashes(AXIS_WIDTH, DASHED), Cap.BUTT)]
               for camera, colour in enumerate(colours(motors))]
    return [layer for camera in wedges + sensors for layer in camera]


def dashes(width: float, pattern: tuple[float, float]) -> tuple[float, float]:
    """A pattern of dashes, in widths of their stroke, in pixels."""
    return width * pattern[0], width * pattern[1]


def pose_covariances(motors: Motor, ellipses: Quadric, spreads: Scalar) -> list:
    """Each camera's position uncertainty, the region where its ellipse `[cameras]` is negative, with a dashed outline;
    and its heading's spread `[cameras]`, in radians either way of its optical axis, as a dotted arc and the dotted lines
    out to its ends."""
    # The outline: from each ellipse's centre, the pole of the line at infinity, out along each direction to where the
    # ellipse vanishes.
    centres = ellipses.solve(mv.w)                                              # [cameras] Point
    centres = centres / (mv.w & centres)
    turns = np.linspace(0.0, 2 * np.pi, N_CURVE_SEGMENTS + 1)
    headings = mv.direction(np.stack([np.cos(turns), np.sin(turns)], axis=-1))[:, None]   # [segments + 1, 1] Point
    reach = (-(ellipses(centres) & centres) / (ellipses(headings) & headings)).square_root()   # [segments + 1, cameras] Scalar
    outlines = centres + headings * reach                                       # [segments + 1, cameras] Point
    layers = []
    for camera, colour in enumerate(colours(motors)):
        # The fan in the camera's frame, about its pinhole from its axis, carried out by its pose.
        spread = spreads[camera].to_array()
        turns = np.linspace(-spread, spread, N_CURVE_SEGMENTS + 1)
        around = ARC_RADIUS * np.stack([-np.sin(turns), np.cos(turns)], axis=-1)
        fan = motors[camera] >> mv.point(around)                                # [segments + 1] Point
        spokes = motors[camera] >> mv.point(np.stack([np.zeros((2, 2)), around[[0, -1]]], axis=1))   # [2, 2] Point
        layers += [Regions(ellipses[camera:camera + 1], colour, 0.28),
                   Strokes(outlines[:, camera][None], OUTLINE_WIDTH, colour, 0.28, dashes(OUTLINE_WIDTH, DASHED), Cap.BUTT),
                   Strokes(fan[None], ARC_WIDTH, colour, 1.0, dashes(ARC_WIDTH, DOTTED), Cap.BUTT),
                   Strokes(spokes, SPOKE_WIDTH, colour, 1.0, dashes(SPOKE_WIDTH, DOTTED), Cap.BUTT)]
    return layers


def picture(layers: list, x_range: tuple[float, float] = X_RANGE) -> np.ndarray:
    """The layers over white, across `x_range` and `Y_RANGE`, PIXELS rows down: RGB bytes."""
    return draw(window(x_range, Y_RANGE, PIXELS), layers, np.ones(3))


# --- figures ------------------------------------------------------------------------------
def draw_rig(motors: Motor) -> plt.Figure:
    """The cameras."""
    return image_figure(picture(cameras(motors)))


def draw_cones(motors: Motor, world_cones: Quadric) -> plt.Figure:
    """The cameras and their sight cones through the scene points."""
    return image_figure(picture(cones(motors, world_cones) + cameras(motors)))


def draw_reconstruction(motors: Motor, world_cones: Quadric, fused: Quadric, points: Point) -> plt.Figure:
    """Sight cones fused into splats around the reconstructed points."""
    return image_figure(picture(cones(motors, world_cones) + splats(motors, fused, points) + cameras(motors)))


def draw_reconstruction_with_covariance(
    motors: Motor, world_cones: Quadric, fused: Quadric, points: Point, ellipses: Quadric, spreads: Scalar,
) -> plt.Figure:
    """The reconstruction with each camera's pose uncertainty: the ellipse of its position and the spread of its
    heading."""
    return image_figure(picture(cones(motors, world_cones) + splats(motors, fused, points) + cameras(motors)
                                + pose_covariances(motors, ellipses, spreads)))


# README picture: this also draws plots/multiview_convergence.gif, tuned as a visual for the README; a change here changes
# that picture, so check it after any change, and to experiment, copy this instead.
def animate_convergence(states: Iterable, room: float = CAMERA_ROOM) -> Iterator[np.ndarray]:
    """RGB frames of the reconstruction, one per state, across the scene drawn, widened to hold the first state's
    cameras with `room` around them where they lie outside it."""
    states = iter(states)
    first = next(states)
    centres = coordinates(first[0] >> mv.point(np.zeros(2)))[:, 0]               # [cameras]
    x_range = (min(X_RANGE[0], centres.min() - room), max(X_RANGE[1], centres.max() + room))
    for motors, world_cones, fused, points in chain([first], states):
        yield picture(cones(motors, world_cones) + splats(motors, fused, points) + cameras(motors), x_range)
