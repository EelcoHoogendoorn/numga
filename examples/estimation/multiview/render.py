"""Render top-down views and convergence animations for planar multi-camera reconstruction on the GPU.

Every pixel sample is a point of space: a sight cone or a fused splat covers it where its quadric's value
there, `quadric(point) & point`, lies below the level it is drawn at, while the cameras' field-of-view wedges, sight lines, sensors, axes,
and pose covariances cover it through the pairings of their lines with the sample point.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.algebras import PGA2D

from examples import instantiate
from examples.util.wgsl import image_figure
from examples.util.wgsl.planar import SOLID, Cap, Quadrics, Polygons, Regions, Strokes, draw, window

# The figures draw PGA2D geometry, from above.
core = instantiate("examples.estimation.multiview.core", PGA2D)
Motor, Point, Quadric, Scalar = core.Motor, core.Point, core.Quadric, core.Scalar
mv = core.mv

CAMERA_COLORS = np.array([mcolors.to_rgb(colour) for colour in ("#0284c7", "#ec4899", "#8b5cf6")])
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
DASHED, DOTTED = np.array([3.7, 1.6]), np.array([1.0, 1.65])
N_CURVE_SEGMENTS = 96


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """The Cartesian (x, y) coordinates of points from their pairings with the coordinate planes."""
    # Divided by the weight rather than normalized, which keeps its sign: a point of negative weight reads true too.
    return (stack((mv.x, mv.y)) & (points / (mv.w & points))[..., None]).to_array()


def cones(motors: Motor, world_cones: Quadric) -> list[Quadrics]:
    """Each camera's sight-cone layer in its colour, widened with depth in front of the focal plane by `PIXEL_ANGLE`."""
    focal_planes = motors >> mv.y                                                # [cameras] Plane+
    return [Quadrics(world_cones[:, camera], np.zeros(1), focal_planes[camera] * PIXEL_ANGLE, np.full(1, PIXEL_ANGLE * 0.05),
                   (focal_planes[camera] - mv.w * 0.02)[None], colour, 0.30)
            for camera, colour in enumerate(CAMERA_COLORS[:motors.shape[0]])]


def splats(motors: Motor, fused_splats: Quadric, points: Point) -> list[Quadrics]:
    """The layer of fused splat quadrics around the triangulated points where they lie in front of a camera."""
    floor = np.maximum((fused_splats(points) & points).to_array(), 0.0)          # [landmarks]
    depth = ((motors >> mv.y)[None, :] & points[:, None]).to_array().mean(axis=1)  # [landmarks]
    radius = np.sqrt((PIXEL_ANGLE * np.maximum(depth, 0.2)) ** 2 + floor)        # [landmarks]
    front_planes = ((motors >> mv.y) - mv.w * 0.02)[None]                        # [clips, cameras] Plane+
    return [Quadrics(fused_splats, floor, mv.w * radius, np.zeros(1), front_planes, SPLAT_COLOR, 0.95)]


def cameras(motors: Motor) -> list:
    """The field-of-view wedge, bounding sight lines, sensor, and dashed optical axis for each camera."""
    # In the camera's frame: the pinhole, the sensor's ends, where the sight lines turned the half field either way
    # from the axis meet the sensor in front of the pinhole, and the axis's end past the sensor.
    pinhole, axis = mv.point([0, 0]), mv.direction([0, 1])                       # [] Point-, Direction-
    sensor = mv.y - WEDGE_DEPTH * mv.w                                           # [] Plane+
    half_fields = np.array([-HALF_FIELD, HALF_FIELD])
    sight_turns = (mv.xy * (-half_fields / 2)).exp()                             # [ends] Motor+
    sensor_ends = (sensor ^ (pinhole & (sight_turns >> axis))).normalized()      # [ends] Point-
    axis_end = pinhole + axis * (WEDGE_DEPTH * AXIS_REACH)                       # [] Point-
    local_marks = concatenate((pinhole[None], sensor_ends, axis_end[None]))      # [marks] Point-
    marks = motors[:, None] >> local_marks                                       # [cameras, marks] Point-
    camera_colours = CAMERA_COLORS[:motors.shape[0]]
    wedges = [[Polygons(marks[camera:camera + 1, :3], colour, 0.18),
               Strokes(marks[camera:camera + 1, [0, 1, 2, 0]], EDGE_WIDTH, colour, 0.18, SOLID, Cap.PROJECTING),
               Strokes(marks[camera][np.array([[0, 1], [0, 2]])], RAY_WIDTH, colour, 0.6, SOLID, Cap.PROJECTING)]
              for camera, colour in enumerate(camera_colours)]
    sensors = [[Strokes(marks[camera:camera + 1, [1, 2]], SENSOR_WIDTH, colour, 1.0, SOLID, Cap.PROJECTING),
                Strokes(marks[camera:camera + 1, [0, 3]], AXIS_WIDTH, colour, 1.0, AXIS_WIDTH * DASHED, Cap.BUTT)]
               for camera, colour in enumerate(camera_colours)]
    return [layer for camera in wedges + sensors for layer in camera]


def pose_covariances(motors: Motor, ellipsoids: Quadric, spreads: Scalar) -> list:
    """Each camera's position uncertainty ellipsoid and dotted heading-spread fan."""
    # The outline: from each ellipsoid's centre, the pole of the plane at infinity, out along each direction to where the
    # ellipsoid vanishes.
    centres = ellipsoids.solve(mv.w)                                             # [cameras] Point-
    # Divided by its weight rather than normalized, which keeps the weight's sign: the outline runs out from a centre
    # of unit weight.
    centres = centres / (mv.w & centres)                                         # [cameras] Point-
    turns = (mv.xy * (-np.linspace(0.0, 2 * np.pi, N_CURVE_SEGMENTS + 1) / 2)).exp()   # [segments + 1] Motor+
    headings = (turns >> mv.direction([1, 0]))[:, None]                         # [segments + 1, cameras] Direction-
    reach = (-(ellipsoids(centres) & centres) / (ellipsoids(headings) & headings)).square_root()   # [segments + 1, cameras] Scalar+
    outlines = centres + headings * reach                                        # [segments + 1, cameras] Point-
    layers = []
    for camera, colour in enumerate(CAMERA_COLORS[:motors.shape[0]]):
        # The fan in the camera's frame, about its pinhole from its axis, carried out by its pose.
        fan_angles = spreads[camera] * np.linspace(-1, 1, N_CURVE_SEGMENTS + 1)    # [segments + 1] Scalar+
        fan_turns = (mv.xy * (-fan_angles / 2)).exp()                            # [segments + 1] Motor+
        fan = motors[camera] >> (fan_turns >> mv.point([0, ARC_RADIUS]))         # [segments + 1] Point-
        pinhole = motors[camera] >> mv.point([0, 0])                             # [] Point-
        spokes = stack((pinhole.broadcast_to((2,)), fan[[0, -1]]), axis=1)       # [spokes, ends] Point-
        layers += [Regions(ellipsoids[camera:camera + 1], colour, 0.28),
                   Strokes(outlines[:, camera][None], OUTLINE_WIDTH, colour, 0.28, OUTLINE_WIDTH * DASHED, Cap.BUTT),
                   Strokes(fan[None], ARC_WIDTH, colour, 1.0, ARC_WIDTH * DOTTED, Cap.BUTT),
                   Strokes(spokes, SPOKE_WIDTH, colour, 1.0, SPOKE_WIDTH * DOTTED, Cap.BUTT)]
    return layers


def draw_rig(motors: Motor) -> plt.Figure:
    """Draw the camera rig."""
    return image_figure(draw(window(X_RANGE, Y_RANGE, PIXELS), cameras(motors), np.ones(3)))


def draw_cones(motors: Motor, world_cones: Quadric) -> plt.Figure:
    """Draw the cameras and their sight cones through the scene points."""
    return image_figure(draw(window(X_RANGE, Y_RANGE, PIXELS), cones(motors, world_cones) + cameras(motors), np.ones(3)))


def draw_reconstruction(motors: Motor, world_cones: Quadric, fused_splats: Quadric, points: Point) -> plt.Figure:
    """Draw the sight cones fused into splats around the reconstructed landmarks."""
    layers = cones(motors, world_cones) + splats(motors, fused_splats, points) + cameras(motors)
    return image_figure(draw(window(X_RANGE, Y_RANGE, PIXELS), layers, np.ones(3)))


def draw_reconstruction_with_covariance(
    motors: Motor, world_cones: Quadric, fused_splats: Quadric, points: Point, ellipsoids: Quadric, spreads: Scalar,
) -> plt.Figure:
    """Draw the reconstruction with each camera's position uncertainty ellipsoid and heading spread."""
    layers = cones(motors, world_cones) + splats(motors, fused_splats, points) + cameras(motors) + pose_covariances(motors, ellipsoids, spreads)
    return image_figure(draw(window(X_RANGE, Y_RANGE, PIXELS), layers, np.ones(3)))


# README picture: this also draws plots/multiview_convergence.gif, tuned as a visual for the README; a change here changes
# that picture, so check it after any change, and to experiment, copy this instead.
def animate_convergence(
    states: Iterable[tuple[Motor, Quadric, Quadric, Point]], room: float = CAMERA_ROOM,
) -> Iterator[np.ndarray]:
    """Yield RGB frames of the reconstruction across the scene, widened to hold the first state's cameras with `room` around them."""
    states = iter(states)
    first = next(states)
    centres = coordinates(first[0] >> mv.point([0, 0]))[:, 0]                    # [cameras]
    x_range = (min(X_RANGE[0], centres.min() - room), max(X_RANGE[1], centres.max() + room))
    for motors, world_cones, fused_splats, points in chain([first], states):
        layers = cones(motors, world_cones) + splats(motors, fused_splats, points) + cameras(motors)
        yield draw(window(x_range, Y_RANGE, PIXELS), layers, np.ones(3))
