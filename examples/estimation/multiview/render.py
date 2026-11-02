"""Render top-down views and convergence animations for planar multi-camera reconstruction on the GPU, under a camera of
the plane.

Every sample of the picture is a point of space: a sight cone covers it where its value there, `cone(point) & point`,
the squared offset of the point's image on the sensor scaled by depth, lies below the squared pixel angle times its
depth, and a fused splat where its value lies below its level; the cameras' field-of-view wedges are triangles, and
their sight lines, sensors, axes and pose covariances strokes.
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
from examples.util.animation import capture
from examples.util.wgsl import contours, quadrics, simplices
from examples.util.wgsl.camera import Camera, flat_panel
from examples.util.wgsl.simplices import Cap, Topology

# The figures draw PGA2D geometry, from above.
core = instantiate("examples.estimation.multiview.core", PGA2D)
Motor, Point, Quadric, Scalar = core.Motor, core.Point, core.Quadric, core.Scalar
mv = core.mv
# Every point at unit weight, as a quadric: a dyad of the plane at infinity.
unit_weight = mv.w * (mv.w & Point)                                             # [] Plane+ <- Point-

CAMERA_COLORS = np.array([mcolors.to_rgb(colour) for colour in ("#0284c7", "#ec4899", "#8b5cf6")])
SPLAT_COLOR = np.array([0.98, 0.48, 0.04])
X_RANGE = (-1.25, 1.25)
Y_RANGE = (-0.30, 2.95)
# The pixels along the longer side of what a figure draws.
PIXELS = 300
# The room an animation leaves on every side of its first state's cameras and landmarks, a share of the longer side of
# what they span: the picture's margin, and room for the cameras and landmarks to move as they align.
ROOM = 0.25
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


def canvas(x_range: tuple[float, float], y_range: tuple[float, float] = Y_RANGE) -> tuple[plt.Figure, plt.Axes, Camera]:
    """A figure `PIXELS` pixels along the longer of `x_range` and `y_range` and as many square pixels along the other,
    its axes on them, and the camera its pictures are read by."""
    spans = np.array([x_range[1] - x_range[0], y_range[1] - y_range[0]])
    n_columns, n_rows = np.round(PIXELS * spans / spans.max()).astype(int)
    figure = plt.figure(figsize=(n_columns / 100, n_rows / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set(xlim=x_range, ylim=y_range)
    return figure, ax, flat_panel(ax, PGA2D)


def polylines(points: Point, width: float, colour: np.ndarray, opacity: float, camera: Camera,
              dashes: tuple[float, float] = (1.0, 0.0), cap: Cap = Cap.PROJECTING) -> np.ndarray:
    """Polylines through points `[n, k]`, `width` pixels across."""
    size, n_points = points.shape
    return simplices.strokes(points.reshape(-1), Topology.chains([n_points] * size), colour, width, camera, dashes=dashes,
                             cap=cap, opacity=opacity)


def cones(motors: Motor, world_cones: Quadric, camera: Camera) -> list[np.ndarray]:
    """Each camera's sight cones in its colour, where a cone's value lies below the squared `PIXEL_ANGLE` times the
    squared depth from the focal plane, in front of the focal plane: the sight lines on the side the camera sees."""
    focal_planes = motors >> mv.y                                                # [cameras] Plane+
    # The squared reach of a pixel at each depth, as a quadric: a dyad of the focal plane.
    reaches = (focal_planes * PIXEL_ANGLE) * ((focal_planes * PIXEL_ANGLE) & Point)   # [cameras] Plane+ <- Point-
    # In front of each focal plane: where it pairs positively with the point, the half-plane's form negative there.
    fronts = -(focal_planes * (mv.w & Point) + mv.w * (focal_planes & Point)) / 2     # [cameras] Plane+ <- Point-
    sights = world_cones - reaches                                               # [landmarks, cameras] Plane+ <- Point-
    return [quadrics.regions(stack((sights[:, index], fronts[index].broadcast_to(sights.shape[:1])), axis=-1),
                             colour, camera, 0.30)
            for index, colour in enumerate(CAMERA_COLORS[:motors.shape[0]])]


def splats(motors: Motor, fused_splats: Quadric, points: Point, camera: Camera) -> list[np.ndarray]:
    """The fused splat quadrics around the triangulated points, each where its value lies below its value at its point
    by the squared pixel angle times the squared depth of the point, and the value at its point once more."""
    floor = (fused_splats(points) & points).clip(0, np.inf)                    # [landmarks] Scalar+
    depth = ((motors >> mv.y)[None, :] & points[:, None]).mean(axis=1)         # [landmarks] Scalar+
    levels = 2 * floor + (PIXEL_ANGLE * depth.clip(0.2, np.inf)).squared()     # [landmarks] Scalar+
    return [quadrics.regions((fused_splats - unit_weight * levels)[:, None], SPLAT_COLOR, camera, 0.95)]


def camera_marks(motors: Motor) -> Point:
    """Each camera's pinhole, its sensor's ends, where the sight lines turned the half field either way from the axis
    meet the sensor in front of the pinhole, and its axis's end past the sensor, `[cameras, marks] Point`."""
    pinhole, axis = mv.point([0, 0]), mv.direction([0, 1])                       # [] Point-, Direction-
    sensor = mv.y - WEDGE_DEPTH * mv.w                                           # [] Plane+
    half_fields = np.array([-HALF_FIELD, HALF_FIELD])
    sight_turns = (mv.xy * (-half_fields / 2)).exp()                             # [ends] Motor+
    sensor_ends = (sensor ^ (pinhole & (sight_turns >> axis))).normalized()      # [ends] Point-
    axis_end = pinhole + axis * (WEDGE_DEPTH * AXIS_REACH)                       # [] Point-
    local_marks = concatenate((pinhole[None], sensor_ends, axis_end[None]))      # [marks] Point-
    return motors[:, None] >> local_marks                                        # [cameras, marks] Point-


def cameras(motors: Motor, camera: Camera) -> list[np.ndarray]:
    """The field-of-view wedge, bounding sight lines, sensor, and dashed optical axis for each camera."""
    marks = camera_marks(motors)                                                 # [cameras, marks] Point-
    camera_colours = CAMERA_COLORS[:motors.shape[0]]
    wedge = Topology(3, [[0, 1, 2]])
    wedges = [[contours.picture(marks[index, :3], wedge, np.zeros(3), colour, camera, opacity=0.18),
               simplices.strokes(marks[index, :3], Topology.loops([3]), colour, EDGE_WIDTH, camera, opacity=0.18),
               polylines(marks[index][np.array([[0, 1], [0, 2]])], RAY_WIDTH, colour, 0.6, camera)]
              for index, colour in enumerate(camera_colours)]
    sensors = [[polylines(marks[index:index + 1, [1, 2]], SENSOR_WIDTH, colour, 1.0, camera),
                polylines(marks[index:index + 1, [0, 3]], AXIS_WIDTH, colour, 1.0, camera, AXIS_WIDTH * DASHED,
                          Cap.BUTT)]
               for index, colour in enumerate(camera_colours)]
    return [picture for pictures in wedges + sensors for picture in pictures]


def pose_covariances(motors: Motor, ellipsoids: Quadric, spreads: Scalar, camera: Camera) -> list[np.ndarray]:
    """Each camera's position uncertainty ellipsoid and dotted heading-spread fan."""
    # The outline: from each ellipsoid's centre, the pole of the plane at infinity, out along each direction to where the
    # ellipsoid vanishes.
    centres = ellipsoids.solve(mv.w)                                             # [cameras] Point-
    # Divided by its weight rather than normalized, which keeps the weight's sign: the outline runs out from a centre
    # of unit weight.
    centres = centres / (mv.w & centres)                                         # [cameras] Point-
    turns = (mv.xy * (-np.linspace(0.0, 2 * np.pi, N_CURVE_SEGMENTS, endpoint=False) / 2)).exp()   # [segments] Motor+
    headings = (turns >> mv.direction([1, 0]))[:, None]                         # [segments, cameras] Direction-
    reach = (-(ellipsoids(centres) & centres) / (ellipsoids(headings) & headings)).square_root()   # [segments, cameras] Scalar+
    outlines = centres + headings * reach                                        # [segments, cameras] Point-
    pictures = []
    for index, colour in enumerate(CAMERA_COLORS[:motors.shape[0]]):
        # The fan in the camera's frame, about its pinhole from its axis, carried out by its pose.
        fan_angles = spreads[index] * np.linspace(-1, 1, N_CURVE_SEGMENTS + 1)     # [segments + 1] Scalar+
        fan_turns = (mv.xy * (-fan_angles / 2)).exp()                                  # [segments + 1] Motor+
        fan = motors[index] >> (fan_turns >> mv.point([0, ARC_RADIUS]))          # [segments + 1] Point-
        pinhole = motors[index] >> mv.point([0, 0])                              # [] Point-
        spokes = stack((pinhole.broadcast_to((2,)), fan[[0, -1]]), axis=1)       # [spokes, ends] Point-
        pictures += [quadrics.regions(ellipsoids[index:index + 1, None], colour, camera, 0.28),
                     simplices.strokes(outlines[:, index], Topology.loops([N_CURVE_SEGMENTS]), colour, OUTLINE_WIDTH,
                                       camera, dashes=OUTLINE_WIDTH * DASHED, cap=Cap.BUTT, opacity=0.28),
                     polylines(fan[None], ARC_WIDTH, colour, 1.0, camera, ARC_WIDTH * DOTTED, Cap.BUTT),
                     polylines(spokes, SPOKE_WIDTH, colour, 1.0, camera, SPOKE_WIDTH * DOTTED, Cap.BUTT)]
    return pictures


def draw_rig(motors: Motor) -> plt.Figure:
    """Draw the camera rig."""
    figure, ax, camera = canvas(X_RANGE)
    simplices.overlay(ax, cameras(motors, camera))
    return figure


def draw_cones(motors: Motor, world_cones: Quadric) -> plt.Figure:
    """Draw the cameras and their sight cones through the scene points."""
    figure, ax, camera = canvas(X_RANGE)
    simplices.overlay(ax, cones(motors, world_cones, camera) + cameras(motors, camera))
    return figure


def draw_reconstruction(motors: Motor, world_cones: Quadric, fused_splats: Quadric, points: Point) -> plt.Figure:
    """Draw the sight cones fused into splats around the reconstructed landmarks."""
    figure, ax, camera = canvas(X_RANGE)
    simplices.overlay(ax, cones(motors, world_cones, camera) + splats(motors, fused_splats, points, camera)
                      + cameras(motors, camera))
    return figure


def draw_reconstruction_with_covariance(
    motors: Motor, world_cones: Quadric, fused_splats: Quadric, points: Point, ellipsoids: Quadric, spreads: Scalar,
) -> plt.Figure:
    """Draw the reconstruction with each camera's position uncertainty ellipsoid and heading spread."""
    figure, ax, camera = canvas(X_RANGE)
    simplices.overlay(ax, cones(motors, world_cones, camera) + splats(motors, fused_splats, points, camera)
                      + cameras(motors, camera) + pose_covariances(motors, ellipsoids, spreads, camera))
    return figure


def animate_convergence(
    states: Iterable[tuple[Motor, Quadric, Quadric, Point]], room: float = ROOM,
) -> Iterator[np.ndarray]:
    """Yield RGB frames of the reconstruction, framed on the first state's cameras and triangulated landmarks with `room`
    around them, a share of the longer side of what they span."""
    states = iter(states)
    first = next(states)
    first_motors, _, _, first_points = first
    drawn = coordinates(concatenate((camera_marks(first_motors).reshape(-1), first_points)))   # [marks + landmarks, 2]
    low, high = drawn.min(axis=0), drawn.max(axis=0)
    margin = room * (high - low).max()
    figure, ax, camera = canvas((low[0] - margin, high[0] + margin), (low[1] - margin, high[1] + margin))
    image = simplices.overlay(ax, [np.zeros(camera.shape + (4,))])
    for motors, world_cones, fused_splats, points in chain([first], states):
        image.set_data(simplices.composited(cones(motors, world_cones, camera)
                                            + splats(motors, fused_splats, points, camera) + cameras(motors, camera)))
        yield capture(figure)
    plt.close(figure)
