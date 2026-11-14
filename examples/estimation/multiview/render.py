"""Top-down planar views and convergence animations draw multi-camera sight cones, fused landmark splats, camera wedges, and pose uncertainty regions on the GPU.

Each pixel of a figure samples a point in the plane. Implicit quadric regions shade each sight cone where `cone(point) & point` stays below the squared pixel angle scaled by squared depth in front of the focal plane, and shade each fused splat around its reconstructed landmark, while triangulated wedges and polyline strokes draw the cameras, optical axes, and pose covariance fans.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.algebras import PGA2D
from numga.render import Camera, Cap, contours, drawing, quadrics, simplices
from numga.topology import Topology

from examples import instantiate
from examples.util.animation import capture
from examples.util.panels import overlay, panel

# Planar figures instantiate the multiview algebra in 2D and view the rig from above.
core = instantiate("examples.estimation.multiview.core", PGA2D)
Motor, Point, Quadric, Scalar = core.Motor, core.Point, core.Quadric, core.Scalar
mv = core.mv
# Dyad on the plane at infinity evaluating to one on every unit-weight point.
unit_weight = mv.w * (mv.w & Point)                                             # [] Plane+ <- Point-

camera_colours = np.array([mcolors.to_rgb(colour) for colour in ("#0284c7", "#ec4899", "#8b5cf6")])
splat_colour = np.array([0.98, 0.48, 0.04])
scene_x_range = (-1.25, 1.25)
scene_y_range = (-0.30, 2.95)
n_pixels = 300
# Fractional margin around the initial cameras and landmarks in convergence animations.
frame_margin = 0.25
# Angular half-width of one pixel in radians, setting the drawn sight-cone opening per unit depth.
pixel_angle = 0.035
wedge_depth, half_field, axis_reach = 0.28, np.radians(38.0), 1.15
point_px = 60 / 72
edge_width, ray_width, sensor_width, axis_width = 1.0 * point_px, 1.1 * point_px, 2.0 * point_px, 1.2 * point_px
outline_width, arc_width, spoke_width = 1.6 * point_px, 1.3 * point_px, 1.1 * point_px
arc_radius = 0.38
dash_pattern, dot_pattern = np.array([3.7, 1.6]), np.array([1.0, 1.65])
n_curve_segments = 96


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """Cartesian `(x, y)` coordinates (`[..., 2]`) come from pairing unit-weight `points` (`[...] Point-`) with the coordinate planes."""
    # Dividing by point weight rather than Euclidean norm preserves sign when reading signed projective points.
    return (stack((mv.x, mv.y)) & (points / (mv.w & points))[..., None]).to_array()


def scene_canvas(
    x_range: tuple[float, float], y_range: tuple[float, float] = scene_y_range,
) -> tuple[plt.Figure, plt.Axes, Camera]:
    """A borderless figure spanning `n_pixels` along the longer side of `x_range` and `y_range` pairs its axes with a planar camera."""
    spans = np.array([x_range[1] - x_range[0], y_range[1] - y_range[0]])
    n_columns, n_rows = np.round(n_pixels * spans / spans.max()).astype(int)
    figure = plt.figure(figsize=(n_columns / 100, n_rows / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set(xlim=x_range, ylim=y_range)
    return figure, ax, panel(ax, mv.x, mv.y, mv.w)


def polylines(
    points: Point, width: float, colour: np.ndarray, opacity: float, camera: Camera,
    dashes: tuple[float, float] = (1.0, 0.0), cap: Cap = Cap.PROJECTING,
) -> np.ndarray:
    """Open polyline strokes of pixel `width` connect each chain of points in `points` (`[n_chains, n_points] Point-`)."""
    n_chains, n_points = points.shape
    return simplices.strokes(
        points.reshape(-1), Topology.chains([n_points] * n_chains), colour, width, camera,
        dashes=dashes, cap=cap, opacity=opacity,
    )


def cone_layers(motors: Motor, world_cones: Quadric, camera: Camera) -> list[np.ndarray]:
    """Translucent quadric regions shade each camera's sight cones in front of its focal plane where the cone value stays within `pixel_angle` per unit depth."""
    n_cameras = motors.shape[0]
    focal_planes = motors >> mv.y                                                # [n_cameras] Plane+
    # A dyad on each focal plane measures squared pixel reach as a function of depth.
    reaches = (focal_planes * pixel_angle) * ((focal_planes * pixel_angle) & Point)  # [n_cameras] Plane+ <- Point-
    # Half-plane form negative in front of each focal plane, where the plane pairs positively with a point.
    fronts = -(focal_planes * (mv.w & Point) + mv.w * (focal_planes & Point)) / 2    # [n_cameras] Plane+ <- Point-
    sights = world_cones - reaches                                               # [..., n_cameras] Plane+ <- Point-
    return [quadrics.regions(stack((sights[:, index], fronts[index].broadcast_to(sights.shape[:1])), axis=-1),
                             colour, camera, 0.30)
            for index, colour in enumerate(camera_colours[:n_cameras])]


def splat_layers(motors: Motor, fused_splats: Quadric, points: Point, camera: Camera) -> list[np.ndarray]:
    """Quadric regions draw the fused landmark splats around `points` (`[...] Point-`) at a level set scaled by mean depth and residual cost."""
    residual_floor = (fused_splats(points) & points).clip(0, np.inf)             # [...] Scalar+
    mean_depth = ((motors >> mv.y)[None, :] & points[:, None]).mean(axis=1)      # [...] Scalar+
    levels = 2 * residual_floor + (pixel_angle * mean_depth.clip(0.2, np.inf)).squared()  # [...] Scalar+
    return [quadrics.regions((fused_splats - unit_weight * levels)[:, None], splat_colour, camera, 0.95)]


def camera_marks(motors: Motor) -> Point:
    """Pinhole, left and right sensor ends, and optical-axis tip in the world (`[..., 4] Point-`) for each camera pose in `motors`."""
    pinhole, axis = mv.point([0, 0]), mv.direction([0, 1])                       # [] Point-, Direction-
    sensor = mv.y - wedge_depth * mv.w                                           # [] Plane+
    half_fields = np.array([-half_field, half_field])
    sight_turns = (mv.xy * (-half_fields / 2)).exp()                             # [2] Motor+
    turned_axes = sight_turns >> axis                                            # [2] Direction-
    sensor_ends = (sensor ^ (pinhole & turned_axes)).normalized()                # [2] Point-
    axis_end = pinhole + axis * (wedge_depth * axis_reach)                       # [] Point-
    local_marks = concatenate((pinhole[None], sensor_ends, axis_end[None]))      # [4] Point-
    return motors[:, None] >> local_marks                                        # [..., 4] Point-


def camera_layers(motors: Motor, camera: Camera) -> list[np.ndarray]:
    """Field-of-view wedge, bounding sight lines, sensor bar, and dashed optical axis for each camera in `motors`."""
    n_cameras = motors.shape[0]
    marks = camera_marks(motors)                                                 # [n_cameras, 4] Point-
    colours = camera_colours[:n_cameras]
    wedge = Topology(3, [[0, 1, 2]])
    wedges = [[contours.picture(marks[index, :3], wedge, np.zeros(3), colour, camera, opacity=0.18),
               simplices.strokes(marks[index, :3], Topology.loops([3]), colour, edge_width, camera, opacity=0.18),
               polylines(marks[index][np.array([[0, 1], [0, 2]])], ray_width, colour, 0.6, camera)]
              for index, colour in enumerate(colours)]
    sensors = [[polylines(marks[index:index + 1, [1, 2]], sensor_width, colour, 1.0, camera),
                polylines(marks[index:index + 1, [0, 3]], axis_width, colour, 1.0, camera, axis_width * dash_pattern,
                          Cap.BUTT)]
               for index, colour in enumerate(colours)]
    return [picture for pictures in wedges + sensors for picture in pictures]


def covariance_layers(motors: Motor, ellipsoids: Quadric, spreads: Scalar, camera: Camera) -> list[np.ndarray]:
    """Shaded position uncertainty ellipsoids, dashed ellipsoid outlines, and dotted heading-spread fans for each camera."""
    n_cameras = motors.shape[0]
    # Trace each ellipsoid's boundary radially from its unit-weight centre to where the quadric vanishes.
    raw_centres = ellipsoids.solve(mv.w)                                         # [n_cameras] Point-
    centres = raw_centres / (mv.w & raw_centres)                                 # [n_cameras] Point-
    angles = np.linspace(0.0, 2 * np.pi, n_curve_segments, endpoint=False)
    turns = (mv.xy * (-angles / 2)).exp()                                        # [n_curve_segments] Motor+
    headings = (turns >> mv.direction([1, 0]))[:, None]                          # [n_curve_segments, 1] Direction-
    reach = (-(ellipsoids(centres) & centres) / (ellipsoids(headings) & headings)).square_root()  # [n_curve_segments, n_cameras] Scalar+
    outlines = centres + headings * reach                                        # [n_curve_segments, n_cameras] Point-
    pictures = []
    for index, colour in enumerate(camera_colours[:n_cameras]):
        fan_angles = spreads[index] * np.linspace(-1, 1, n_curve_segments + 1)   # [n_curve_segments + 1] Scalar+
        fan_motors = motors[index] * (mv.xy * (-fan_angles / 2)).exp()           # [n_curve_segments + 1] Motor+
        fan = fan_motors >> mv.point([0, arc_radius])                            # [n_curve_segments + 1] Point-
        pinhole = motors[index] >> mv.point([0, 0])                              # [] Point-
        spokes = stack((pinhole.broadcast_to((2,)), fan[[0, -1]]), axis=1)       # [2, 2] Point-
        pictures += [quadrics.regions(ellipsoids[index:index + 1, None], colour, camera, 0.28),
                     simplices.strokes(outlines[:, index], Topology.loops([n_curve_segments]), colour, outline_width,
                                       camera, dashes=outline_width * dash_pattern, cap=Cap.BUTT, opacity=0.28),
                     polylines(fan[None], arc_width, colour, 1.0, camera, arc_width * dot_pattern, Cap.BUTT),
                     polylines(spokes, spoke_width, colour, 1.0, camera, spoke_width * dot_pattern, Cap.BUTT)]
    return pictures


def rig_figure(motors: Motor) -> plt.Figure:
    """Top-down figure of the camera rig at `motors` (`[...] Motor+`)."""
    figure, ax, camera = scene_canvas(scene_x_range)
    overlay(ax, camera_layers(motors, camera))
    return figure


def cones_figure(motors: Motor, world_cones: Quadric) -> plt.Figure:
    """Top-down figure of the cameras at `motors` together with their world sight cones."""
    figure, ax, camera = scene_canvas(scene_x_range)
    overlay(ax, cone_layers(motors, world_cones, camera) + camera_layers(motors, camera))
    return figure


def reconstruction_figure(motors: Motor, world_cones: Quadric, fused_splats: Quadric, points: Point) -> plt.Figure:
    """Top-down figure of sight cones fused into splats around the reconstructed landmarks `points`."""
    figure, ax, camera = scene_canvas(scene_x_range)
    overlay(ax, cone_layers(motors, world_cones, camera) + splat_layers(motors, fused_splats, points, camera)
            + camera_layers(motors, camera))
    return figure


def covariance_figure(
    motors: Motor, world_cones: Quadric, fused_splats: Quadric, points: Point, ellipsoids: Quadric, spreads: Scalar,
) -> plt.Figure:
    """Top-down figure of the reconstructed scene overlaid with each camera's position ellipsoid and heading spread."""
    figure, ax, camera = scene_canvas(scene_x_range)
    overlay(ax, cone_layers(motors, world_cones, camera) + splat_layers(motors, fused_splats, points, camera)
            + camera_layers(motors, camera) + covariance_layers(motors, ellipsoids, spreads, camera))
    return figure


def convergence_animation(
    states: Iterable[tuple], room: float = frame_margin,
) -> Iterator[np.ndarray]:
    """Yields RGB frames of an alignment trajectory `states`, framed around the initial cameras and triangulated landmarks with fractional margin `room`."""
    states = iter(states)
    first = next(states)
    first_motors, _, _, first_points, *_ = first
    drawn = coordinates(concatenate((camera_marks(first_motors).reshape(-1), first_points)))  # [..., 2]
    low, high = drawn.min(axis=0), drawn.max(axis=0)
    margin = room * (high - low).max()
    figure, ax, camera = scene_canvas((low[0] - margin, high[0] + margin), (low[1] - margin, high[1] + margin))
    image = overlay(ax, [np.zeros(camera.shape + (4,))])
    for motors, world_cones, fused_splats, points, *_ in chain([first], states):
        image.set_data(drawing.composited(cone_layers(motors, world_cones, camera)
                                          + splat_layers(motors, fused_splats, points, camera)
                                          + camera_layers(motors, camera)))
        yield capture(figure)
    plt.close(figure)
