"""Drawing and coordinate read-out for the scenegraph example.

Figures show the robot arm and the camera's optics in the world, drawn on the GPU, and the photograph on
the camera's sensor. The mathematics never imports this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
import numpy as np

from numga import stack
from examples.util.animation import capture
from examples.geometry.scenegraph.core import Motor, Point, mv, unit_box_topology
from examples.util.wgsl import image_figure, meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene


# Colors for the robot arm bodies:
BODY_COLORS = [
    "#4A5568",  # Body 0: Base pedestal (charcoal slate)
    "#2B6CB0",  # Body 1: Turret (steel blue)
    "#3182CE",  # Body 2: Upper arm (vibrant azure)
    "#4299E1",  # Body 3: Forearm (sky blue)
    "#ED8936",  # Body 4: Gripper wrist (accent amber)
]

RAY_COLORS = [
    "#E53E3E",  # Red / crimson
    "#38A169",  # Green / emerald
    "#D69E2E",  # Golden amber
]

FLOOR_COLOR, FRONT_LENS_COLOR, REAR_LENS_COLOR = "#E2E8F0", "#319795", "#805AD5"
# The sensor chip's colours, and how much of what lies behind each of its two faces hides.
SENSOR_COLOR, SENSOR_EDGE_COLOR, SENSOR_OPACITY = "#F6AD55", "#C05621", 0.33
# The floor grid's lines in the plane z == 0, and the points along a lens rim.
FLOOR_LINES = np.linspace(-1.2, 1.2, 7)
N_RIM = 72
# The view's elevation and azimuth, in radians, azimuth zero looking along +y; the margin as a share of the larger
# side, and the pixels across the scene.
ELEVATION, AZIMUTH = np.radians(18.0), np.radians(-55.0 + 90.0)
MARGIN, N_PIXELS = 0.1, 900
# The widths of the floor grid, the lens rims, and the rays and the sensor's edge, and the radius of a ray's hit on
# the sensor, in pixels.
FLOOR_WIDTH, RIM_WIDTH, RAY_WIDTH, HIT_RADIUS = 3.0, 4.5, 3.5, 5.0


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


# --- 3D scene view ----------------------------------------------------------------------
def floor() -> np.ndarray:
    """The floor grid's lines in the plane z == 0, across x and across y, `[lines, 2, 3]`."""
    # Lines along x, from end to end of the grid, at each place across y; and the same with x and y swapped.
    along_x = np.stack(np.broadcast_arrays(FLOOR_LINES[None, [0, -1]], FLOOR_LINES[:, None], 0.0), axis=-1)
    return np.concatenate([along_x, along_x[..., [1, 0, 2]]])


def optics(camera_pose: Motor) -> tuple[np.ndarray, np.ndarray]:
    """The optical elements in the camera frame, moved into the world by the camera pose: the front lens rim,
    radius 0.3 at z == 0, and the rear lens rim, radius 0.25 at z == -0.3, `[2, rim + 1, 3]`; and the corners of the
    1.6 x 1.2 sensor chip at z == -1.25, `[4, 3]`."""
    rims = meshes.circles(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.3]]), np.array([0.3, 0.25]), np.array([0.0, 0.0, 1.0]), N_RIM)
    sensor = np.array([[-0.8, -0.6, -1.25], [0.8, -0.6, -1.25], [0.8, 0.6, -1.25], [-0.8, 0.6, -1.25]])
    return euclidean(camera_pose >> mv.point(rims)), euclidean(camera_pose >> mv.point(sensor))


def framing(scenes: list[tuple[Point, Motor, tuple]]) -> solids.View:
    """One view framing the arm, the optics, the rays and the floor of every scene."""
    points = [mv.point(floor())]
    for world_vertices, camera_pose, rays in scenes:
        rims, sensor = optics(camera_pose)
        points += [world_vertices, mv.point(rims), mv.point(sensor), *rays]
    return solids.view(points, 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def camera_and_floor(camera_pose: Motor, view: solids.View) -> tuple[list, list, list]:
    """What stands with the camera, and the floor: the floor grid and the sensor's edge as solids, the sensor chip as
    a translucent solid, and the lens rims as strokes, for `solids.draw` and `Scene`."""
    rims, sensor = optics(camera_pose)
    # The sensor chip as a plate a pixel thick, with its edge.
    chip = solids.plates(sensor[None], view.pixel)                            # [1, 6] Quadric
    corners = mv.point(sensor)                                                # [4] Point
    edge, edge_joints = solids.edges(corners, corners[[1, 2, 3, 0]], RAY_WIDTH / 2 * view.pixel)
    grid = solids.rods(mv.point(floor()[:, 0]), mv.point(floor()[:, 1]), FLOOR_WIDTH / 2 * view.pixel)
    lenses = strokes.polylines([(rims[0], mcolors.to_rgb(FRONT_LENS_COLOR)), (rims[1], mcolors.to_rgb(REAR_LENS_COLOR))],
                               RIM_WIDTH, view)
    parts = [(grid, np.array(mcolors.to_rgb(FLOOR_COLOR))),
             (edge, np.array(mcolors.to_rgb(SENSOR_EDGE_COLOR))), (edge_joints, np.array(mcolors.to_rgb(SENSOR_EDGE_COLOR)))]
    return parts, [(chip, np.array(mcolors.to_rgb(SENSOR_COLOR)), SENSOR_OPACITY)], [(lenses, 1.0)]


def arm_and_rays(world_vertices: Point, rays: tuple, view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The articulated robot arm and the traced rays, with their hits on the sensor, as solids."""
    _, faces = unit_box_topology()
    # Every body a box, the half-spaces inside its faces.
    arm = solids.polyhedra(euclidean(world_vertices), faces)                  # [bodies, faces] Quadric
    body_colours = np.array([mcolors.to_rgb(color) for color in BODY_COLORS])
    # Traced rays: scene point to pupil, pupil to rear lens, rear lens to sensor, each a run of segments.
    paths = np.stack([euclidean(stage) for stage in rays], axis=-2)           # [rays, stages, 3]
    segments = paths.shape[1] - 1
    ray_colours = np.array([mcolors.to_rgb(color) for color in RAY_COLORS[:len(paths)]])
    beams, beam_joints = solids.edges(mv.point(paths[:, :-1].reshape(-1, 3)), mv.point(paths[:, 1:].reshape(-1, 3)),
                                      RAY_WIDTH / 2 * view.pixel)
    hits = solids.ball(rays[3], HIT_RADIUS * view.pixel)[:, None]             # [rays, 1] Quadric
    return [(arm, body_colours),
            (beams, np.repeat(ray_colours, segments, axis=0)), (beam_joints, np.repeat(ray_colours, 2 * segments, axis=0)),
            (hits, ray_colours)]


def scene_3d(world_vertices: Point, camera_pose: Motor, rays: tuple, view: solids.View) -> np.ndarray:
    """The 3D world scene: articulated robot arm, camera lenses and sensor, and traced rays, `[rows, columns, 3]`."""
    parts, translucent, lenses = camera_and_floor(camera_pose, view)
    return solids.draw(parts + arm_and_rays(world_vertices, rays, view), view, np.ones(3), translucent=translucent,
                       meshes=lenses)


# --- sensor photograph ------------------------------------------------------------------
# The sensor's pixels across and up, and the photograph's glint, faint, as on painted metal.
SENSOR_PIXELS = np.array([640, 480])
PHOTO_GLOSS = 0.15


def photograph_image(world_vertices: Point, projected_pixels: Point, camera_pose: Motor) -> tuple[np.ndarray, solids.View]:
    """The photograph on the sensor's pixels, rendered, and the view it is seen by: each box's faces at their
    places on the sensor, held at their depth in front of the lens, so nearer faces hide farther ones."""
    _, faces = unit_box_topology()
    pixels = euclidean(projected_pixels)[..., :2]                              # [bodies, vertices, 2]
    # The depth in front of the lens, along the camera's own z.
    seen = camera_pose << world_vertices
    depth = ((mv.z & seen) / (mv.w & seen)).to_array()                        # [bodies, vertices]
    # The depth in the photograph's own units, at the scale the camera shows the arm, so the faces keep their slopes
    # and catch the light much as they would: the pixels the bodies span over the room they take in the world.
    world = euclidean(world_vertices)
    scale = np.ptp(pixels, axis=1).max(axis=-1) / np.ptp(world, axis=1).max(axis=-1)   # [bodies]
    places = np.concatenate([pixels * [1, -1], -(depth - depth.mean()) [..., None] * np.median(scale)], axis=-1)
    corners = mv.point(places.reshape(-1, 3))                                 # [bodies * vertices] Point
    quads = faces[None] + len(places[0]) * np.arange(len(places))[:, None, None]   # [bodies, faces, 4]
    triangles = quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    colours = np.repeat([mcolors.to_rgb(colour) for colour in BODY_COLORS], 2 * len(faces), axis=0)[:, None]
    # The sensor seen square on, its middle at the photograph's.
    middle = mv.point([SENSOR_PIXELS[0] / 2, -SENSOR_PIXELS[1] / 2, 0.0])
    view = solids.View(solids.aim(middle, np.pi / 2, 0.0), SENSOR_PIXELS / 2, SENSOR_PIXELS[0])
    image = meshes.draw(corners, triangles, colours, meshes.flat(corners, triangles), view, np.ones(3), gloss=PHOTO_GLOSS)
    return image, view


def plot_camera_image(ax: plt.Axes, world_vertices: Point, projected_pixels: Point, camera_pose: Motor) -> AxesImage:
    """The photograph on the sensor, with the sensor's edge. The photograph shown, for an animation to set its data."""
    image, view = photograph_image(world_vertices, projected_pixels, camera_pose)
    shown = show(ax, image, view)
    left, _, bottom, _ = view.extent
    ax.add_patch(Rectangle((left, bottom), *SENSOR_PIXELS, fill=False, edgecolor="#4A5568", linewidth=1.2))
    return shown


# --- figures ----------------------------------------------------------------------------
def draw_scene_3d(world_vertices: Point, camera_pose: Motor, rays: tuple) -> plt.Figure:
    """The 3D scene on its own."""
    return image_figure(scene_3d(world_vertices, camera_pose, rays, framing([(world_vertices, camera_pose, rays)])))


def draw_camera_image(world_vertices: Point, projected_pixels: Point, camera_pose: Motor) -> plt.Figure:
    """The sensor photograph on its own."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=140, layout="constrained")
    plot_camera_image(ax, world_vertices, projected_pixels, camera_pose)
    return fig


def scenegraph_axes(dpi: int):
    """Side-by-side 3D scene and sensor photograph."""
    fig = plt.figure(figsize=(11.5, 5.5), dpi=dpi)
    ax_3d = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax_2d = fig.add_axes([0.56, 0.08, 0.41, 0.84])
    return fig, ax_3d, ax_2d


def draw_scenegraph(world_vertices: Point, projected_pixels: Point, camera_pose: Motor, rays: tuple) -> plt.Figure:
    """The 3D scene with its optics beside the photograph on the sensor."""
    fig, ax_3d, ax_2d = scenegraph_axes(140)
    view = framing([(world_vertices, camera_pose, rays)])
    show(ax_3d, scene_3d(world_vertices, camera_pose, rays, view), view)
    plot_camera_image(ax_2d, world_vertices, projected_pixels, camera_pose)
    return fig


def animate_scenegraph(frames: Iterable) -> Iterator[np.ndarray]:
    """RGB frames of the two-panel figure, one per state, the scene seen by one view that frames them all. The camera
    stands still: its optics and the floor are set up once, from the first state, and the arm and the rays drawn among
    them."""
    frames = list(frames)
    view = framing([(world_vertices, camera_pose, rays) for world_vertices, _, camera_pose, rays in frames])
    fig, ax_3d, ax_2d = scenegraph_axes(100)
    world_vertices, projected_pixels, camera_pose, rays = frames[0]
    parts, translucent, lenses = camera_and_floor(camera_pose, view)
    scene = Scene(view, np.ones(3), parts, lenses, translucent=translucent)
    # Lay the figure out once; only the two pictures change between frames.
    scene_image = show(ax_3d, scene.draw(arm_and_rays(world_vertices, rays, view)), view)
    photograph = plot_camera_image(ax_2d, world_vertices, projected_pixels, camera_pose)
    yield capture(fig)
    for world_vertices, projected_pixels, camera_pose, rays in frames[1:]:
        scene_image.set_data(scene.draw(arm_and_rays(world_vertices, rays, view)))
        photograph.set_data(photograph_image(world_vertices, projected_pixels, camera_pose)[0])
        yield capture(fig)
    plt.close(fig)
