"""World-space 3D views and sensor photographs draw the five-part robot arm, two-lens camera, and refracted rays on the GPU.

Solid quadrics and rim loops render the floor grid, articulated box links, lens rims, translucent sensor chip, and piecewise ray paths in 3D, while flat-shaded triangles render the depth-sorted raster photograph on the sensor pixel grid.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np

from numga import concatenate, stack
from numga.render import Scene, drawing, meshes, simplices, solids
from numga.topology import Topology

from examples.geometry.scenegraph.core import Motor, Plane, Point, ga, mv, unit_box_topology
from examples.util.animation import capture
from examples.util.panels import fitted, image_figure, show


body_colors = [
    "#4A5568",
    "#2B6CB0",
    "#3182CE",
    "#4299E1",
    "#ED8936",
]

ray_colors = [
    "#E53E3E",
    "#38A169",
    "#D69E2E",
]

floor_color, front_lens_color, rear_lens_color = "#E2E8F0", "#319795", "#805AD5"
sensor_color, sensor_edge_color, sensor_opacity = "#F6AD55", "#C05621", 0.33
floor_lines = np.linspace(-1.2, 1.2, 7)
n_rim = 72
elevation, azimuth = np.radians(18.0), np.radians(-55.0 + 90.0)
margin, n_pixels = 0.1, 450
floor_width, rim_width, ray_width, hit_radius = 3.0, 4.5, 3.5, 5.0
scene_thickness, animation_thickness = 0.5, 0.37
# Half-turn about x flips downward raster rows upward and orients sensor depth toward the viewer.
half_turn_about_x = mv.yz                                                    # [] Motor+


# --- 3D scene view ----------------------------------------------------------------------
def floor_grid() -> Point:
    """Fourteen endpoint pairs (`[14, 2] Point-`) form the seven-by-seven floor grid in the plane `z == 0`."""
    along_x = mv.point(np.stack(np.broadcast_arrays(floor_lines[None, [0, -1]], floor_lines[:, None], 0.0), axis=-1))  # [7, 2] Point-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Motor+
    along_y = quarter_turn >> along_x                                        # [7, 2] Point-
    return concatenate([along_x, along_y])                                   # [14, 2] Point-


def camera_optics(camera_pose: Motor) -> tuple[Point, Point]:
    """Front and rear lens rims (`[2, n_rim] Point-`) and sensor chip corners (`[4] Point-`) stand in the world under `camera_pose`."""
    rims = meshes.circles(mv.point([[0.0, 0.3, 0.0], [0.0, 0.25, -0.3]]), mv.xy, n_rim)  # [2, n_rim] Point-
    sensor = mv.point([[-0.8, -0.6, -1.25], [0.8, -0.6, -1.25], [0.8, 0.6, -1.25], [-0.8, 0.6, -1.25]])  # [4] Point-
    return camera_pose >> rims, camera_pose >> sensor


def scene_framing(scenes: list[tuple[Point, Motor, tuple]]) -> solids.View:
    """Shared 3D camera framing encloses the floor grid, articulated links, lens rims, sensor, and rays across `scenes`."""
    points = [floor_grid()]
    for world_vertices, camera_pose, rays in scenes:
        rims, sensor = camera_optics(camera_pose)
        points += [world_vertices, rims, sensor, *rays]
    return solids.view(points, 0.0, elevation, azimuth, margin, n_pixels)


def camera_and_floor(camera_pose: Motor, view: solids.View, thickness: float) -> tuple[list, list, list]:
    """Opaque floor and sensor-border solids, translucent sensor plate, and glossy lens-rim tubes form the static camera rig."""
    rims, sensor = camera_optics(camera_pose)
    chip = solids.plates(sensor[None], view.pixel)                            # [1, 6] Plane+
    edge, edge_joints = solids.edges(sensor, sensor[[1, 2, 3, 0]], thickness * ray_width / 2 * view.pixel)
    grid_lines = floor_grid()                                                 # [14, 2] Point-
    grid = solids.rods(grid_lines[:, 0], grid_lines[:, 1], thickness * floor_width / 2 * view.pixel)
    rim_colours = np.repeat([mcolors.to_rgb(front_lens_color), mcolors.to_rgb(rear_lens_color)], n_rim, axis=0)
    lenses = simplices.tubes(
        rims.reshape(-1), Topology.loops([n_rim] * 2), rim_colours, thickness * rim_width, view, gloss=solids.gloss,
    )
    parts = [
        (grid, np.array(mcolors.to_rgb(floor_color))),
        (edge, np.array(mcolors.to_rgb(sensor_edge_color))),
        (edge_joints, np.array(mcolors.to_rgb(sensor_edge_color))),
    ]
    return parts, [(chip, np.array(mcolors.to_rgb(sensor_color)), sensor_opacity)], [(lenses, 1.0)]


def arm_and_rays(
    world_vertices: Point, rays: tuple, view: solids.View, thickness: float,
) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Solid polyhedra for the five robot-arm links and cylinder-and-sphere quadrics for the traced rays."""
    _, faces = unit_box_topology()
    arm = solids.polyhedra(world_vertices, faces)                             # [5, 6] Plane+
    body_colours = np.array([mcolors.to_rgb(color) for color in body_colors])
    n_rays = rays[0].shape[0]
    paths = stack(rays, axis=-1)                                              # [n_rays, 4] Point-
    n_segments = paths.shape[1] - 1
    ray_colours = np.array([mcolors.to_rgb(color) for color in ray_colors[:n_rays]])
    beams, beam_joints = solids.edges(
        paths[:, :-1].reshape(-1), paths[:, 1:].reshape(-1), thickness * ray_width / 2 * view.pixel,
    )
    hits = solids.ball(rays[3], thickness * hit_radius * view.pixel)[:, None]  # [n_rays, 1] Plane+ <- Point-
    return [
        (arm, body_colours),
        (beams, np.repeat(ray_colours, n_segments, axis=0)),
        (beam_joints, np.repeat(ray_colours, 2 * n_segments, axis=0)),
        (hits, ray_colours),
    ]


def world_scene_image(
    world_vertices: Point, camera_pose: Motor, rays: tuple, view: solids.View, thickness: float,
) -> np.ndarray:
    """Shaded RGB image of the 3D robot arm, compound camera, floor grid, and traced optical rays."""
    parts, translucent, lenses = camera_and_floor(camera_pose, view, thickness)
    return solids.draw(
        parts + arm_and_rays(world_vertices, rays, view, thickness), view, np.ones(3),
        inks=[], translucent=translucent, meshes=lenses,
    )


# --- sensor photograph ------------------------------------------------------------------
Scalar = ga.gatype.scalar()                                                 # Scalar+
sensor_pixels = np.array([640, 480])
photo_gloss = 0.15
sensor_margin = 0.015
border_colour, border_width = "#4A5568", 1.2


def sensor_view(ax: plt.Axes) -> solids.View:
    """Orthographic view aimed squarely at the `sensor_pixels` raster grid with fractional margin `sensor_margin`."""
    ax.set_axis_off()
    middle = mv.point([sensor_pixels[0] / 2, -sensor_pixels[1] / 2, 0.0])
    return fitted(
        solids.View(solids.aim(middle, np.pi / 2, 0.0), (1 + sensor_margin) * sensor_pixels / 2, sensor_pixels[0]),
        ax,
    )


def point_spread(points: Point, planes: Plane) -> Scalar:
    """Maximum coordinate span (`[5] Scalar+`) of each body's vertices in `points` (`[5, 8] Point-`) across `planes`."""
    readings = planes[None, :, None] & points.normalized()[:, None, :]       # [5, ..., 8] Scalar+
    spans = readings.max(axis=-1) - readings.min(axis=-1)
    return spans.max(axis=-1)


def photograph_image(
    world_vertices: Point, projected_pixels: Point, camera_pose: Motor, view: solids.View,
) -> np.ndarray:
    """Flat-shaded RGB raster photograph of the posed arm on the sensor grid, depth-buffered by camera-frame `z`."""
    _, faces = unit_box_topology()
    depth = mv.z & (camera_pose << world_vertices).normalized()               # [5, 8] Scalar+
    # Match relief depth scale to pixel magnification so face normals catch directional lighting faithfully.
    scales = point_spread(projected_pixels, stack((mv.x, mv.y))) / point_spread(world_vertices, stack((mv.x, mv.y, mv.z)))
    ordered = scales[scales.argsort()]
    scale = (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2
    on_sensor = projected_pixels - mv.z.dual() * (mv.z & projected_pixels)    # [5, 8] Point-
    relief = on_sensor + mv.z.dual() * ((depth - depth.mean()) * scale)       # [5, 8] Point-
    corners = (half_turn_about_x >> relief).reshape(-1)                       # [40] Point-
    body_vertices = np.arange(np.prod(relief.shape)).reshape(relief.shape)    # [5, 8]
    triangles = body_vertices[:, faces][..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    colours = np.repeat([mcolors.to_rgb(colour) for colour in body_colors], 2 * len(faces), axis=0)[:, None]
    return meshes.draw(
        corners, triangles, colours, meshes.flat(corners, triangles), view, np.ones(3),
        face_values=np.zeros(4), colour=meshes.plain, gloss=photo_gloss,
    )


def framed_photograph(
    world_vertices: Point, projected_pixels: Point, camera_pose: Motor, view: solids.View, width: float,
) -> np.ndarray:
    """RGBA sensor photograph composited with a rectangular frame border of pixel `width`."""
    photograph = photograph_image(world_vertices, projected_pixels, camera_pose, view) / 255
    width_px, height_px = sensor_pixels
    corners = mv.point([[0, 0, 0], [width_px, 0, 0], [width_px, -height_px, 0], [0, -height_px, 0]])  # [4] Point-
    border = simplices.strokes(
        corners, Topology.loops([4]), np.array(mcolors.to_rgb(border_colour)), width, simplices.view_camera(view),
    )
    return drawing.composited([np.concatenate([photograph, np.ones(photograph.shape[:2] + (1,))], axis=-1), border])


def sensor_photograph_artist(
    ax: plt.Axes, world_vertices: Point, projected_pixels: Point, camera_pose: Motor, view: solids.View,
) -> AxesImage:
    """Image artist displaying the framed sensor photograph on `ax`."""
    return show(
        ax, framed_photograph(world_vertices, projected_pixels, camera_pose, view, border_width * ax.figure.dpi / 72),
        view,
    )


# --- figures ----------------------------------------------------------------------------
def world_scene_figure(world_vertices: Point, camera_pose: Motor, rays: tuple) -> plt.Figure:
    """Standalone figure of the 3D robot arm, compound lens train, sensor chip, and traced rays."""
    return image_figure(
        world_scene_image(
            world_vertices, camera_pose, rays, scene_framing([(world_vertices, camera_pose, rays)]), scene_thickness,
        )
    )


def sensor_photograph_figure(world_vertices: Point, projected_pixels: Point, camera_pose: Motor) -> plt.Figure:
    """Standalone figure of the framed sensor photograph."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=70, layout="constrained")
    sensor_photograph_artist(ax, world_vertices, projected_pixels, camera_pose, sensor_view(ax))
    return fig


def scenegraph_axes(dpi: int) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Side-by-side figure and `(ax_3d, ax_2d)` axes for the 3D scene and sensor photograph at `dpi`."""
    fig = plt.figure(figsize=(11.5, 5.5), dpi=dpi)
    ax_3d = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax_2d = fig.add_axes([0.56, 0.08, 0.41, 0.84])
    return fig, ax_3d, ax_2d


def scenegraph_figure(world_vertices: Point, projected_pixels: Point, camera_pose: Motor, rays: tuple) -> plt.Figure:
    """Side-by-side figure pairing the 3D scene and traced optical rays with the resulting sensor photograph."""
    fig, ax_3d, ax_2d = scenegraph_axes(140)
    view = fitted(scene_framing([(world_vertices, camera_pose, rays)]), ax_3d)
    show(ax_3d, world_scene_image(world_vertices, camera_pose, rays, view, 1.0), view)
    sensor_photograph_artist(ax_2d, world_vertices, projected_pixels, camera_pose, sensor_view(ax_2d))
    return fig


def scenegraph_animation(frames: Iterable) -> Iterator[np.ndarray]:
    """Yields RGB frames of the side-by-side 3D scene and sensor photograph across `frames`."""
    frames = list(frames)
    fig, ax_3d, ax_2d = scenegraph_axes(52)
    view = fitted(
        scene_framing([(world_vertices, camera_pose, rays) for world_vertices, _, camera_pose, rays in frames]),
        ax_3d,
    )
    sensor = sensor_view(ax_2d)
    world_vertices, projected_pixels, camera_pose, rays = frames[0]
    parts, translucent, lenses = camera_and_floor(camera_pose, view, animation_thickness)
    scene = Scene(view, np.ones(3), parts, lenses, inks=[], translucent=translucent)
    scene_image = show(
        ax_3d, scene.draw(arm_and_rays(world_vertices, rays, view, animation_thickness), inks=[], translucent=[], meshes=[]),
        view,
    )
    photograph = sensor_photograph_artist(ax_2d, world_vertices, projected_pixels, camera_pose, sensor)
    yield capture(fig)
    for world_vertices, projected_pixels, camera_pose, rays in frames[1:]:
        scene_image.set_data(
            scene.draw(arm_and_rays(world_vertices, rays, view, animation_thickness), inks=[], translucent=[], meshes=[])
        )
        photograph.set_data(
            framed_photograph(world_vertices, projected_pixels, camera_pose, sensor, border_width * fig.dpi / 72)
        )
        yield capture(fig)
    plt.close(fig)
