"""Drawing for the projective camera example.

This module consumes the geometry that `core` produces and turns it into a figure.
It constructs viewport geometry only, never scene geometry, and the mathematics
never imports it. The shadow scene is ray traced on the GPU: edges as rods, the
shadow trail as a row of beads, the point light a ball and the sun's direction an
arrow; the two camera images are drawn flat.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_rgb

from numga import stack
from examples.geometry.projection.core import Line, Motor, Point, mv
from examples.util.wgsl import fitted, show, solids

# The image panels' margin around both images, as a fraction of their larger side.
MARGIN = 0.08
# The shadow scene: the colours of the body, of its shadows under the point light and the sun, and of the sun's
# direction, drawn as an arrow of unit length from a place above the scene.
BODY_COLOUR, LIGHT_COLOUR, SUN_COLOUR = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#fbbf24", "#a855f7"))
SUN_ARROW_START = np.array([-2.0, 2.0, 4.0])
background = np.ones(3)
# The shadow scene's view, elevation and azimuth in radians, azimuth zero looking along +y, and its margin as a share
# of the larger side.
elevation, azimuth = np.radians([30.0, -60.0 + 90.0])
scene_margin = 0.1
# Radii in pixels: of the body's edges, of the shadows' edges, of the beads along the trail, of the point light, and
# of the arrow's shaft and head, with its head's length.
body_radius, shadow_radius, bead_radius, light_radius = 1.6, 1.2, 2.4, 6.5
shaft_radius, head_radius, head_length = 2.4, 6.5, 19.5


def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def screen_coordinates(motor: Motor, image: Point) -> np.ndarray:
    """Read (x, y) screen coordinates of image points on a rig moved by motor."""
    return euclidean(motor << image)[..., :2]


def draw_edges(ax, coords: np.ndarray, edges, color: str, **kwargs) -> None:
    """Draw the given edge list through 2D or 3D coordinates."""
    for a, b in edges:
        ax.plot(*zip(coords[a], coords[b]), color=color, **kwargs)


def render_shadow_scene(
    ax,
    body: Point,
    edges,
    shadows: tuple,
    point_light: Point,
    sun: Point,
) -> None:
    """Draw the body, blue, its shadows under the point light, yellow, and the sun, purple, the point light, the
    sun's direction, and one corner's shadow trail as the light moves, beaded."""
    point_shadow, sun_shadow, shadow_trail = shadows
    ends = np.array(edges)                                                     # [edges, 2]
    # The sun's direction, its pairings with the planes x, y and z.
    sun_dir = (stack((mv.x, mv.y, mv.z)) & sun).to_array()
    sun_end = SUN_ARROW_START + sun_dir / np.linalg.norm(sun_dir)
    corners, light_corners, sun_corners, trail, light = (
        solids.mv.point(euclidean(points)) for points in (body, point_shadow, sun_shadow, shadow_trail, point_light))
    arrow_start, arrow_end = solids.mv.point(SUN_ARROW_START), solids.mv.point(sun_end)
    # Framed by everything drawn, as many pixels across as the axes' room holds.
    view = fitted(solids.view([corners, light_corners, sun_corners, trail, light, arrow_start, arrow_end], 0.0,
                              elevation, azimuth, scene_margin, 1), ax)
    pixel = view.pixel
    shaft, head = solids.arrows(arrow_start, arrow_end, shaft_radius * pixel, head_length * pixel, head_radius * pixel)
    rods, joints = solids.edges(corners[ends[:, 0]], corners[ends[:, 1]], body_radius * pixel)
    parts = [(rods, BODY_COLOUR), (joints, BODY_COLOUR),
             (solids.rods(light_corners[ends[:, 0]], light_corners[ends[:, 1]], shadow_radius * pixel), LIGHT_COLOUR),
             (solids.rods(sun_corners[ends[:, 0]], sun_corners[ends[:, 1]], shadow_radius * pixel), SUN_COLOUR),
             (solids.ball(trail, bead_radius * pixel)[:, None], LIGHT_COLOUR),
             (solids.ball(light, light_radius * pixel)[None, None], LIGHT_COLOUR),
             (shaft[None], SUN_COLOUR), (head[None], SUN_COLOUR)]
    show(ax, solids.draw(parts, view, background, inks=[], translucent=[], meshes=[]), view)


def screen_line_endpoints(rig: Motor, lines: Line, half_width: float) -> np.ndarray:
    """Clip lines on a rig's screen at x == -half_width and x == half_width, returning (..., 2, 2)
    screen coordinates."""
    return np.stack([screen_coordinates(rig, lines ^ (rig >> (mv.x - mv.w * x))) for x in (-half_width, half_width)], axis=-2)


def render_stereo_scene(
    ax_1, ax_2,
    edges,
    stereo: tuple,
    rig_1: Motor, rig_2: Motor,
    margin: float,
) -> None:
    """Draw both images, the subject's edges, blue; camera 2's also shows its corners, red, on the epipolar
    lines of camera 1's corners. Both are framed alike, by the images with a margin as a fraction of their
    larger side."""
    image_1, image_2, _epipole_2, epipolar_lines_2, _correspondence = stereo
    px_1 = screen_coordinates(rig_1, image_1)
    px_2 = screen_coordinates(rig_2, image_2)
    both = np.concatenate([px_1, px_2])
    low, high = both.min(axis=0), both.max(axis=0)
    low, high = low - margin * (high - low).max(), high + margin * (high - low).max()
    draw_edges(ax_1, px_1, edges, "#38bdf8", linewidth=2.0)
    draw_edges(ax_2, px_2, edges, "#38bdf8", linewidth=2.0)
    for (a, b) in screen_line_endpoints(rig_2, epipolar_lines_2, np.abs([low[0], high[0]]).max()):
        ax_2.plot([a[0], b[0]], [a[1], b[1]], color="#f43f5e", linewidth=0.8, alpha=0.8)
    ax_2.scatter(px_2[:, 0], px_2[:, 1], color="#f43f5e", s=18, zorder=3)
    for ax in (ax_1, ax_2):
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
        ax.set_axis_off()


def draw_projection(
    body: Point,
    edges,
    shadows: tuple,
    point_light: Point,
    sun: Point,
    stereo: tuple,
    rig_1: Motor,
    rig_2: Motor,
) -> plt.Figure:
    """Lay out the shadow scene and both camera views."""
    fig = plt.figure(figsize=(16, 5), dpi=120)
    scene_axes, camera_1_axes, camera_2_axes = (fig.add_subplot(1, 3, i + 1) for i in range(3))
    render_stereo_scene(camera_1_axes, camera_2_axes, edges, stereo, rig_1, rig_2, MARGIN)
    # The figure laid out, the scene's axes as bare as its picture leaves them, before the picture is traced at the
    # size of their room.
    scene_axes.set_axis_off()
    fig.tight_layout()
    render_shadow_scene(scene_axes, body, edges, shadows, point_light, sun)
    return fig
