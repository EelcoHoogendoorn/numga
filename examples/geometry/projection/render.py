"""Drawing for the projective camera example.

This module consumes the geometry that `core` produces and turns it into a figure.
It constructs viewport geometry only, never scene geometry, and the mathematics
never imports it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.geometry.projection.core import Line, Motor, Point, mv

# The image panels' margin around both images, as a fraction of their larger side.
MARGIN = 0.08


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
    """Draw the body, blue, its shadows under the point light, yellow, and the sun, dashed purple, the point
    light, the sun's direction, and one corner's shadow trail as the light moves, dotted."""
    point_shadow, sun_shadow, shadow_trail = shadows
    draw_edges(ax, euclidean(body), edges, "#38bdf8", linewidth=2.0)
    draw_edges(ax, euclidean(point_shadow), edges, "#fbbf24", linewidth=1.5)
    draw_edges(ax, euclidean(sun_shadow), edges, "#a855f7", linewidth=1.5, linestyle="--")
    trail = euclidean(shadow_trail)
    light, sun_start = euclidean(point_light), np.array([-2.0, 2.0, 4.0])
    ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="#fbbf24", linewidth=1.0, linestyle=":")
    ax.scatter(*light, color="#fbbf24", s=60)
    # The sun's direction, its pairings with the planes x, y and z.
    sun_dir = (stack((mv.x, mv.y, mv.z)) & sun).to_array()
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    ax.quiver(*sun_start, *sun_dir, length=1.0, color="#a855f7")
    # Framed by everything drawn, at equal aspect.
    drawn = np.concatenate([euclidean(body), euclidean(point_shadow), euclidean(sun_shadow), trail, light[None],
                            sun_start[None], (sun_start + sun_dir)[None]])
    low, high = drawn.min(axis=0), drawn.max(axis=0)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(high - low, zoom=1.05)
    ax.set_axis_off()


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
    render_shadow_scene(fig.add_subplot(1, 3, 1, projection="3d"), body, edges, shadows, point_light, sun)
    render_stereo_scene(fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3), edges, stereo, rig_1, rig_2, MARGIN)
    fig.tight_layout()
    return fig
