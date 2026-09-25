"""Drawing for the projective camera example.

This module consumes the geometry that `core` produces and turns it into a figure.
It constructs viewport geometry only, never scene geometry, and the mathematics
never imports it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.projection.core import Line, Motor, Point


def euclidean(points: Point) -> np.ndarray:
    """Read xyz coordinates using an explicit coordinate basis, independent of storage order."""
    k = points.cast(points.algebra.subspace("yzw zxw xyw zyx")).kernel
    return k[..., :3] / k[..., 3:]


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
    """Draw the body, both lights, both shadows, and one corner's shadow trail in 3D."""
    point_shadow, sun_shadow, shadow_trail = shadows
    draw_edges(ax, euclidean(body), edges, "#38bdf8", linewidth=2.0)
    draw_edges(ax, euclidean(point_shadow), edges, "#fbbf24", linewidth=1.5)
    draw_edges(ax, euclidean(sun_shadow), edges, "#a855f7", linewidth=1.5, linestyle="--")
    trail = euclidean(shadow_trail)
    ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="#fbbf24", linewidth=1.0, linestyle=":", label="corner shadow as light moves")
    ax.scatter(*euclidean(point_light), color="#fbbf24", s=60, label="point light")
    sun_dir = sun.cast(sun.algebra.subspace("yzw zxw xyw")).kernel
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    ax.quiver(-2.0, 2.0, 4.0, *sun_dir, length=1.0, color="#a855f7", label="sun direction")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(0, 4.5)
    ax.set_box_aspect([1, 1, 0.75])
    ax.set_title("Shadows: (light ∨ point) ∧ ground")
    ax.legend(loc="upper left", fontsize=8)


def screen_line_endpoints(rig: Motor, lines: Line, half_width: float) -> np.ndarray:
    """Clip lines on a rig's screen at x == -half_width and x == half_width, returning (..., 2, 2)
    screen coordinates."""
    mv = rig.context.multivector
    ends = []
    for x in (-half_width, half_width):
        ends.append(screen_coordinates(rig, lines ^ (rig >> (mv.x - mv.w * x))))
    return np.stack(ends, axis=-2)


def render_stereo_scene(
    ax_1, ax_2,
    edges,
    stereo: tuple,
    rig_1: Motor, rig_2: Motor,
    half_width: float,
) -> None:
    """Draw both images; camera 2 also shows the epipolar lines of camera 1's corners."""
    image_1, image_2, _epipole_2, epipolar_lines_2, _correspondence = stereo
    px_1 = screen_coordinates(rig_1, image_1)
    px_2 = screen_coordinates(rig_2, image_2)
    draw_edges(ax_1, px_1, edges, "#38bdf8", linewidth=2.0)
    draw_edges(ax_2, px_2, edges, "#38bdf8", linewidth=2.0)
    for (a, b) in screen_line_endpoints(rig_2, epipolar_lines_2, half_width):
        ax_2.plot([a[0], b[0]], [a[1], b[1]], color="#f43f5e", linewidth=0.8, alpha=0.8)
    ax_2.scatter(px_2[:, 0], px_2[:, 1], color="#f43f5e", s=18, zorder=3)
    for ax, title in ((ax_1, "Camera 1"), (ax_2, "Camera 2 with epipolar lines of camera 1")):
        ax.set_xlim(-half_width, half_width); ax.set_ylim(-half_width, half_width)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.grid(True, alpha=0.3)


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
    render_stereo_scene(fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3), edges, stereo, rig_1, rig_2, 0.5)
    fig.tight_layout()
    return fig
