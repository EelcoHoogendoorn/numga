"""Plane sides and triangle winding under reflection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.orientation.normals.core import Axis, Plane, Point, Spear, Surface, ga

SIDE = "#218c9d"
WINDING = "#ce713f"
FACE = "#b8c1c8"
MIRROR = "#8095aa"
LINE = "#3b4a57"
FIRST = "#c8553d"
SECOND = "#2f6690"


# --- plumbing -------------------------------------------------------------------------
def draw_normals(vertices: Point, side: Plane, winding: Surface, mirror: Plane) -> plt.Figure:
    values = vertices.cast(ga.subspace("wzy wxz wyx xyz")).kernel
    points = values[..., :3] / values[..., 3:]
    centres = points.mean(axis=1)
    side_normals = side.cast(ga.subspace("x y z w")).kernel[..., :3]
    winding_normals = winding.cast(ga.subspace("x y z w")).kernel[..., :3]
    normals = np.stack([side_normals, winding_normals])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    lower = points.min(axis=(0, 1))
    upper = points.max(axis=(0, 1))
    centre = (lower + upper) / 2
    span = np.max(upper - lower)
    radius = span * 0.62
    arrow_length = span * 0.22

    sheet = mirror_sheet(mirror, centre, span * 0.32)

    figure = plt.figure(figsize=(10, 4.2), layout="constrained")
    for index, colour in enumerate((SIDE, WINDING)):
        ax = figure.add_subplot(1, 2, index + 1, projection="3d", computed_zorder=False)
        ax.set(xlim=(centre[0] - radius, centre[0] + radius),
               ylim=(centre[1] - radius, centre[1] + radius),
               zlim=(centre[2] - radius, centre[2] + radius))
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.view_init(elev=38, azim=-68)
        ax.set_axis_off()
        ax.add_collection3d(Poly3DCollection([sheet], facecolors=MIRROR, edgecolors=MIRROR,
                                            alpha=0.12, linewidths=1, zorder=1))
        ax.add_collection3d(Poly3DCollection(points, facecolors=FACE, edgecolors="#59626a",
                                            alpha=0.75, linewidths=1.5, zorder=2))
        ax.quiver(*centres.T, *normals[index].T, length=arrow_length,
                  color=colour, linewidth=3, arrow_length_ratio=0.25, zorder=3)

    # Follow the supplied vertex order, so reflection visibly reverses the winding.
    edges = np.roll(points, -1, axis=1) - points
    starts = (points + edges * 0.25).reshape(-1, 3)
    arrows = (edges * 0.5).reshape(-1, 3)
    ax.quiver(*starts.T, *arrows.T, color=WINDING, linewidth=2.5,
              arrow_length_ratio=0.4, zorder=4)
    return figure


def draw_line_in_mirror(ends: Point, lines: Spear, mirror: Plane) -> plt.Figure:
    """Two points in the mirror and the line through them, one panel per line: as it is, and reflected two ways."""
    values = ends.cast(ga.subspace("wzy wxz wyx xyz")).kernel
    points = values[..., :3] / values[..., 3:]
    middle = points.mean(axis=0)
    directions = lines.cast(ga.subspace("yz zx xy")).kernel
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    span = np.linalg.norm(points[1] - points[0])
    radius = span * 0.75
    sheet = mirror_sheet(mirror, middle, span * 0.7)

    figure = plt.figure(figsize=(14, 4.2), layout="constrained")
    for index, direction in enumerate(directions):
        ax = figure.add_subplot(1, len(directions), index + 1, projection="3d", computed_zorder=False)
        ax.set(xlim=(middle[0] - radius, middle[0] + radius),
               ylim=(middle[1] - radius, middle[1] + radius),
               zlim=(middle[2] - radius, middle[2] + radius))
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.view_init(elev=24, azim=-28)
        ax.set_axis_off()
        ax.add_collection3d(Poly3DCollection([sheet], facecolors=MIRROR, edgecolors=MIRROR,
                                            alpha=0.12, linewidths=1, zorder=1))
        ax.plot(*points.T, color=LINE, linewidth=2, zorder=2)
        ax.scatter(*points.T, color=LINE, s=40, depthshade=False, zorder=3)
        ax.quiver(*(middle - direction * span * 0.2), *direction, length=span * 0.4,
                  color=WINDING, linewidth=3, arrow_length_ratio=0.3, zorder=4)
    return figure


def draw_pairs_in_mirror(planes: Plane, ends: Point, meets: Axis, joins: Spear, mirror: Plane) -> plt.Figure:
    """Two points on a line in the mirror, and two planes crossing in that line, before and after the
    mirror: points in the top row, planes in the bottom row, each coloured by which one it is."""
    values = ends.cast(ga.subspace("wzy wxz wyx xyz")).kernel
    points = values[..., :3] / values[..., 3:]                              # [panels, 2, 3]
    normals = planes.cast(ga.subspace("x y z w")).kernel[..., :3]           # [panels, 2, 3]
    middle = points[0].mean(axis=0)
    along = points[0, 1] - points[0, 0]
    span = np.linalg.norm(along)
    along /= span
    radius = span * 1.1
    sheet = mirror_sheet(mirror, middle, span * 0.95)
    colours = (FIRST, SECOND)

    figure = plt.figure(figsize=(9, 8.4), layout="constrained")
    for row in range(2):
        for panel in range(len(points)):
            ax = figure.add_subplot(2, len(points), row * len(points) + panel + 1, projection="3d",
                                    computed_zorder=False)
            ax.set(xlim=(middle[0] - radius, middle[0] + radius),
                   ylim=(middle[1] - radius, middle[1] + radius),
                   zlim=(middle[2] - radius, middle[2] + radius))
            ax.set_box_aspect((1, 1, 1), zoom=1.3)
            ax.view_init(elev=22, azim=-62)
            ax.set_axis_off()
            ax.add_collection3d(Poly3DCollection([sheet], facecolors=MIRROR, edgecolors=MIRROR,
                                                alpha=0.12, linewidths=1, zorder=1))
            ax.plot(*points[panel].T, color=LINE, linewidth=2.5, zorder=3)
            if row == 0:
                ax.scatter(*points[panel].T, color=colours, s=70, depthshade=False, zorder=4)
                continue
            for normal, colour in zip(normals[panel], colours):
                across = np.cross(normal, along)
                across /= np.linalg.norm(across)
                corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
                patch = middle + span * 0.6 * (corners[:, :1] * across + corners[:, 1:] * along)
                ax.add_collection3d(Poly3DCollection([patch], facecolors=colour, edgecolors=colour,
                                                    alpha=0.35, linewidths=1.5, zorder=2))
    return figure


def mirror_sheet(mirror: Plane, centre: np.ndarray, half_width: float) -> np.ndarray:
    """The corners `[4, 3]` of a square piece of the mirror plane, around the foot of `centre`."""
    values = mirror.cast(ga.subspace("x y z w")).kernel
    length = np.linalg.norm(values[:3])
    normal = values[:3] / length
    foot = centre - normal * ((centre @ values[:3] + values[3]) / length)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    across = np.cross(normal, reference)
    across /= np.linalg.norm(across)
    up = np.cross(normal, across)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return foot + half_width * (corners[:, :1] * across + corners[:, 1:] * up)
