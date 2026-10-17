"""Ordered points and the chosen sides of a circle's two cutting surfaces."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.orientation.conformal_circle.core import Point, Sphere

PLANE = "#218c9d"
SPHERE = "#ce713f"
CURVE = "#3d354a"
MARKERS = ("#9861ba", "#dd9650", "#4f9b85")


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    values = points.dual().cast(points.algebra.subspace("x y z w e")).kernel
    return values[..., :3] / (values[..., 4:] - values[..., 3:4])


def draw_circle(points: Point, curve: Point, surface: Point, planes: Sphere,
                spheres: Sphere) -> plt.Figure:
    vertices = coordinates(points)
    curves = coordinates(curve)
    surfaces = coordinates(surface)
    plane_values = planes.cast(planes.algebra.subspace("x y z w e")).kernel
    sphere_values = spheres.cast(spheres.algebra.subspace("x y z w e")).kernel

    figure = plt.figure(figsize=(10, 4.5), layout="constrained")
    sheet = np.array([[-1.23, -1.23, 0], [1.23, -1.23, 0],
                      [1.23, 1.23, 0], [-1.23, 1.23, 0]])
    for index in range(len(vertices)):
        ax = figure.add_subplot(1, 2, index + 1, projection="3d", computed_zorder=False)
        ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), zlim=(-1.4, 1.4))
        ax.set_box_aspect((1, 1, 1), zoom=1.25)
        ax.view_init(elev=26, azim=-58)
        ax.set_axis_off()

        ax.add_collection3d(Poly3DCollection([sheet], facecolors=PLANE, edgecolors=PLANE,
                                            alpha=0.09, linewidths=0.8, zorder=1))
        globe = surfaces[index]
        ax.plot_wireframe(*np.moveaxis(globe, -1, 0), color=SPHERE,
                          linewidth=0.6, alpha=0.27, zorder=2)
        circle = curves[index]
        ax.plot(*circle.T, color=CURVE, linewidth=2.8, zorder=4)
        ax.scatter(*vertices[index].T, c=MARKERS, s=85, edgecolors="white",
                   linewidths=1.5, depthshade=False, zorder=6)

        # Short arrows follow the point order around the circle in both views.
        count = len(circle) - 1
        starts = np.arange(3) * count // 3 + count // 12
        chord = circle[starts + count // 18] - circle[starts]
        ax.quiver(*circle[starts].T, *chord.T, color=CURVE, linewidth=2.3,
                  arrow_length_ratio=0.6, zorder=5)

        # The arrows point into the positive sides of the incidence functions.
        plane_arrow = -plane_values[index, :3]
        ax.quiver(0, 0, 0, *plane_arrow, length=0.85, color=PLANE,
                  linewidth=3.5, arrow_length_ratio=0.24, zorder=7)
        sphere_anchor = circle[count // 3]
        sphere_arrow = (-sphere_values[index, :3]
                        - (sphere_values[index, 3] - sphere_values[index, 4]) * sphere_anchor)
        ax.quiver(*sphere_anchor, *sphere_arrow, length=0.45, color=SPHERE,
                  linewidth=3, arrow_length_ratio=0.3, zorder=7)
    return figure


def animate_circle_motions(grid: Point, fixed_points: Point) -> list[np.ndarray]:
    """A coloured disk turns rigidly beside its reflected, conformally distorted image."""
    positions = coordinates(grid)[..., :2]
    fixed = coordinates(fixed_points)[..., :2]
    ring_indices = np.linspace(0, positions.shape[2] - 1, 6, dtype=int)[1:-1]
    spoke_indices = np.linspace(0, positions.shape[3] - 1, 13, dtype=int)[:-1]
    colours = plt.get_cmap("twilight_shifted")(np.linspace(0, 1, len(spoke_indices), endpoint=False))

    figure, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=90)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, wspace=0.06)
    artists = []
    for ax, disk in zip(axes, positions[0]):
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), aspect="equal")
        ax.set_axis_off()
        ax.fill(*disk[-1].T, color="#f5f6f8", zorder=0)
        ax.plot(*disk[-1].T, color=CURVE, linewidth=1.8, zorder=2)
        rings = LineCollection(disk[ring_indices], colors="#b8c1cc", linewidths=1)
        spokes = LineCollection(disk[:, spoke_indices].transpose(1, 0, 2),
                                colors=colours, linewidths=1.4, alpha=0.85)
        ax.add_collection(rings)
        ax.add_collection(spokes)
        dots = ax.scatter(*disk[-1, spoke_indices].T, c=colours, s=48,
                          edgecolors="white", linewidths=1, zorder=4)
        artists.append((rings, spokes, dots))
    axes[1].scatter(*fixed.T, s=135, facecolors="none", edgecolors=CURVE,
                    linewidths=1.6, zorder=5)

    frames = []
    for pair in positions:
        for disk, (rings, spokes, dots) in zip(pair, artists):
            rings.set_segments(disk[ring_indices])
            spokes.set_segments(disk[:, spoke_indices].transpose(1, 0, 2))
            dots.set_offsets(disk[-1, spoke_indices])
        frames.append(capture(figure))
    plt.close(figure)
    return frames
