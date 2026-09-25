"""Drawing and coordinate readout for the Cayley-Klein example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.cayley_klein.core import Line, Point, Polarity, Scalar, ga, point

BOX = (-1.2, 1.9, -1.2, 1.6)
# Grid samples per axis for level sets.
RESOLUTION = 400


def euclidean(points: Point) -> np.ndarray:
    """Dehomogenize points to (..., 2) chart coordinates."""
    k = points.cast(ga.subspace("yw wx xy")).kernel
    return k[..., :2] / k[..., 2:]


def draw_points(ax, points: Point, **style) -> None:
    xy = euclidean(points).reshape(-1, 2)
    ax.scatter(xy[:, 0], xy[:, 1], zorder=6, **style)


def draw_lines(ax, lines: Line, **style) -> None:
    """Draw lines mv.x * a + mv.y * b + mv.w * c through their foot from the origin, along their
    direction."""
    a, b, c = np.moveaxis(lines.cast(ga.subspace("x y w")).kernel.reshape(-1, 3), -1, 0)
    foot = -np.stack([a, b], axis=-1) * (c / (a * a + b * b))[:, None]
    along = np.stack([-b, a], axis=-1)
    for start, direction in zip(foot, along):
        ax.axline(start, start + direction, **style)


def draw_level_sets(ax, quadrics: Polarity, **style) -> None:
    """Draw each locus P & quadric(P) == 0 by contouring it on a grid."""
    xs = np.linspace(BOX[0], BOX[1], RESOLUTION)
    ys = np.linspace(BOX[2], BOX[3], RESOLUTION)
    samples = point(np.stack(np.meshgrid(xs, ys), axis=-1))
    quadrics = quadrics.reshape(-1)
    values = (samples[..., None] & quadrics(samples[..., None])).to_array()
    for index in range(quadrics.shape[0]):
        ax.contour(xs, ys, values[..., index], levels=[0.0], **style)


def style_axis(ax, title: str) -> None:
    ax.set_xlim(BOX[0], BOX[1])
    ax.set_ylim(BOX[2], BOX[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)


def draw_hyperbolic_plane(
    C: Polarity, vertices: Point, sides: Line, angles: Scalar, area: Scalar, P: Point, foot: Point,
    normal: Line, reflected: Point, pole: Point, circles: Polarity, centres: Point,
) -> plt.Figure:
    """The triangle with its perpendicular on the left, the circles on the right."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6), dpi=120)
    draw_level_sets(left, C, colors="black", linewidths=2.0)
    draw_lines(left, sides, color="#2563eb", linewidth=2.0)
    draw_lines(left, normal, color="#dc2626", linewidth=1.6)
    draw_points(left, vertices, color="#1d4ed8", s=40)
    draw_points(left, P, color="#dc2626", s=50, label="P")
    draw_points(left, foot, color="#b91c1c", marker="s", s=40, label="foot")
    draw_points(left, reflected, color="#f97316", s=45, label="reflection")
    draw_points(left, pole, color="#a855f7", marker="D", s=45, label="pole of BC")
    for xy, theta in zip(euclidean(vertices), angles.to_array()):
        left.annotate(f"{np.degrees(theta):.1f}°", xy, textcoords="offset points", xytext=(6, 6), fontsize=9)
    style_axis(left, f"Triangle area {area.to_array():.3f} by Gauss-Bonnet; the perpendicular runs through the pole")
    left.legend(loc="lower left", fontsize=8)

    draw_level_sets(right, C, colors="black", linewidths=2.0)
    for family, colour in zip(circles, ("#3b82f6", "#f97316")):
        draw_level_sets(right, family, colors=colour, linewidths=1.4)
    draw_points(right, centres, color="#111827", s=35)
    style_axis(right, "Circles as level sets of a quadric")

    fig.tight_layout()
    return fig
