"""Constructors and drawing primitives for the Cayley-Klein example in PGA2D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA2D

# ---------------------------------------------------------------------------
# 1. PGA2D Setup
# ---------------------------------------------------------------------------
ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector

# Whole-extensor types (GATypes); map types read output <= inputs:
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Line = ga.gatype.vector()
Polarity = ga.gatype((Line.output_subspace, Point.output_subspace))   # polar line <= point
Pole = ga.gatype((Point.output_subspace, Line.output_subspace))       # pole <= line


# ---------------------------------------------------------------------------
# 2. Points
# ---------------------------------------------------------------------------
def point(xy: np.ndarray) -> Point:
    """Construct affine points (x, y, 1) from an (..., 2) coordinate array."""
    return mv.antivector(np.concatenate([xy, np.ones_like(xy[..., :1])], axis=-1))


def euclidean(points: Point) -> np.ndarray:
    """Dehomogenize points to (..., 2) chart coordinates, cast into the layout Point declares."""
    k = points.cast(Point.output_subspace).kernel
    return k[..., :2] / k[..., 2:]


def grid(box: tuple[float, float, float, float], n: int) -> Point:
    """An n x n grid of points covering the box (x0, x1, y0, y1)."""
    xs = np.linspace(box[0], box[1], n)
    ys = np.linspace(box[2], box[3], n)
    return point(np.stack(np.meshgrid(xs, ys), axis=-1))


# ---------------------------------------------------------------------------
# 4. Drawing
# ---------------------------------------------------------------------------
def draw_points(ax, points: Point, **style) -> None:
    xy = np.atleast_2d(euclidean(points))
    ax.scatter(xy[:, 0], xy[:, 1], zorder=6, **style)


def draw_line(ax, line: Line, box: tuple[float, float, float, float], **style) -> None:
    """Draw the part of a line inside the box: meet it with the four edges, keep the hits on the box."""
    x0, x1, y0, y1 = box
    edges = mv.vector(np.array([[1.0, 0.0, -x0], [1.0, 0.0, -x1], [0.0, 1.0, -y0], [0.0, 1.0, -y1]]))
    hits = line.wedge(edges).cast(Point.output_subspace).kernel
    hits = hits[np.abs(hits[:, 2]) > 1e-12]
    xy = hits[:, :2] / hits[:, 2:]
    inside = (xy[:, 0] >= x0 - 1e-9) & (xy[:, 0] <= x1 + 1e-9) & (xy[:, 1] >= y0 - 1e-9) & (xy[:, 1] <= y1 + 1e-9)
    xy = xy[inside]
    if len(xy) >= 2:
        ax.plot(xy[:2, 0], xy[:2, 1], **style)


def draw_level_set(ax, quadric: Polarity, box: tuple[float, float, float, float], n: int = 400, **style) -> None:
    """Draw the locus P ∨ quadric(P) = 0 by contouring it on a grid."""
    samples = grid(box, n)
    values = samples.regressive(quadric(samples)).kernel[..., 0]
    xs = np.linspace(box[0], box[1], n)
    ys = np.linspace(box[2], box[3], n)
    ax.contour(xs, ys, values, levels=[0.0], **style)


def style_axis(ax, title: str, box: tuple[float, float, float, float]) -> None:
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)


def new_figure() -> tuple[plt.Figure, np.ndarray]:
    return plt.subplots(1, 2, figsize=(13, 6), dpi=120)


BOX = (-1.2, 1.9, -1.2, 1.6)


def draw_geometry(C, to_next, perpendicular, vertices, P, foot, reflected, pole, angles, area, circles, centres, plot_path) -> plt.Figure:
    angles, area = angles.kernel[..., 0], area.kernel.item()
    fig, (left, right) = new_figure()
    draw_level_set(left, C, BOX, colors="black", linewidths=2.0)
    for line in (to_next[0], to_next[1], to_next[2]):
        draw_line(left, line, BOX, color="#2563eb", linewidth=2.0)
    draw_line(left, perpendicular, BOX, color="#dc2626", linewidth=1.6)
    draw_points(left, vertices, color="#1d4ed8", s=40)
    draw_points(left, P, color="#dc2626", s=50, label="P")
    draw_points(left, foot, color="#b91c1c", marker="s", s=40, label="foot")
    draw_points(left, reflected, color="#f97316", s=45, label="reflection")
    draw_points(left, pole, color="#a855f7", marker="D", s=45, label="pole of BC")
    for vertex, theta in zip(vertices, angles):
        left.annotate(f"{np.degrees(theta):.1f}°", euclidean(vertex), textcoords="offset points", xytext=(6, 6), fontsize=9)
    style_axis(left, f"Triangle area {area:.3f} by Gauss-Bonnet; the perpendicular runs through the pole", BOX)
    left.legend(loc="lower left", fontsize=8)

    draw_level_set(right, C, BOX, colors="black", linewidths=2.0)
    for family, colour in zip(circles, ("#3b82f6", "#f97316")):
        for k in range(family.shape[0]):
            draw_level_set(right, family[k], BOX, colors=colour, linewidths=1.4)
    draw_points(right, centres, color="#111827", s=35)
    style_axis(right, "Circles as level sets of a quadric", BOX)

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig
