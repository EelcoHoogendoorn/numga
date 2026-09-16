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
# 3. Numeric Escapes
# ---------------------------------------------------------------------------
def arccosh(value: Extensor) -> np.ndarray:
    """A hyperbolic Cayley measure from its invariant, the only place a distance leaves the algebra."""
    return np.arccosh(np.maximum(value.kernel[..., 0], 1.0))


def arccos(value: Extensor) -> np.ndarray:
    """An elliptic Cayley measure, or an angle, from its invariant."""
    return np.arccos(np.clip(value.kernel[..., 0], -1.0, 1.0))


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
