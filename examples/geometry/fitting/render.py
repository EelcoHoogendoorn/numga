"""Drawing for the fitting example: read the geometry out as coordinates and plot it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.fitting.core import Point, ga


def euclidean(points: Point) -> np.ndarray:
    values = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def new_axes() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(7, 6), dpi=120, layout="constrained")
    return fig, fig.add_subplot(projection="3d")


def finish(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", fontsize=8)


def scatter_samples(ax: plt.Axes, points: Point) -> None:
    ax.scatter(*euclidean(points).T, color="#94a3b8", s=6, alpha=0.5)


def draw_point_fit(points: Point, truth: Point, fit: Point) -> plt.Figure:
    """The noisy cloud with the true and fitted centre."""
    fig, ax = new_axes()
    scatter_samples(ax, points)
    ax.scatter(*euclidean(truth), color="#0284c7", s=90, marker="x", label="truth")
    ax.scatter(*euclidean(fit), color="#f43f5e", s=50, label="fit")
    finish(ax, "Point: minimise |P ∨ X|²")
    return fig


def draw_line_fit(points: Point, truth_ends: Point, fit_ends: Point) -> plt.Figure:
    """The noisy points with the true and fitted line, each drawn between its two end points."""
    fig, ax = new_axes()
    scatter_samples(ax, points)
    ax.plot(*euclidean(truth_ends).T, color="#0284c7", linestyle="--", linewidth=2.0, label="truth")
    ax.plot(*euclidean(fit_ends).T, color="#f43f5e", linestyle="-", linewidth=2.0, label="fit")
    finish(ax, "Line: minimise |P ∨ L|²")
    return fig


def draw_plane_fit(points: Point, truth_quad: Point, fit_quad: Point) -> plt.Figure:
    """The noisy points with the true and fitted plane, each drawn as a closed quad."""
    fig, ax = new_axes()
    scatter_samples(ax, points)
    for quad, color, style, label in ((truth_quad, "#0284c7", "--", "truth"), (fit_quad, "#f43f5e", "-", "fit")):
        corners = euclidean(quad)
        loop = np.vstack([corners, corners[:1]])
        ax.plot(*loop.T, color=color, linestyle=style, linewidth=2.0, label=label)
    finish(ax, "Plane: minimise |P ∨ π|²")
    return fig


def draw_bundle_fit(ends: Point, truth: Point, fit: Point) -> plt.Figure:
    """The bundle as segments between paired end points, with the true and fitted point."""
    fig, ax = new_axes()
    for a, b in zip(*euclidean(ends)):
        ax.plot(*np.stack([a, b]).T, color="#94a3b8", linewidth=0.8, alpha=0.7)
    ax.scatter(*euclidean(truth), color="#0284c7", s=90, marker="x", label="truth")
    ax.scatter(*euclidean(fit), color="#f43f5e", s=50, label="fit")
    finish(ax, "Point to lines: minimise |L ∨ X|²")
    return fig
