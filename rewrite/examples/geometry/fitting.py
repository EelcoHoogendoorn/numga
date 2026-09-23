"""Least-squares fitting of PGA3D primitives to points, in one pattern.

A point, a line and a plane are fitted to noisy samples with the same three lines: the
join of the samples with the unknown left open, that residual squared and summed into a
quadratic form, and its smallest unit eigenvector. Unit is the unknown's own reverse
product, which is degenerate exactly on the coefficients least squares leaves free. The
roles swap freely: a point fitted to a bundle of lines is their point of closest approach.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, GAType, NumpyContext
from numga.algebras import PGA3D

from examples import PLOT_DIR


# --- scenario algebra ------------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Plane = ga.gatype.vector()
Scalar = ga.gatype.scalar()


# --- math ------------------------------------------------------------------
def fit(Unknown: GAType, samples: Extensor) -> Extensor:
    """Fit a PGA primitive to a batch of points or lines by squared join distance.

    Unknown is the point, line or plane slot being solved for.
    """
    samples = samples.normalized()
    # Joining each sample with the open unknown measures its incidence error:
    # point & plane is a scalar, point & line a plane, point & point a line.
    residual = samples & Unknown
    misfit = (residual.reverse() | residual).sum(axis=0)

    # The primitive's reverse product fixes its geometric size, leaving its
    # position free: plane normal, line direction, or point weight has unit norm.
    norm = (mv.rotor() >> Unknown).reverse() | Unknown
    values, modes = misfit.eig(norm)
    return smallest_finite(values, modes)


# --- plumbing: finite eigenmodes and sample construction --------------------
def smallest_finite(values: Scalar, modes: Extensor) -> Extensor:
    """Select the real mode with the smallest finite generalized eigenvalue.

    The norm is degenerate on ideal components, so some eigenvalues are infinite.
    """
    eigenvalues = values.to_array()
    index = np.argmin(np.where(values.isfinite(), eigenvalues.real, np.inf))
    return mv(modes.output_subspace, modes[index].kernel.real)


def point(xyz: np.ndarray) -> Point:
    return mv.yzw * xyz[..., 0] + mv.zxw * xyz[..., 1] + mv.xyw * xyz[..., 2] + mv.zyx


def euclidean(points: Point) -> np.ndarray:
    values = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def cloud(n: int, spread: float, rng: np.random.Generator) -> Point:
    return point(rng.normal(scale=spread, size=(n, 3)))


def segment(n: int, half_length: float) -> Point:
    """Points along the y axis."""
    return mv.zyx + mv.zxw * np.linspace(-half_length, half_length, n)


def patch(n: int, half_width: float, rng: np.random.Generator) -> Point:
    """Points on a square of the plane z = 0."""
    uv = rng.uniform(-half_width, half_width, size=(n, 2))
    return mv.zyx + mv.yzw * uv[:, 0] + mv.zxw * uv[:, 1]


def jitter(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    return point(euclidean(points) + rng.normal(scale=sigma, size=(*points.shape, 3)))


def bundle(n: int, spread: float, rng: np.random.Generator) -> Line:
    """Lines with unit directions, passing near the origin."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    feet = point(rng.normal(scale=spread, size=(n, 3)))
    ideal = mv.yzw * directions[:, 0] + mv.zxw * directions[:, 1] + mv.xyw * directions[:, 2]
    return feet & ideal


# --- plotting --------------------------------------------------------------
def line_caps(half_length: float) -> Plane:
    return mv.y - mv.w * np.array([half_length, -half_length])


def patch_edges(half_width: float) -> Line:
    x_planes = mv.x - mv.w * (np.array([1, 1, -1, -1]) * half_width)
    y_planes = mv.y - mv.w * (np.array([1, -1, -1, 1]) * half_width)
    return x_planes ^ y_planes


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(7, 6), dpi=120, layout="constrained")
    return fig, fig.add_subplot(projection="3d")


def save_figure(fig: plt.Figure, ax: plt.Axes, path: Path) -> plt.Figure:
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Figure saved to {path}")
    return fig


def draw_point_fit(data: Point, truth: Point, fit: Point, path: Path) -> plt.Figure:
    """Draw the noisy cloud with the true and fitted centre."""
    fig, ax = new_figure()
    xyz = euclidean(data)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#94a3b8", s=6, alpha=0.5)
    ax.scatter(*euclidean(truth), color="#0284c7", s=90, marker="x", label="truth")
    ax.scatter(*euclidean(fit), color="#f43f5e", s=50, label="fit")
    ax.set_title("Point: minimise |P ∨ X|²")
    return save_figure(fig, ax, path)


def draw_line_fit(data: Point, truth: Line, fit: Line, caps: Plane, path: Path) -> plt.Figure:
    """Draw the noisy points with the true and fitted line, clipped by two cap planes."""
    fig, ax = new_figure()
    xyz = euclidean(data)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#94a3b8", s=6, alpha=0.5)
    for line, color, style, label in ((truth, "#0284c7", "--", "truth"), (fit, "#f43f5e", "-", "fit")):
        ends = euclidean(line.wedge(caps))
        ax.plot(ends[:, 0], ends[:, 1], ends[:, 2], color=color, linestyle=style, linewidth=2.0, label=label)
    ax.set_title("Line: minimise |P ∨ L|²")
    return save_figure(fig, ax, path)


def draw_plane_fit(data: Point, truth: Plane, fit: Plane, edges: Line, path: Path) -> plt.Figure:
    """Draw the noisy points with the true and fitted plane, as quads cut by four edge lines."""
    fig, ax = new_figure()
    xyz = euclidean(data)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#94a3b8", s=6, alpha=0.5)
    for plane, color, style, label in ((truth, "#0284c7", "--", "truth"), (fit, "#f43f5e", "-", "fit")):
        quad = euclidean(plane.wedge(edges))
        loop = np.vstack([quad, quad[:1]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linestyle=style, linewidth=2.0, label=label)
    ax.set_title("Plane: minimise |P ∨ π|²")
    return save_figure(fig, ax, path)


def draw_bundle_fit(rays: Line, truth: Point, fit: Point, half_length: float, path: Path) -> plt.Figure:
    """Draw the bundle as segments through the fitted point along each line's direction."""
    fig, ax = new_figure()
    directions = rays.wedge(mv.w)                       # each line's point at infinity
    # An ideal point has no weight to normalize; its length is the norm of its dual, a direction.
    unit_directions = directions / directions.dual().norm()
    for sign in (-1.0, 1.0):
        end = euclidean(fit + unit_directions * (sign * half_length))
        start = np.broadcast_to(euclidean(fit), end.shape)
        for a, b in zip(start, end):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#94a3b8", linewidth=0.8, alpha=0.7)
    ax.scatter(*euclidean(truth), color="#0284c7", s=90, marker="x", label="truth")
    ax.scatter(*euclidean(fit), color="#f43f5e", s=50, label="fit")
    ax.set_title("Point to lines: minimise |L ∨ X|²")
    return save_figure(fig, ax, path)


# --- scenarios -------------------------------------------------------------
def point_to_points() -> plt.Figure:
    """Fit the centroid of a noisy point cloud."""
    rng = np.random.default_rng(0)
    pose = (mv.xw * 0.4 - mv.yw * 0.3 + mv.zw * 0.6).exp()
    points = jitter(pose >> cloud(200, 0.5, rng), 0.05, rng)
    centroid = fit(Point, points)
    return draw_point_fit(points, pose >> mv.zyx, centroid,
                          PLOT_DIR / "fitting_point_to_points.png")


def line_to_points() -> plt.Figure:
    """Fit the principal line through a noisy segment."""
    rng = np.random.default_rng(0)
    pose = (mv.xw * 0.4 - mv.yw * 0.3 + mv.zw * 0.6).exp() * (mv.yz * 0.3).exp() * (mv.xy * 0.5).exp()
    points = jitter(pose >> segment(120, 2.0), 0.05, rng)
    line = fit(Line, points)
    return draw_line_fit(points, pose >> mv.xz, line, pose >> line_caps(2.5),
                         PLOT_DIR / "fitting_line_to_points.png")


def plane_to_points() -> plt.Figure:
    """Fit a plane through a noisy patch."""
    rng = np.random.default_rng(0)
    pose = (mv.xw * 0.4 - mv.yw * 0.3 + mv.zw * 0.6).exp() * (mv.yz * 0.3).exp() * (mv.xy * 0.5).exp()
    points = jitter(pose >> patch(200, 2.0, rng), 0.05, rng)
    plane = fit(Plane, points)
    return draw_plane_fit(points, pose >> mv.z, plane, pose >> patch_edges(2.0),
                          PLOT_DIR / "fitting_plane_to_points.png")


def point_to_lines() -> plt.Figure:
    """Triangulate the point of closest approach to a bundle of lines."""
    rng = np.random.default_rng(0)
    pose = (mv.xw * 0.4 - mv.yw * 0.3 + mv.zw * 0.6).exp() * (mv.yz * 0.3).exp() * (mv.xy * 0.5).exp()
    rays = pose >> bundle(30, 0.05, rng)
    intersection = fit(Point, rays)
    return draw_bundle_fit(rays, pose >> mv.zyx, intersection, 2.0,
                           PLOT_DIR / "fitting_point_to_lines.png")


if __name__ == "__main__":
    point_to_points()
    line_to_points()
    plane_to_points()
    point_to_lines()
    plt.show()
