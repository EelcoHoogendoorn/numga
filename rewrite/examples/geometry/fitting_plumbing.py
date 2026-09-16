"""Sample generators, drawing frames, and rendering for the fitting example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from numga import Extensor, NumpyContext
from numga.algebras import PGA3D

# ---------------------------------------------------------------------------
# 1. PGA3D Setup
# ---------------------------------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

# Whole-extensor types (GATypes):
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()


# ---------------------------------------------------------------------------
# 2. Points and Samples
# ---------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Construct affine points (x, y, z, 1) from an (..., 3) coordinate array."""
    return mv.antivector(np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1))


def euclidean(points: Point) -> np.ndarray:
    """Dehomogenize points to (..., 3) coordinates, cast into the layout Point declares."""
    k = points.cast(Point.output_subspace).kernel
    return k[..., :3] / k[..., 3:]


origin: Point = point(np.zeros(3))


def cloud(n: int, spread: float, rng: np.random.Generator) -> Point:
    """Gaussian cloud of n points around the origin."""
    return point(rng.normal(scale=spread, size=(n, 3)))


def segment(n: int, half_length: float) -> Point:
    """n points evenly spaced along the y axis through the origin."""
    t = np.linspace(-half_length, half_length, n)
    return point(np.stack([np.zeros_like(t), t, np.zeros_like(t)], axis=-1))


def patch(n: int, half_width: float, rng: np.random.Generator) -> Point:
    """n points uniformly scattered over a square of the plane z = 0."""
    uv = rng.uniform(-half_width, half_width, size=(n, 2))
    return point(np.concatenate([uv, np.zeros_like(uv[:, :1])], axis=-1))


def jitter(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    """Add isotropic Gaussian noise to the Euclidean coordinates of points."""
    return point(euclidean(points) + rng.normal(scale=sigma, size=(*points.shape, 3)))


def bundle(n: int, spread: float, rng: np.random.Generator) -> Line:
    """n unit-direction lines in random directions, each passing within about spread of the origin."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    feet = point(rng.normal(scale=spread, size=(n, 3)))
    return feet.regressive(mv.antivector(np.concatenate([directions, np.zeros((n, 1))], axis=-1)))


def line_caps(half_length: float) -> Plane:
    """The two planes y = ±half_length, for clipping a line near the y axis when drawing."""
    return mv.vector(np.array([[0.0, 1.0, 0.0, -half_length], [0.0, 1.0, 0.0, half_length]]))


def patch_edges(half_width: float) -> Line:
    """Four lines parallel to z through the corners of the square |x|, |y| <= half_width."""
    x_planes = mv.vector(np.array([[1.0, 0.0, 0.0, -s * half_width] for s in (1, 1, -1, -1)]))
    y_planes = mv.vector(np.array([[0.0, 1.0, 0.0, -s * half_width] for s in (1, -1, -1, 1)]))
    return x_planes.wedge(y_planes)


# ---------------------------------------------------------------------------
# 3. Numerics, Checks and Rendering
# ---------------------------------------------------------------------------
def smallest_eigenvector(form: Extensor) -> Extensor:
    """The unit element of an arity-2 form's slot type that minimises the form.

    Unit means the slot type's own reverse product. Where that product is degenerate the
    generalized eigenproblem has infinite eigenvalues on the free coefficients, and the
    smallest finite eigenvalue is the constrained minimum.
    """
    slot = form.gatype.input_subspaces[0]
    unit = ctx.lower(ga.operator.reverse(slot) | ga.gatype(slot))
    N = form.kernel.squeeze()
    N = 0.5 * (N + N.T)
    M = np.asarray(unit.kernel).squeeze()
    w, V = scipy.linalg.eig(N, M)
    w = np.where(np.isfinite(w), w.real, np.inf)
    return mv(slot, V[:, np.argmin(w)].real)


def same_element(a: Extensor, b: Extensor, atol: float) -> bool:
    """Whether two nullary extensors agree as projective elements, up to scale and sign."""
    ka, kb = a.kernel, b.cast(a.gatype.output_subspace).kernel
    ka, kb = ka / np.abs(ka).max(), kb / np.abs(kb).max()
    return np.allclose(ka, kb, atol=atol) or np.allclose(ka, -kb, atol=atol)


def render_point_fit(ax, data: Point, truth: Point, fit: Point) -> None:
    """Draw the noisy cloud with the true and fitted centre."""
    xyz = euclidean(data)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#94a3b8", s=6, alpha=0.5)
    ax.scatter(*euclidean(truth), color="#0284c7", s=90, marker="x", label="truth")
    ax.scatter(*euclidean(fit), color="#f43f5e", s=50, label="fit")
    ax.set_title("Point: minimise |P ∨ X|²")


def render_line_fit(ax, data: Point, truth: Line, fit: Line, caps: Plane) -> None:
    """Draw the noisy points with the true and fitted line, clipped by two cap planes."""
    xyz = euclidean(data)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#94a3b8", s=6, alpha=0.5)
    for line, color, style, label in ((truth, "#0284c7", "--", "truth"), (fit, "#f43f5e", "-", "fit")):
        ends = euclidean(line.wedge(caps))
        ax.plot(ends[:, 0], ends[:, 1], ends[:, 2], color=color, linestyle=style, linewidth=2.0, label=label)
    ax.set_title("Line: minimise |P ∨ L|²")


def render_plane_fit(ax, data: Point, truth: Plane, fit: Plane, edges: Line) -> None:
    """Draw the noisy points with the true and fitted plane, as quads cut by four edge lines."""
    xyz = euclidean(data)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#94a3b8", s=6, alpha=0.5)
    for plane, color, style, label in ((truth, "#0284c7", "--", "truth"), (fit, "#f43f5e", "-", "fit")):
        quad = euclidean(plane.wedge(edges))
        loop = np.vstack([quad, quad[:1]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linestyle=style, linewidth=2.0, label=label)
    ax.set_title("Plane: minimise |P ∨ π|²")


def render_bundle_fit(ax, rays: Line, truth: Point, fit: Point, half_length: float) -> None:
    """Draw the bundle as segments through the fitted point along each line's direction."""
    directions = rays.wedge(mv.w)                       # each line's point at infinity
    unit = mv.scalar(1.0 / np.linalg.norm(directions.kernel, axis=-1, keepdims=True))
    for sign in (-1.0, 1.0):
        end = euclidean(fit + directions * unit * (sign * half_length))
        start = np.broadcast_to(euclidean(fit), end.shape)
        for a, b in zip(start, end):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#94a3b8", linewidth=0.8, alpha=0.7)
    ax.scatter(*euclidean(truth), color="#0284c7", s=90, marker="x", label="truth")
    ax.scatter(*euclidean(fit), color="#f43f5e", s=50, label="fit")
    ax.set_title("Point to lines: minimise |L ∨ X|²")


def new_figure() -> tuple[plt.Figure, list]:
    """One row of four 3D panels with a shared look."""
    fig = plt.figure(figsize=(20, 5), dpi=120)
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
    for ax in axes:
        ax.set_box_aspect([1, 1, 1])
    return fig, axes
