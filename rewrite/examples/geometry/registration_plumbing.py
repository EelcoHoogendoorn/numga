"""Sample generators, the eigensolve, and rendering for the orientation estimation example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from numga.algebra import Algebra
from numga.gatype.traits import Versor

# ---------------------------------------------------------------------------
# 1. Euclidean 3D Setup
# ---------------------------------------------------------------------------
ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector

# Whole-extensor types (GATypes):
Vector = ga.gatype.vector()
Rotor = ga.gatype.rotor()
Scalar = ga.gatype.scalar()
Scale = Scalar.with_traits(Versor)
Alignment = ga.gatype((ga.subspace.scalar(), ga.subspace.even(), ga.subspace.even()))


# ---------------------------------------------------------------------------
# 2. Samples and Numerics
# ---------------------------------------------------------------------------
def cloud(n: int, rng: np.random.Generator) -> Vector:
    """A Gaussian cloud of n points, stretched so its shape has three distinct axes."""
    return mv.vector(rng.normal(size=(n, 3)) * np.array([2.0, 1.0, 0.5]))


def jitter(points: Vector, sigma: float, rng: np.random.Generator) -> Vector:
    """Add isotropic Gaussian noise to points."""
    return mv.vector(points.kernel + rng.normal(scale=sigma, size=points.kernel.shape))


def scale_root(value: Scalar) -> Scale:
    """Construct the positive scalar factor of a similarity from its squared scale."""
    return ctx.extensor(Scale, np.sqrt(value.kernel))


def same_rotor(a: Rotor, b: Rotor, atol: float) -> bool:
    """Whether two rotors agree up to the overall sign."""
    return np.allclose(a.kernel, b.kernel, atol=atol) or np.allclose(a.kernel, -b.kernel, atol=atol)


# ---------------------------------------------------------------------------
# 3. Rendering
# ---------------------------------------------------------------------------
def render_registration(ax, source: Vector, target: Vector, aligned: Vector, title: str) -> None:
    """Draw source, target and aligned source clouds, with a segment per correspondence."""
    s, t, a = source.kernel, target.kernel, aligned.kernel
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], color="#94a3b8", s=10, label="source")
    ax.scatter(t[:, 0], t[:, 1], t[:, 2], color="#0284c7", s=18, marker="x", label="target")
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], color="#f43f5e", s=10, label="aligned source")
    for p, q in zip(a, t):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#f43f5e", linewidth=0.5, alpha=0.6)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", fontsize=8)


def new_figure() -> tuple[plt.Figure, list]:
    """Two 3D panels side by side."""
    fig = plt.figure(figsize=(11, 5), dpi=120)
    return fig, [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]


def draw_registration(source, target, moved, estimate, similarity, translation, plot_path) -> plt.Figure:
    fig, axes = new_figure()
    render_registration(axes[0], source, target, estimate >> source, "Rotation")
    render_registration(axes[1], source, moved, (similarity >> source) + translation, "Rotation, scale and translation")
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig
