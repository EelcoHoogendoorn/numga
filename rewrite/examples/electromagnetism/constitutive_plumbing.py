"""Setup, the dispersion scan, and rendering for the constitutive extensor example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from numga.algebra import Algebra

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: R_{1,3}, t+ x- y- z-)
# ---------------------------------------------------------------------------
STA = Algebra("t+x-y-z-")
ctx = NumpyContext(STA)
mv = ctx.multivector

V = STA.subspace.vector()
B = STA.subspace.bivector()
Spatial = STA.subspace("x y z")                 # polarisations in the temporal gauge a . t = 0
Scalar = STA.gatype.scalar()
Vector = STA.gatype(V)
SpatialVector = STA.gatype(Spatial)
Constitutive = STA.gatype((B, B))               # excitation bivector G <= field bivector F
Permittivity = STA.gatype((V, V))               # D <= E, a symmetric map on spatial vectors

t, x, y, z = mv.vector(np.eye(4))


# ---------------------------------------------------------------------------
# 2. Dispersion Scan
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3. Rendering
# ---------------------------------------------------------------------------
def render_dispersion(ax, speeds: np.ndarray, curves: dict[str, np.ndarray], expected: dict[str, list[float]]) -> None:
    """Dispersion curves per medium, with the expected phase speeds marked."""
    for name, curve in curves.items():
        line, = ax.semilogy(speeds, curve, label=name, linestyle="--" if "axion" in name else "-")
        for v in expected[name]:
            ax.axvline(v, color=line.get_color(), linestyle=":", linewidth=1.0)
    ax.set_xlabel("Phase speed ω / |k|")
    ax.set_ylabel("Smallest singular value of the wave map")
    ax.set_title("Allowed waves are where k · χ(k ∧ a) = 0 has a solution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    return fig, ax


def minimum_speeds(speeds: Scalar, curve: Scalar) -> Scalar:
    """Select resolved local minima from a sampled dispersion curve."""
    values = curve.kernel[..., 0]
    interior = (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:]) & (values[1:-1] < 2e-3)
    return speeds[1:-1][interior]


def draw_media(speeds: Scalar, curves: Scalar, expected: list[list[float]], plot_path: str) -> plt.Figure:
    """Read scalar curves into the dispersion plot."""
    names = ["glass at rest", "glass with axion term", "crystal along z", "ferrite along z",
             "glass moving with the wave", "glass moving against the wave"]
    fig, ax = new_figure()
    render_dispersion(ax, speeds.kernel[..., 0], dict(zip(names, curves.kernel[..., 0])), dict(zip(names, expected)))
    plt.tight_layout()
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig
