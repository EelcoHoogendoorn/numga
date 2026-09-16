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
Vector = STA.gatype(V)
SpatialVector = STA.gatype(Spatial)
Constitutive = STA.gatype((B, B))               # excitation bivector G <= field bivector F
Permittivity = STA.gatype((V, V))               # D <= E, a symmetric map on spatial vectors

t, x, y, z = mv.vector(np.eye(4))


# ---------------------------------------------------------------------------
# 2. Dispersion Scan
# ---------------------------------------------------------------------------
def dispersion(chi: Constitutive, direction: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    """Smallest singular value of the wave map a -> k . chi(k ^ a) at each phase speed.

    k = (speed, direction) is the wave vector; a solution of the source-free Maxwell equation
    exists where the map loses rank, so the returned curve touches zero at the allowed speeds.
    """
    k: Vector = mv.scalar(speeds[..., None]) * t + mv(Spatial, direction)
    wave = k.commutator(chi(k.wedge(Spatial)))
    return np.linalg.svd(wave.kernel, compute_uv=False)[..., -1]


def phase_speeds(chi: Constitutive, direction: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    """The phase speeds where the dispersion curve dips to zero."""
    curve = dispersion(chi, direction, speeds)
    interior = (curve[1:-1] <= curve[:-2]) & (curve[1:-1] <= curve[2:]) & (curve[1:-1] < 2e-3)
    return speeds[1:-1][interior]


def polarisation(chi: Constitutive, direction: np.ndarray, speed: float) -> SpatialVector:
    """The polarisation of the wave at a phase speed: the wave map's null vector, as a spatial vector."""
    k = mv.vector(np.array([speed, *direction]))
    wave = k.commutator(chi(k.wedge(Spatial)))
    _, _, Vt = np.linalg.svd(wave.kernel)
    return mv(Spatial, Vt[-1])


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
