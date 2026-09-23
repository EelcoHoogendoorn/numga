"""Three views of the spherical conic: its foci, its tangent great circles, its cone and polhodes."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from examples.quadrics.spherical_quadrics.core import Plane, Point, Scalar, ga


def xyz(points: Point) -> np.ndarray:
    """Coordinates (..., 3) of points on the basis points yz, zx and xy."""
    return points.cast(ga.subspace("yz zx xy")).kernel


def normals(planes: Plane) -> np.ndarray:
    """Normals (..., 3) of planes on the basis planes x, y and z."""
    return planes.cast(ga.subspace("x y z")).kernel


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_box_aspect([1, 1, 1])


def draw_spherical_conic(
    curve: Point, foci: Point, arcs: Point, sample: Point, theta_a: float, tangents: Plane,
    touching: Plane, circles: Point, surface: Point, grid: Point, potential: Scalar,
) -> plt.Figure:
    fig = plt.figure(figsize=(18, 6), dpi=140)
    pk, fk, ak, sk = xyz(curve), xyz(foci), xyz(arcs), xyz(sample)
    sx, sy, sz = np.moveaxis(xyz(grid), -1, 0)
    level = potential.to_array()

    # Primal spherical oval, its antipodal loop, the foci and geodesics to one point.
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(sx, sy, sz, facecolors=cm.coolwarm((level - level.min()) / (level.max() - level.min())),
                     alpha=0.35, rstride=2, cstride=2, shade=False)
    ax1.plot(*pk.T, color="gold", linewidth=3.5, label=r"Spherical Oval $p \vee C(p) = 0$")
    ax1.plot(*(-pk).T, color="gold", linewidth=2.0, linestyle="--", alpha=0.7, label=r"Antipodal loop $-p$")
    ax1.scatter(*fk.T, color="red", s=70, zorder=6, label=r"Foci $F_1, F_2$")
    ax1.plot(*ak[0].T, color="lime", linewidth=2.2, label=r"Geodesic $d(P, F_1)$")
    ax1.plot(*ak[1].T, color="cyan", linewidth=2.2, label=r"Geodesic $d(P, F_2)$")
    ax1.scatter(*sk, color="white", edgecolor="black", s=80, zorder=7, label=r"Sample point $P$")
    style(ax1, f"Primal Point Locus on $S^2$\n$d(P, F_1) + d(P, F_2) = {2 * theta_a:.3f}$ rad (const)")

    # Plane-based dual view: the oval as the envelope of its tangent great circles.
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_wireframe(sx, sy, sz, color="slategray", alpha=0.15, linewidth=0.5)
    ax2.plot(*pk.T, color="gold", linewidth=3.0, label="Spherical Oval (Envelope)")
    ck = xyz(circles)
    colors = cm.plasma(np.linspace(0.1, 0.9, len(ck)))
    for circle, pole, color in zip(ck, normals(touching), colors):
        ax2.plot(*circle.T, color=color, alpha=0.55, linewidth=1.2)
        ax2.quiver(*circle[0], *(0.35 * pole), color=color, arrow_length_ratio=0.3, alpha=0.8, linewidth=1.0)
    ax2.plot(*normals(tangents).T, color="magenta", linewidth=2.0, linestyle=":", label=r"Dual Plane Conic $\pi \in S^2$")
    style(ax2, r"Plane-Based Dual View" "\n" r"Envelope of Tangent Great Circles $\pi \cdot x = 0$")

    # The quadratic cone through the oval, and the polhodes: level sets P ∨ C(P) = c.
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_wireframe(sx, sy, sz, color="slategray", alpha=0.15, linewidth=0.5)
    ax3.plot_surface(*np.moveaxis(xyz(surface), -1, 0), color="khaki", alpha=0.3, rstride=2, cstride=2, shade=True)
    ax3.plot(*pk.T, color="gold", linewidth=3.5, label="Cone-Sphere Intersection")
    fractions = np.linspace(0.0, 0.7, 4)[1:]
    for levels, color in ((level.min() * fractions[::-1], "salmon"), (level.max() * fractions, "deepskyblue")):
        ax3.contour(sx, sy, sz, level, levels=levels, colors=[color], linewidths=1.2, alpha=0.7)
    style(ax3, r"3D Quadratic Cone & Confocal Polhodes" "\n" r"$Q(v) \cdot v = c$ on the Momentum Sphere")

    fig.tight_layout()
    return fig
