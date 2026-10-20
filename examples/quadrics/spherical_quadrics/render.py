"""Three views of the spherical conic: its foci, its tangent great circles, its cone and polhodes."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from numga import stack
from examples.quadrics.spherical_quadrics.core import Plane, Point, Scalar, ga, mv


# --- plumbing -------------------------------------------------------------------------
def xyz(points: Point) -> np.ndarray:
    """Coordinates (..., 3) of points: their pairings with the planes x, y and z."""
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def normals(planes: Plane) -> np.ndarray:
    """Normals (..., 3) of planes on the basis planes x, y and z."""
    return planes.cast(ga.subspace("x y z")).kernel


def style(ax) -> None:
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])


def draw_spherical_conic(
    curve: Point, foci: Point, arcs: Point, sample: Point, tangents: Plane,
    touching: Plane, circles: Point, surface: Point, grid: Point, potential: Scalar,
) -> plt.Figure:
    """The oval with its foci and the geodesics from them to one point, over the polarity's form; the oval
    as the envelope of its tangent great circles, with their poles; and the cone through it, with the
    level sets of the form, the polhodes."""
    fig = plt.figure(figsize=(18, 6), dpi=140)
    pk, fk, ak, sk = xyz(curve), xyz(foci), xyz(arcs), xyz(sample)
    sx, sy, sz = np.moveaxis(xyz(grid), -1, 0)
    level = potential.to_array()

    # Primal spherical oval, its antipodal loop, the foci and geodesics to one point.
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(sx, sy, sz, facecolors=cm.coolwarm((level - level.min()) / (level.max() - level.min())),
                     alpha=0.35, rstride=2, cstride=2, shade=False)
    ax1.plot(*pk.T, color="gold", linewidth=3.5)
    ax1.plot(*(-pk).T, color="gold", linewidth=2.0, linestyle="--", alpha=0.7)
    ax1.scatter(*fk.T, color="red", s=70, zorder=6)
    ax1.plot(*ak[0].T, color="lime", linewidth=2.2)
    ax1.plot(*ak[1].T, color="cyan", linewidth=2.2)
    ax1.scatter(*sk, color="white", edgecolor="black", s=80, zorder=7)
    style(ax1)

    # Plane-based dual view: the oval as the envelope of its tangent great circles.
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_wireframe(sx, sy, sz, color="slategray", alpha=0.15, linewidth=0.5)
    ax2.plot(*pk.T, color="gold", linewidth=3.0)
    ck = xyz(circles)
    colors = cm.plasma(np.linspace(0.1, 0.9, len(ck)))
    for circle, pole, color in zip(ck, normals(touching), colors):
        ax2.plot(*circle.T, color=color, alpha=0.55, linewidth=1.2)
        ax2.quiver(*circle[0], *(0.35 * pole), color=color, arrow_length_ratio=0.3, alpha=0.8, linewidth=1.0)
    ax2.plot(*normals(tangents).T, color="magenta", linewidth=2.0, linestyle=":")
    style(ax2)

    # The quadratic cone through the oval, and the polhodes: level sets of P & C(P).
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_wireframe(sx, sy, sz, color="slategray", alpha=0.15, linewidth=0.5)
    ax3.plot_surface(*np.moveaxis(xyz(surface), -1, 0), color="khaki", alpha=0.3, rstride=2, cstride=2, shade=True)
    ax3.plot(*pk.T, color="gold", linewidth=3.5)
    fractions = np.linspace(0.0, 0.7, 4)[1:]
    for levels, color in ((level.min() * fractions[::-1], "salmon"), (level.max() * fractions, "deepskyblue")):
        ax3.contour(sx, sy, sz, level, levels=levels, colors=[color], linewidths=1.2, alpha=0.7)
    style(ax3)

    fig.tight_layout()
    return fig
