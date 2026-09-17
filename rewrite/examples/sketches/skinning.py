"""Linear blend skinning two ways in PGA3D: blend the motors, or blend their maps.

A bone's transform is a motor m; its action on points is the extensor m >> P, a 4x4 matrix.
Matrix skinning blends the maps and applies the blend; motor skinning (dual quaternion
blending) blends the motors, renormalises, and applies the sandwich. In this library those
are the same kind of object, so each blend is one line and the well-known artefact of the
matrix version, the collapsing radius under a twist, is a one-number comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

from examples import PLOT_DIR

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
P = ga.subspace.antivector()
Point = ga.gatype.antivector()
Scalar = ga.gatype.scalar()


# --- plumbing -------------------------------------------------------------------------
def cylinder(rings: int, around: int) -> tuple[Point, Scalar]:
    """Unit-radius skin around the x axis from x = 0 to 1, and the weight x of the second bone."""
    x, t = np.meshgrid(np.linspace(0.0, 1.0, rings), np.linspace(0.0, 2 * np.pi, around, endpoint=False), indexing="ij")
    coords = np.stack([x, np.cos(t), np.sin(t), np.ones_like(x)], axis=-1).reshape(-1, 4)
    return mv.antivector(coords), mv.scalar(coords[:, :1])


def radius(points: Point) -> np.ndarray:
    k = points.cast(P).kernel
    return np.linalg.norm(k[..., 1:3] / k[..., 3:], axis=-1)


def surface(ax, points: Point, rings: int, around: int, title: str) -> None:
    """Draw the skinned cylinder as a shaded quad mesh, striped along its length."""
    k = points.cast(P).kernel.reshape(rings, around, 4)
    k = np.concatenate([k, k[:, :1]], axis=1)                       # close each ring
    stripes = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, around + 1))[None].repeat(rings, axis=0)
    ax.plot_surface(k[..., 0], k[..., 1], k[..., 2], facecolors=stripes, edgecolor="black", linewidth=0.2, shade=True)
    ax.set_title(title); ax.set_box_aspect((1, 1, 1)); ax.set_xlim(0, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)


# --- math -----------------------------------------------------------------------------
def draw_skinning(motor_skin, slerp_skin, matrix_skin, rings, around, plot_path) -> plt.Figure:
    fig = plt.figure(figsize=(15, 5), dpi=120)
    surface(fig.add_subplot(1, 3, 1, projection="3d"), motor_skin, rings, around, "motor blend (normalised lerp)")
    surface(fig.add_subplot(1, 3, 2, projection="3d"), slerp_skin, rings, around, "motor blend (slerp)")
    surface(fig.add_subplot(1, 3, 3, projection="3d"), matrix_skin, rings, around, "matrix blend")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")

    return fig


def main(plot_path: str = str(PLOT_DIR / "sketch_skinning.png")) -> plt.Figure:
    rings, around = 12, 24
    skin, w = cylinder(rings, around)
    root = mv.rotor()
    twist = (mv.yz * (np.radians(150.0) / 2)).exp()                     # second bone: 150° about x

    motor_skin = (root + w * (twist - root)).normalized() >> skin
    matrix_skin = ((root >> P) + w * ((twist >> P) - (root >> P)))(skin)
    # Slerp follows the geodesic: the log of the relative motor, scaled by the weight and
    # exponentiated. The normalised lerp is a chord of it, so both are rigid but their twist
    # angles are spaced differently along the blend.
    slerp_skin = ((twist / root).log() * w).exp() * root >> skin

    print(f"min radius, motor blend:  {radius(motor_skin).min():.3f}")
    print(f"min radius, matrix blend: {radius(matrix_skin).min():.3f}  (cos 75° = {np.cos(np.radians(75)):.3f})")

    fig = draw_skinning(motor_skin, slerp_skin, matrix_skin, rings, around, plot_path)

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    np.testing.assert_allclose(radius(motor_skin), 1.0, atol=1e-12)
    np.testing.assert_allclose(radius(slerp_skin), 1.0, atol=1e-12)
    return fig


if __name__ == "__main__":
    main()
