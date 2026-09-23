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
from examples.pga3d import point

# --- scenario algebra -----------------------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
Scalar = ga.gatype.scalar()


# --- math -----------------------------------------------------------------------------
def blend_skin(vertices: Point, weight: Scalar, root: Motor, tip: Motor) -> tuple[Point, Point, Point]:
    """Deform vertices by normalised motor blending, motor slerp, and map blending."""
    # Blend motors and normalise, or blend their point maps directly.
    motor_skin = (root + weight * (tip - root)).normalized() >> vertices
    matrix_skin = ((root >> Point) + weight * ((tip >> Point) - (root >> Point)))(vertices)

    # Slerp follows the geodesic: the log of the relative motor, scaled by the weight and
    # exponentiated. The normalised lerp is a chord of it, so both are rigid but their twist
    # angles are spaced differently along the blend.
    slerp_skin = ((tip / root).log() * weight).exp() * root >> vertices
    return motor_skin, slerp_skin, matrix_skin


# --- plotting -------------------------------------------------------------------------
def xyz(points: Point) -> np.ndarray:
    k = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return k[..., :3] / k[..., 3:]


def radius(points: Point) -> np.ndarray:
    return np.linalg.norm(xyz(points)[..., 1:], axis=-1)


def surface(ax, points: Point, rings: int, around: int, title: str) -> None:
    """Draw the skinned cylinder as a shaded quad mesh, striped along its length."""
    k = xyz(points).reshape(rings, around, 3)
    k = np.concatenate([k, k[:, :1]], axis=1)                       # close each ring
    stripes = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, around + 1))[None].repeat(rings, axis=0)
    ax.plot_surface(k[..., 0], k[..., 1], k[..., 2], facecolors=stripes, edgecolor="black", linewidth=0.2, shade=True)
    ax.set_title(title); ax.set_box_aspect((1, 1, 1)); ax.set_xlim(0, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)


def draw_skinning(motor_skin: Point, slerp_skin: Point, matrix_skin: Point,
                  rings: int, around: int, plot_path: str) -> plt.Figure:
    fig = plt.figure(figsize=(15, 5), dpi=120)
    surface(fig.add_subplot(1, 3, 1, projection="3d"), motor_skin, rings, around, "motor blend (normalised lerp)")
    surface(fig.add_subplot(1, 3, 2, projection="3d"), slerp_skin, rings, around, "motor blend (slerp)")
    surface(fig.add_subplot(1, 3, 3, projection="3d"), matrix_skin, rings, around, "matrix blend")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")

    return fig


# --- scenario -------------------------------------------------------------------------
def cylinder(rings: int, around: int) -> tuple[Point, Scalar]:
    """Unit-radius skin around the x axis from x = 0 to 1, and the weight x of the second bone."""
    x, t = np.meshgrid(np.linspace(0.0, 1.0, rings), np.linspace(0.0, 2 * np.pi, around, endpoint=False), indexing="ij")
    coords = np.stack([x, np.cos(t), np.sin(t)], axis=-1).reshape(-1, 3)
    return point(coords), mv.scalar(x.reshape(-1, 1))


def main(plot_path: str = str(PLOT_DIR / "sketch_skinning.png")) -> plt.Figure:
    rings, around = 12, 24
    skin, w = cylinder(rings, around)
    root = mv.rotor()
    twist = (mv.yz * (np.radians(150.0) / 2)).exp()                     # second bone: 150° about x

    motor_skin, slerp_skin, matrix_skin = blend_skin(skin, w, root, twist)
    fig = draw_skinning(motor_skin, slerp_skin, matrix_skin, rings, around, plot_path)

    # --- readout -----------------------------------------------------------------------
    print(f"min radius, motor blend:  {radius(motor_skin).min():.3f}")
    print(f"min radius, matrix blend: {radius(matrix_skin).min():.3f}  (cos 75° = {np.cos(np.radians(75)):.3f})")

    # --- checks ------------------------------------------------------------------------
    np.testing.assert_allclose(radius(motor_skin), 1.0, atol=1e-12)
    np.testing.assert_allclose(radius(slerp_skin), 1.0, atol=1e-12)
    return fig


if __name__ == "__main__":
    main()
