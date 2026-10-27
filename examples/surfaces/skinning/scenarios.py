"""The skinning scene deforms a cylinder over two bones when the tip bone twists about their shared axis."""

from __future__ import annotations

import numpy as np

from examples.surfaces.skinning.core import blend_skin, cylinder, mv

# The cylinder's rings along the bones and vertices around each, and how far the tip bone is twisted.
N_RINGS, N_AROUND = 12, 24
TWIST = np.radians(150.0)


# --- math -----------------------------------------------------------------------------
def skinning():
    """The two-bone cylinder is skinned by normalised motor blending, motor slerp, and point-map blending."""
    vertices, weights = cylinder(N_RINGS, N_AROUND)                            # [vertices] Point-, [vertices]
    root = mv.rotor()                                                          # [] Motor+
    # The tip bone is turned about the shared x axis.
    tip = (mv.yz * (TWIST / 2)).exp()                                          # [] Motor+
    return blend_skin(vertices, weights, root, tip)                            # [vertices] Point-, Point-, Point-


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.surfaces.skinning import render

    motor_skin, slerp_skin, matrix_skin = skinning()
    save_figure(render.draw_skinning(motor_skin, slerp_skin, matrix_skin, N_RINGS, N_AROUND), "skinning")

    # checks
    # Both motor blends are rigid about the axis, so every vertex keeps its unit radius; the point-map blend
    # averages two points on a circle and pulls a vertex of weight w in to the chord between them.
    _, weights = cylinder(N_RINGS, N_AROUND)
    np.testing.assert_allclose((motor_skin.normalized() & mv.yz).norm().to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose((slerp_skin.normalized() & mv.yz).norm().to_array(), 1.0, atol=1e-12)
    chord_radius = np.sqrt(1 - 2 * weights * (1 - weights) * (1 - np.cos(TWIST)))
    np.testing.assert_allclose((matrix_skin.normalized() & mv.yz).norm().to_array(), chord_radius, atol=1e-12)


if __name__ == "__main__":
    main()
