"""The skinning scene: a cylinder on two bones, the second twisted about the shared axis."""

from __future__ import annotations

import numpy as np

from examples.surfaces.skinning.core import blend_skin, cylinder, mv, radius

# The cylinder's rings along the bones and vertices around each, and how far the second bone is twisted.
N_RINGS, N_AROUND = 12, 24
TWIST = np.radians(150.0)


# --- math -----------------------------------------------------------------------------
def skinning():
    """The cylinder skinned three ways: blended motors, normalised and slerped, and blended maps."""
    skin, weight = cylinder(N_RINGS, N_AROUND)                                 # [vertices] Point, Scalar
    root = mv.rotor()                                                          # [] Motor
    # The second bone, turned about x.
    twist = (mv.yz * (TWIST / 2)).exp()                                        # [] Motor
    return blend_skin(skin, weight, root, twist)                               # [vertices] Point each


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.surfaces.skinning import render

    motor_skin, slerp_skin, matrix_skin = skinning()
    save_figure(render.draw_skinning(motor_skin, slerp_skin, matrix_skin, N_RINGS, N_AROUND), "skinning")

    # checks
    # Both motor blends are rigid about the axis, so every vertex keeps its unit radius; the matrix blend
    # averages two points of a circle, and pulls a vertex of weight w in to the chord between them.
    _, weight = cylinder(N_RINGS, N_AROUND)
    w = weight.to_array()
    np.testing.assert_allclose(radius(motor_skin).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose(radius(slerp_skin).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose(radius(matrix_skin).to_array(), np.sqrt(1 - 2 * w * (1 - w) * (1 - np.cos(TWIST))), atol=1e-12)


if __name__ == "__main__":
    main()
