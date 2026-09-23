"""The skinning scene: a cylinder on two bones, the second twisted 150° about the shared axis."""

from __future__ import annotations

import numpy as np

from examples.geometry.skinning.core import blend_skin, cylinder, mv, radius


def skinning():
    """The cylinder skinned three ways, with its ring and around counts for meshing."""
    rings, around = 12, 24
    skin, weight = cylinder(rings, around)
    root = mv.rotor()
    twist = (mv.yz * (np.radians(150.0) / 2)).exp()                     # second bone: 150° about x
    motor_skin, slerp_skin, matrix_skin = blend_skin(skin, weight, root, twist)

    # --- checks ------------------------------------------------------------------------
    # Both motor blends are rigid about the axis, so every vertex keeps its unit radius.
    np.testing.assert_allclose(radius(motor_skin).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose(radius(slerp_skin).to_array(), 1.0, atol=1e-12)
    return motor_skin, slerp_skin, matrix_skin, rings, around


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.geometry.skinning import render

    motor_skin, slerp_skin, matrix_skin, rings, around = skinning()
    save_figure(render.draw_skinning(motor_skin, slerp_skin, matrix_skin, rings, around), "skinning")
    print(f"min radius, motor blend:  {radius(motor_skin).to_array().min():.3f}")
    print(f"min radius, matrix blend: {radius(matrix_skin).to_array().min():.3f}  (cos 75° = {np.cos(np.radians(75)):.3f})")
