"""Scenes for the stressed cube: seen from the lab, from the principal frame and from the frame of
greatest shear, and from a frame turning half a turn in the shear plane, with Mohr's circle."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.mechanics.stress import core

mv, Vector = core.mv, core.Vector
x, y, z = core.axes
# The material: the first Lamé parameter and the shear modulus.
LAME = 2.0
SHEAR_MODULUS = 1.0
# The strain: stretches along x, y and z, and a shear in the xy plane.
STRETCHES = np.array([0.05, -0.05, 0.08])
SHEAR = 0.30
# The frames of the turning view, over half a turn.
FRAMES = 60


# --- math -----------------------------------------------------------------------------
def material():
    """The strain, the stress it causes, and the principal stresses, directions and frame."""
    strain = (core.axes * STRETCHES * (core.axes | Vector)).sum(axis=0) + SHEAR * (x * (y | Vector) + y * (x | Vector))   # [] Strain
    stress = core.cauchy_stress(strain, LAME, SHEAR_MODULUS)                  # [] Stress
    return strain, stress, *core.principal_frame(stress)


def frames():
    """The cube seen from the lab, from the principal frame, and from the frame an eighth of a turn
    on, where the shear is greatest; with the principal stresses."""
    strain, stress, values, directions, principal = material()
    rotors = stack([mv.rotor(), principal, ((x ^ y) * (-np.pi / 8)).exp() * principal])   # [3] Rotor
    views = core.Views(strain, stress, directions, rotors)

    # --- checks
    # In the principal frame no face is sheared, and the normal tractions on the faces facing x, y and
    # z are the principal stresses, the turned axes their directions.
    np.testing.assert_allclose(views.shear[1].norm().to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.sort((views.normal[1, :3] | core.axes).to_array()), values.to_array(), rtol=1e-12)
    turned = principal >> core.axes                                            # [3] Vector
    np.testing.assert_allclose((stress(turned) ^ turned).norm().to_array(), 0.0, atol=1e-14)
    # An eighth of a turn on, the face facing x feels the largest shear, the radius of Mohr's circle.
    _, radius = core.mohr_circle(values)
    np.testing.assert_allclose(views.shear[2, 0].norm().to_array(), radius.to_array(), rtol=1e-8)
    return views, values


def turning():
    """The cube seen from a frame turning half a turn in the shear plane from the principal frame,
    with the principal stresses."""
    strain, stress, values, directions, principal = material()
    angles = np.linspace(0.0, np.pi, FRAMES, endpoint=False)
    rotors = ((x ^ y) * (-angles / 2)).exp() * principal                      # [frames] Rotor
    views = core.Views(strain, stress, directions, rotors)

    # --- checks
    # The normal and shear traction on the face facing x go around Mohr's circle.
    centre, radius = core.mohr_circle(values)
    normal, shear = views.normal[:, 0] | x, views.shear[:, 0] | y               # [frames] Scalar
    np.testing.assert_allclose(((normal - centre).squared() + shear.squared()).to_array(), radius.squared().to_array(), rtol=1e-8)
    return views, values


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.mechanics.stress import render

    save_figure(render.draw_frames(*frames(), ["lab frame", "principal frame", "greatest shear"]), "stress_frames")
    save_animation(render.animate_turning(*turning()), "stress_rotation", 50)
