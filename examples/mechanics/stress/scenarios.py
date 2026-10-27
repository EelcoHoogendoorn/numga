"""Scenes for the stressed cube: seen from the lab, from the principal frame and from the frame of
greatest shear, and from a frame turning half a turn in the shear plane, with Mohr's circle."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.mechanics.stress import core

mv, Point = core.mv, core.Point
x, y, z = core.axes
# The material: the first Lamé parameter and the shear modulus.
LAME = 2.0
SHEAR_MODULUS = 1.0
# The strain: stretches along x, y and z, and a shear in the xy plane.
STRETCHES = np.array([0.05, -0.05, 0.08])
SHEAR = 0.30
# The frames of the turning view, over half a turn, and the duration of each.
N_FRAMES = 60
DURATION_MS = 50


# --- math -----------------------------------------------------------------------------
def material():
    """The strain, the stress it causes, and the principal stresses, directions and frame."""
    strain = (core.axes * STRETCHES * (core.axes & Point)).sum(axis=0) + SHEAR * (x * (y & Point) + y * (x & Point))   # [] Strain
    stress = core.cauchy_stress(strain, LAME, SHEAR_MODULUS)                  # [] Stress
    return strain, stress, *core.principal_frame(stress)


def frames() -> tuple[tuple[core.Point, core.Point, core.Point, core.Point, core.Point], core.Scalar]:
    """The cube seen from the lab, from the principal frame, and from the frame an eighth of a turn
    on, where the shear is greatest `[3]`; with the principal stresses."""
    strain, stress, values, directions, principal = material()
    rotors = stack([mv.rotor(), principal, ((x ^ y) * (-np.pi / 8)).exp() * principal])   # [3] Rotor+
    return core.views(strain, stress, directions, rotors), values


def turns() -> core.Rotor:
    """The frames turning half a turn in the shear plane from the principal frame `[frames]`."""
    _, _, _, _, principal = material()
    angles = np.linspace(0.0, np.pi, N_FRAMES, endpoint=False)
    return ((x ^ y) * (-angles / 2)).exp() * principal                        # [frames] Rotor+


def turning() -> Iterator[tuple[core.Point, core.Point, core.Point, core.Point, core.Point]]:
    """Per frame, the cube seen from a frame turning in the shear plane."""
    strain, stress, _, directions, _ = material()
    rotors = turns()                                                          # [frames] Rotor+
    for frame in range(N_FRAMES):
        yield core.views(strain, stress, directions, rotors[frame])


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.stress import render

    seen, values = frames()
    save_figure(render.draw_frames(seen, values), "stress_frames")
    save_animation(render.animate_turning(values, turning()), "stress_rotation", DURATION_MS)

    # checks
    # In the principal frame no face is sheared, and the tractions across the faces facing x, y and z
    # are the principal stresses, the turned planes' directions the principal directions.
    _, stress, _, directions, principal = material()
    _, _, normal, shear, _ = seen
    np.testing.assert_allclose(shear[1].norm().to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.sort((core.axes & normal[1, :3]).to_array()), values.to_array(), rtol=1e-12)
    turned = principal >> core.axes                                            # [3] Plane+
    # Two parallel directions join to nothing.
    np.testing.assert_allclose((turned.dual() & stress(turned)).norm().to_array(), 0.0, atol=1e-14)
    # An eighth of a turn on, the face facing x feels the largest shear, the radius of Mohr's circle.
    centre, radius = core.mohr_circle(values)
    np.testing.assert_allclose(shear[2, 0].norm().to_array(), radius.to_array(), rtol=1e-8)
    # Turning, the normal and shear traction on the face facing x go around Mohr's circle.
    strain, _, _, _, _ = material()
    _, _, turning_normal, turning_shear, _ = core.views(strain, stress, directions, turns())   # [frames, 6] Point-
    along, across = x & turning_normal[:, 0], y & turning_shear[:, 0]          # [frames] Scalar+
    np.testing.assert_allclose(((along - centre).squared() + across.squared()).to_array(), radius.squared().to_array(), rtol=1e-8)


if __name__ == "__main__":
    main()
