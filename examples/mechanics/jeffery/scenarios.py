"""Scenes for Jeffery orbits: a sphere, a rod and an ellipsoid with three different axes in a simple
shear flow, the paths of their long axes on the sphere of directions, and the bodies tumbling."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.mechanics.jeffery import core

Vector = core.Vector
x, y, z = core.axes                                        # flow, gradient, vorticity
# The bodies: their names and semi-axes; the rod's long semi-axis over its short ones.
NAMES = ["sphere", "rod", "three axes"]
SEMI_AXES = np.array([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0], [4.0, 2.0, 1.0]])
ASPECT = 4.0
# The flow: its velocity along the first axis, per offset along the second.
SHEAR_RATE = 1.0
# The angles of each body's long axis from the vorticity axis at the start, in the plane of the flow
# and the vorticity, and the start the animation shows.
TILTS = np.array([0.15, 0.4, 0.7, 1.0, 1.3])
SHOWN = 2
# The time span in periods of the rod's orbit, the steps per period, and the steps per frame.
PERIODS = 2
STEPS_PER_PERIOD = 120
STRIDE = 3
# The heights at which the flow is drawn.
HEIGHTS = np.linspace(-3.0, 3.0, 7)


# --- math -----------------------------------------------------------------------------
def flow():
    """Simple shear: the velocity along the flow grows with the offset along the gradient."""
    return SHEAR_RATE * x * (y | Vector)                                     # [] Gradient


def period() -> float:
    """The time a rod of the aspect takes to come back to where it started, Jeffery's period."""
    return 2 * np.pi * (ASPECT + 1 / ASPECT) / SHEAR_RATE


def tumbling() -> core.Tumbling:
    """Each body from each start, tumbling for the time span."""
    stretches = core.stretch(SEMI_AXES)                                      # [bodies] Stretch
    starts = ((x ^ z) * (-(np.pi / 2 - TILTS) / 2)).exp()                    # [starts] Rotor: the long axis towards z
    starts = starts[None].broadcast_to((len(NAMES), len(TILTS)))             # [bodies, starts] Rotor
    rotors = stack(list(core.tumble(starts, stretches[:, None], flow(), period() / STEPS_PER_PERIOD,
                                    PERIODS * STEPS_PER_PERIOD)))           # [steps + 1, bodies, starts] Rotor
    trails = rotors >> x                                                     # [steps + 1, bodies, starts] Vector
    shown = rotors[::STRIDE, :, SHOWN]                                       # [frames, bodies] Rotor
    heights = y * HEIGHTS                                                    # [heights] Vector
    tumbling = core.Tumbling(trails, stretches, shown, shown >> (x * SEMI_AXES[:, 0]), heights, flow()(heights))

    # --- checks
    # The rotors stay unit over the whole span: the bodies stay rigid.
    np.testing.assert_allclose(rotors.scalar_norm_squared().to_array(), 1.0, atol=1e-11)
    # A sphere spins at the fluid's own rate, half the curl of the gradient.
    sphere = core.placed(stretches[0], rotors[-1, 0])                         # [starts] Shape
    np.testing.assert_allclose((core.spin(sphere, flow()) - core.curl(flow()) / 2).norm().to_array(), 0.0, atol=1e-14)
    # The rod's axis is back where it started after one period, on a closed orbit along which
    # Jeffery's orbit constant holds; the tolerances test the step.
    rod = trails[:, 1]                                                       # [steps + 1, starts] Vector
    np.testing.assert_allclose((rod[STEPS_PER_PERIOD] - rod[0]).norm().to_array(), 0.0, atol=2e-2)
    flowing, across, spinning = rod | x, rod | y, rod | z                     # [steps + 1, starts] Scalar each
    constant = ((flowing.squared() + ASPECT**2 * across.squared()) / (ASPECT**2 * spinning.squared())).to_array()
    np.testing.assert_allclose(constant / constant[0], 1.0, rtol=2e-2)
    # The rod spends most of its time near the flow's direction: there its spin nearly vanishes.
    assert np.mean(np.abs(flowing.to_array()) > 0.8 * np.sin(TILTS)) > 0.5
    return tumbling


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.mechanics.jeffery import render

    scene = tumbling()
    save_figure(render.draw_orbits(scene, NAMES), "jeffery_orbits")
    save_animation(render.animate_tumbling(scene, NAMES), "jeffery_tumbling", 60)
