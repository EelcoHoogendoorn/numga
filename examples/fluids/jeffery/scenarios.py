"""Scenes for Jeffery orbits: a sphere, a rod and an ellipsoid with three different axes in a simple
shear flow, the paths of their long axes on the sphere of directions, and the bodies tumbling."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.fluids.jeffery import core

Point = core.Point
x, y, z = core.axes                                        # the planes across the flow, gradient and vorticity
# The bodies, a sphere, a rod and an ellipsoid with three different axes: their semi-axes, and the rod's long
# semi-axis over its short ones.
SEMI_AXES = np.array([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0], [4.0, 2.0, 1.0]])
ASPECT = 4.0
# The flow: its velocity along the first axis, per offset along the second.
SHEAR_RATE = 1.0
# The angles of each body's long axis from the vorticity axis at the start, in the plane of the flow
# and the vorticity, and the start the animation shows.
TILTS = np.array([0.15, 0.4, 0.7, 1.0, 1.3])
SHOWN = 2
# The time span in periods of the rod's orbit, the steps per period, and the steps per frame.
N_PERIODS = 2
N_STEPS_PER_PERIOD = 120
N_STEPS_PER_FRAME = 3
# The heights at which the flow is drawn, and the duration of a frame.
HEIGHTS = np.linspace(-3.0, 3.0, 7)
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def flow() -> core.Gradient:
    """Simple shear: the velocity along the flow grows with the offset along the gradient."""
    return SHEAR_RATE * x.dual() * (y & Point)                               # [] Gradient


def period() -> float:
    """The time a rod of the aspect takes to come back to where it started, Jeffery's period."""
    return 2 * np.pi * (ASPECT + 1 / ASPECT) / SHEAR_RATE


def tumbling() -> core.Rotor:
    """Each body's orientation from each start over the time span `[steps + 1, bodies, starts]`."""
    stretches = core.stretch(SEMI_AXES)                                      # [bodies] Stretch
    starts = ((x ^ z) * (-(np.pi / 2 - TILTS) / 2)).exp()                    # [starts] Rotor: the long axis towards z
    starts = starts[None].broadcast_to((len(SEMI_AXES), len(TILTS)))         # [bodies, starts] Rotor
    return stack(list(core.tumble(starts, stretches[:, None], flow(), period() / N_STEPS_PER_PERIOD,
                                  N_PERIODS * N_STEPS_PER_PERIOD)))          # [steps + 1, bodies, starts] Rotor


def frames(rotors: core.Rotor) -> Iterator[tuple[core.Rotor, core.Point]]:
    """Per frame, every body's orientation from the start shown `[bodies]`, and the tip of its long axis."""
    for index in range(0, rotors.shape[0], N_STEPS_PER_FRAME):
        shown = rotors[index, :, SHOWN]                                      # [bodies] Rotor
        yield shown, shown >> (x.dual() * SEMI_AXES[:, 0])


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.fluids.jeffery import render

    rotors = tumbling()                                                      # [steps + 1, bodies, starts] Rotor
    trails = rotors >> x.dual()                                              # [steps + 1, bodies, starts] Point
    heights = y.dual() * HEIGHTS                                             # [heights] Point
    save_figure(render.draw_orbits(trails), "jeffery_orbits")
    save_animation(render.animate_tumbling(core.stretch(SEMI_AXES), heights, flow()(heights), frames(rotors)),
                   "jeffery_tumbling", DURATION_MS)

    # checks
    # The rotors stay unit over the whole span: the bodies stay rigid.
    np.testing.assert_allclose(rotors.scalar_norm_squared().to_array(), 1.0, atol=1e-11)
    # A sphere spins at the fluid's own rate, half the curl of the gradient.
    sphere = core.placed(core.stretch(SEMI_AXES)[0], rotors[-1, 0])          # [starts] Shape
    np.testing.assert_allclose((core.spin(sphere, flow()) - core.curl(flow()) / 2).norm().to_array(), 0.0, atol=1e-14)
    # The rod's axis is back where it started after one period, on a closed orbit along which
    # Jeffery's orbit constant holds; the tolerances test the step.
    rod = trails[:, 1]                                                       # [steps + 1, starts] Point
    np.testing.assert_allclose((rod[N_STEPS_PER_PERIOD] - rod[0]).norm().to_array(), 0.0, atol=2e-2)
    flowing, across, spinning = x & rod, y & rod, z & rod                     # [steps + 1, starts] Scalar each
    constant = ((flowing.squared() + ASPECT**2 * across.squared()) / (ASPECT**2 * spinning.squared())).to_array()
    np.testing.assert_allclose(constant / constant[0], 1.0, rtol=2e-2)
    # The rod spends most of its time near the flow's direction: there its spin nearly vanishes.
    assert np.mean(np.abs(flowing.to_array()) > 0.8 * np.sin(TILTS)) > 0.5


if __name__ == "__main__":
    main()
