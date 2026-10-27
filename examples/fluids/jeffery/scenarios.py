"""Scenes for Jeffery orbits: a sphere, a rod and an ellipsoid with three different axes in a simple
shear flow, the paths of their long axes on the sphere of directions, and the bodies tumbling."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.fluids.jeffery import core

Point = core.Point
# The planes across the flow and across the gradient, and the vortex plane, the plane the fluid turns in.
x, y, z = core.axes
# The bodies, a sphere, a rod and an ellipsoid with three different axes: their shapes, their semi-axes in proportion,
# and their size, the radius of the ball of the same volume, alike for all so that none crowds out the flow around
# it; their semi-axes; and the rod's long semi-axis over its short ones.
SHAPES = np.array([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0], [4.0, 2.0, 1.0]])
RADIUS = 1.3
SEMI_AXES = RADIUS * SHAPES / np.prod(SHAPES, axis=-1, keepdims=True) ** (1 / 3)
ASPECT = SHAPES[1, 0] / SHAPES[1, 1]
# The flow: its velocity along the first axis, per offset along the second; simple shear, the velocity along the
# flow growing with the offset along the gradient.
SHEAR_RATE = 1.0
FLOW = SHEAR_RATE * x.dual() * (y & Point)                                   # [] Point- <- Point-
# The time a rod of the aspect takes to come back to where it started, Jeffery's period.
PERIOD = 2 * np.pi * (ASPECT + 1 / ASPECT) / SHEAR_RATE
# The angles of each body's long axis from the normal of the vortex plane at the start, in the plane of the flow
# and that normal, and the start the animation shows.
TILTS = np.array([0.15, 0.4, 0.7, 1.0, 1.3])
SHOWN = 2
# The time span in periods of the rod's orbit, the steps per period, and the steps per frame.
N_PERIODS = 2
N_STEPS_PER_PERIOD = 120
N_STEPS_PER_FRAME = 3
# The heights at which the flow is drawn, all where it moves, and the duration of a frame.
HEIGHTS = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
DURATION_MS = 60
# Gallery picture: this makes plots/jeffery.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
GALLERY_PIXELS = 240


# --- math -----------------------------------------------------------------------------
def tumbling() -> core.Rotor:
    """Each body's orientation from each start over the time span `[steps + 1, bodies, starts]`."""
    stretches = core.stretch(SEMI_AXES)                                      # [bodies] Point- <- Point-
    # Each start turns the long axis from the flow towards the normal of the vortex plane, to its tilt from it.
    starts = ((x ^ z) * (-(np.pi / 2 - TILTS) / 2)).exp()                    # [starts] Rotor+
    starts = starts[None].broadcast_to((len(SEMI_AXES), len(TILTS)))         # [bodies, starts] Rotor+
    return stack(list(core.tumble(starts, stretches[:, None], FLOW, PERIOD / N_STEPS_PER_PERIOD,
                                  N_PERIODS * N_STEPS_PER_PERIOD)))          # [steps + 1, bodies, starts] Rotor+


def frames(rotors: core.Rotor) -> Iterator[tuple[core.Rotor, core.Point]]:
    """Per frame, every body's orientation from the start shown `[bodies]`, and the tip of its long axis."""
    for index in range(0, rotors.shape[0], N_STEPS_PER_FRAME):
        shown = rotors[index, :, SHOWN]                                      # [bodies] Rotor+
        yield shown, shown >> (x.dual() * SEMI_AXES[:, 0])


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.jeffery import render

    rotors = tumbling()                                                      # [steps + 1, bodies, starts] Rotor+
    trails = rotors >> x.dual()                                              # [steps + 1, bodies, starts] Point-
    heights = y.dual() * HEIGHTS                                             # [heights] Point-
    save_figure(render.draw_orbits(trails), "jeffery_orbits")
    # Gallery picture: this makes plots/jeffery.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
    save_animation(render.animate_tumbling(core.stretch(SEMI_AXES), heights, FLOW(heights), frames(rotors),
                                           GALLERY_PIXELS), "jeffery", DURATION_MS)

    # checks
    # The rotors stay unit over the whole span: the bodies stay rigid.
    np.testing.assert_allclose(rotors.scalar_norm_squared().to_array(), 1.0, atol=1e-11)
    # A sphere spins at the fluid's own rate, half the curl of the gradient.
    sphere = core.placed(core.stretch(SEMI_AXES)[0], rotors[-1, 0])          # [starts] Point- <- Plane+
    np.testing.assert_allclose((core.spin(sphere, FLOW) - core.curl(FLOW) / 2).norm().to_array(), 0.0, atol=1e-14)
    # The rod's axis is back where it started after one period, on a closed orbit along which
    # Jeffery's orbit constant holds; the tolerances test the step.
    rod = trails[:, 1]                                                       # [steps + 1, starts] Point-
    np.testing.assert_allclose((rod[N_STEPS_PER_PERIOD] - rod[0]).norm().to_array(), 0.0, atol=2e-2)
    flowing, across, spinning = x & rod, y & rod, z & rod                    # [steps + 1, starts] Scalar+, Scalar+, Scalar+
    constant = ((flowing.squared() + ASPECT**2 * across.squared()) / (ASPECT**2 * spinning.squared())).to_array()
    np.testing.assert_allclose(constant / constant[0], 1.0, rtol=2e-2)
    # The rod spends most of its time near the flow's direction: there its spin nearly vanishes.
    assert np.mean(np.abs(flowing.to_array()) > 0.8 * np.sin(TILTS)) > 0.5


if __name__ == "__main__":
    main()
