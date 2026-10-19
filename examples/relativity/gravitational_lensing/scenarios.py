"""A source passing behind two equal point masses: the critical curve and its caustic, the images
as the source crosses the caustic, and small round sources as the lens shows them
on the sky."""

from __future__ import annotations

import numpy as np

from examples.relativity.gravitational_lensing import core

# Two equal masses on the x axis, 1.1 Einstein angles apart: one caustic with six cusps.
POSITIONS = core.mv.x.dual() * np.array([-0.55, 0.55])                        # [masses] Point
MASSES = np.array([0.5, 0.5])


# --- math -----------------------------------------------------------------------------
def deflected(observed: core.Point) -> core.Point:
    """The source direction each observed direction reaches, through the two masses."""
    return core.deflected(observed, POSITIONS, MASSES)                         # [...] Point


def lens(resolution: int) -> tuple[core.Point, core.Point, core.Scalar]:
    """The sky's directions, the source direction each reaches, and the local map's area ratio on the
    sky, zero on the critical curve."""
    directions = core.sky(1.9, resolution)                                     # [rows, columns] Point
    reached = core.deflected(directions, POSITIONS, MASSES)                    # [rows, columns] Point
    local = core.local_map(directions, POSITIONS, MASSES)                      # [rows, columns] Point <- Vector
    area = local.det()                                                         # [rows, columns] Scalar

    # --- checks
    # The reflections have no trace, so the trace is two; the local map is the derivative of the
    # deflection.
    np.testing.assert_allclose(local.trace().to_array(), 2.0, atol=1e-11)
    probe, step = directions[::97, ::97], 1e-6 * core.mv.x.dual()              # [probes, probes] Point
    difference = (core.deflected(probe + step, POSITIONS, MASSES) - core.deflected(probe - step, POSITIONS, MASSES)) / 2e-6
    np.testing.assert_allclose((difference - local[::97, ::97](core.mv.x.dual())).kernel, 0.0, atol=1e-7)
    return directions, reached, area


def light(directions: core.Point, reached: core.Point, centre: core.Point, width: float
          ) -> tuple[core.Scalar, core.Scalar]:
    """The source about a centre unlensed, and the light each sky pixel shows: the source's
    brightness where its sightline arrives."""
    return core.brightness(directions, centre, width), core.brightness(reached, centre, width)


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.relativity.gravitational_lensing import render

    directions, reached, area = lens(480)
    path = (core.mv.x * np.linspace(-0.6, 0.6, 65) + core.mv.y * 0.12).dual()   # [steps] Point
    save_figure(render.draw(directions, area, deflected, POSITIONS, light(directions, reached, path[21], 0.035)), "gravitational_lensing")
    steps = (light(directions, reached, centre, 0.035) for centre in path)
    save_animation(render.animate(directions, area, deflected, POSITIONS, steps), "gravitational_lensing", 90)
