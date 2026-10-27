"""A source passing behind two equal point masses: the critical curve and its caustic, the images
as the source crosses the caustic, and small round sources as the lens shows them
on the sky."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from examples.relativity.gravitational_lensing import core

# Two equal masses on the x axis, 1.1 Einstein angles apart: one caustic with six cusps.
POSITIONS = core.mv.x.dual() * np.array([-0.55, 0.55])                         # [masses] Point-
MASSES = np.array([0.5, 0.5])
# The pixels along each side of the sky the critical curve is traced on.
N_PIXELS = 480
# The source's path, across the caustic 0.12 Einstein angles off the masses' axis, the step the still
# shows, the source's width, and the duration of a frame in milliseconds.
N_PATH_STEPS = 65
PATH = (core.mv.x * np.linspace(-0.6, 0.6, N_PATH_STEPS) + core.mv.y * 0.12).dual()   # [steps] Point-
SHOWN_STEP = 21
SOURCE_WIDTH = 0.035
DURATION_MS = 90


# --- math -----------------------------------------------------------------------------
def deflected(observed: core.Point) -> core.Point:
    """The source direction each observed direction reaches, through the two masses."""
    return core.deflected(observed, POSITIONS, MASSES)                         # [...] Point-


def lens(resolution: int) -> tuple[core.Point, core.Scalar, core.LocalMap]:
    """The sky's directions, the local map's area ratio on the sky, zero on the critical curve, and
    the local map."""
    directions = core.sky(1.9, resolution)                                     # [rows, columns] Point-
    local = core.local_map(directions, POSITIONS, MASSES)                      # [rows, columns] Point- <- Point-
    area = local.det()                                                         # [rows, columns] Scalar+
    return directions, area, local


def light(centre: core.Point, width: float
          ) -> tuple[Callable[[core.Point], core.Scalar], Callable[[core.Point], core.Scalar]]:
    """The source about a centre unlensed, and the light the sky shows in each direction: the
    source's brightness where its sightline arrives."""
    return (lambda at: core.brightness(at, centre, width)), (lambda at: core.brightness(deflected(at), centre, width))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.gravitational_lensing import render

    directions, area, local = lens(N_PIXELS)
    save_figure(render.draw(directions, area, deflected, POSITIONS, light(PATH[SHOWN_STEP], SOURCE_WIDTH)), "gravitational_lensing")
    steps = (light(centre, SOURCE_WIDTH) for centre in PATH)
    save_animation(render.animate(directions, area, deflected, POSITIONS, steps), "gravitational_lensing", DURATION_MS)

    # checks
    # The reflections have no trace, so the trace is two; the local map is the derivative of the
    # deflection.
    np.testing.assert_allclose(local.trace().to_array(), 2.0, atol=1e-11)
    probe, step = directions[::97, ::97], 1e-6 * core.mv.x.dual()              # [probes, probes] Point-
    difference = (core.deflected(probe + step, POSITIONS, MASSES) - core.deflected(probe - step, POSITIONS, MASSES)) / 2e-6
    np.testing.assert_allclose((difference - local[::97, ::97](core.mv.x.dual())).kernel, 0.0, atol=1e-7)


if __name__ == "__main__":
    main()
