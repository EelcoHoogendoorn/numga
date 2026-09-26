"""A source passing behind two equal point masses: the critical curve and its caustic, the images
as the source crosses the caustic, and small round sources as the lens shows them
on the sky."""

from __future__ import annotations

import numpy as np

from examples.relativity.gravitational_lensing import core

# Two equal masses on the x axis, 1.1 Einstein angles apart: one caustic with six cusps.
POSITIONS = core.mv.x * np.array([-0.55, 0.55])                               # [masses] Vector
MASSES = np.array([0.5, 0.5])


# --- math -----------------------------------------------------------------------------
def deflected(observed: core.Vector) -> core.Vector:
    """The source direction each observed direction reaches, through the two masses."""
    return core.deflected(observed, POSITIONS, MASSES)                         # [...] Vector


def lens(resolution: int) -> tuple[core.Vector, core.Vector, core.Scalar]:
    """The sky's directions, the source direction each reaches, and the local map's area ratio on the
    sky, zero on the critical curve."""
    directions = core.sky(1.9, resolution)                                     # [rows, columns] Vector
    reached = core.deflected(directions, POSITIONS, MASSES)                    # [rows, columns] Vector
    local = core.local_map(directions, POSITIONS, MASSES)                      # [rows, columns] Vector <- Vector
    area = local.outermorphism(core.Area)(core.mv.xy) / core.mv.xy             # [rows, columns] Scalar

    # --- checks
    # The reflections have no trace, so the trace is two; the local map is the derivative of the
    # deflection.
    np.testing.assert_allclose(local.trace().to_array(), 2.0, atol=1e-11)
    probe, step = directions[::97, ::97], 1e-6 * core.mv.x                     # [probes, probes] Vector
    difference = (core.deflected(probe + step, POSITIONS, MASSES) - core.deflected(probe - step, POSITIONS, MASSES)) / 2e-6
    np.testing.assert_allclose((difference - local[::97, ::97](core.mv.x)).kernel, 0.0, atol=1e-7)
    return directions, reached, area


def light(directions: core.Vector, reached: core.Vector, centre: core.Vector, width: float
          ) -> tuple[core.Scalar, core.Scalar]:
    """The source about a centre unlensed, and the light each sky pixel shows: the source's
    brightness where its sightline arrives."""
    return core.brightness(directions, centre, width), core.brightness(reached, centre, width)


def tissot(count: int, radius: float) -> tuple[core.Vector, core.Scalar]:
    """Small round sources behind a lattice of sightlines, as they appear on the sky: each circle
    carried by the inverse of the local map, about its sightline; with the area ratio there."""
    spread = np.linspace(-1.25, 1.25, count)
    centres = core.mv.x * spread[None, :] + core.mv.y * spread[:, None]       # [rows, columns] Vector
    angles = np.linspace(0, 2 * np.pi, 64)
    offsets = ((core.mv.xy * (-angles / 2)).exp() >> core.mv.x) * radius             # [angles] Vector
    local = core.local_map(centres, POSITIONS, MASSES)                        # [rows, columns] Vector <- Vector
    seen = local[..., None].solve(offsets)                                 # [rows, columns, angles] Vector
    area = local.outermorphism(core.Area)(core.mv.xy) / core.mv.xy            # [rows, columns] Scalar

    # --- checks
    # With trace two the inverse is `(2 * Vector - local) / area`.
    np.testing.assert_allclose((seen - ((2 * core.Vector - local) / area)[..., None](offsets)).kernel, 0.0, atol=1e-11)
    return centres[..., None] + seen, area


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.relativity.gravitational_lensing import render

    directions, reached, area = lens(480)
    path = core.mv.x * np.linspace(-0.6, 0.6, 65) + core.mv.y * 0.12          # [steps] Vector
    save_figure(render.draw(directions, area, deflected, POSITIONS, light(directions, reached, path[21], 0.035)), "gravitational_lensing")
    steps = (light(directions, reached, centre, 0.035) for centre in path)
    save_animation(render.animate(directions, area, deflected, POSITIONS, steps), "gravitational_lensing", 90)
    save_figure(render.draw_tissot(directions, area, POSITIONS, *tissot(11, 0.025)), "gravitational_lensing_tissot")
