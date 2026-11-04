"""Rayleigh-Bénard convection: a layer turning over from a still, conducting start, and its energy budget."""

from __future__ import annotations

from itertools import islice, tee

import numpy as np

from numga import stack
from examples.fluids.convection import core

mv = core.mv

# The liquid: its density, its viscosity relative to its conduction of heat, the Prandtl number, about water's, and the
# mass a unit volume loses per degree, small, as the Boussinesq approximation takes it; the layer's absolute
# temperature, water's near 300 K, in degrees of the layer's one-kelvin difference; and the heat a unit volume of water
# holds per degree in the units of the flow's energy, for a layer a centimetre deep, its specific heat times the square
# of the depth over the square of its thermal diffusivity.
density, prandtl, expansion = 1.0, 7.0, 1e-3
reference_temperature, heat_capacity = 300.0, 2e13
# The layer turning over: its Rayleigh number, its length, its vertices along and up, its time step, the steps, the
# steps between frames, and the steps before the rolls show, run but not shown; the disturbance's size and its seed;
# the frames' duration in milliseconds, and the colours of their palette.
rayleigh = 2e4
period, n_along, n_across = 4.0, 96, 25
dt, n_steps, n_stride, n_unshown = 5e-4, 300, 2, 80
disturbance, seed = 1e-2, 1
frame_duration_ms = 40
n_animation_colours = 24


def conducting(heights: core.Scalar, size: float) -> core.Scalar:
    """The still layer's temperature at vertices of these heights `Scalar+[vertices]`, a half degree above the
    middle's at the floor and below at the ceiling, with a disturbance of this size inside it."""
    rng = np.random.default_rng(seed)
    noise = heights.context.multivector.scalar(rng.standard_normal((len(heights.batch()), 1))).field()   # Scalar+[vertices]
    return 0.5 - heights + size * heights * (1 - heights) * noise


def turning_over():
    """The open layer with the map reading the closed layer's fields on it, its vertices `Point-[vertices]`, the
    liquid's maps, and the layer's stream function and temperature over time `Scalar+[vertices]` each, with its budget,
    from the still layer disturbed."""
    closed, opened, closing, opening, places = core.layer(n_along, n_across, period)
    vertices = (core.volume + mv(core.Upright, places).dual()).field()          # Point-[opened vertices]
    liquid = core.forms(closed, opened, vertices, density, prandtl)
    temperature = conducting(closing * (mv.y & vertices), disturbance)          # Scalar+[vertices]
    return opened, opening, vertices, liquid, core.convecting(
        liquid, closed.reading(closed.interior[0], 0), closing * vertices, density, expansion,
        np.array(rayleigh * prandtl / expansion), reference_temperature, heat_capacity, temperature, dt, n_steps,
        n_stride)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.convection import render

    opened, opening, vertices, liquid, states = turning_over()
    save_figure(render.draw_layer(opened, vertices, period), "convection_layer")
    # Shown once the rolls begin to show, and kept whole for the flow's growth and the budget.
    animation_states, kept_states = tee(states)
    animation_frames = islice(animation_states, n_unshown // n_stride, None)
    save_animation(render.animate(opened, vertices, opening, period, animation_frames), "convection", frame_duration_ms,
                   colors=n_animation_colours)
    streams, _, budgets = zip(*kept_states)
    times = np.arange(0, n_steps + 1, n_stride) * dt
    # The flow's squared size over the layer, frame after frame from the still start.
    sizes = stack(tuple((liquid.cells * stream.squared()).sites.sum() for stream in streams))   # [frames] Scalar+
    save_figure(render.draw_turnover(times, sizes), "convection_turnover")
    # The budget frame after frame, and what it leaves unaccounted, as a share of the energy dissipated over the run:
    # the kinetic and potential energy, with the energy dissipated and less the potential energy the heat moved,
    # against the still start's.
    kinetic, potential, dissipated, conducted, produced = map(stack, zip(*budgets))   # [frames] Scalar+, Scalar+, Scalar+, Scalar+, Scalar+
    accounted = kinetic + potential + dissipated - conducted                  # [frames] Scalar+
    save_figure(render.draw_budget(times, kinetic - kinetic[0], potential - potential[0], dissipated, conducted,
                                   (accounted - accounted[0]) / dissipated[-1], produced), "convection_budget")


if __name__ == "__main__":
    main()
