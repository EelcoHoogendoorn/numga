"""A conducting layer of liquid heated from below turns over in Rayleigh-Bénard convection, with its flow growth and
energy budget."""

from __future__ import annotations

from itertools import islice, tee

import numpy as np

from numga import stack
from examples.fluids.convection import core

mv = core.mv

# Reference density, Prandtl number (viscosity relative to thermal diffusivity, near water's), Boussinesq thermal
# expansion per degree, absolute reference temperature in one-kelvin units near 300 K, and volumetric heat capacity in
# flow-energy units for a one-centimetre water layer.
density, prandtl, expansion = 1.0, 7.0, 1e-3
reference_temperature, heat_capacity = 300.0, 2e13
# Rayleigh number, periodic length, vertex counts along and across the layer, time step, step counts, initial perturbation
# amplitude and seed, frame duration in milliseconds, and animation palette size.
rayleigh = 2e4
period, n_along, n_across = 4.0, 96, 25
dt, n_steps, n_stride, n_unshown = 5e-4, 300, 2, 80
disturbance, seed = 1e-2, 1
frame_duration_ms = 40
n_animation_colours = 24


def conducting(heights: core.Scalar, size: float) -> core.Scalar:
    """Linear conductive temperature profile running from +0.5 at the floor to -0.5 at the ceiling,
    perturbed in the interior by random noise."""
    rng = np.random.default_rng(seed)
    noise = mv.scalar(rng.standard_normal((len(heights.batch()), 1))).field()   # Scalar+[V]
    return 0.5 - heights + size * heights * (1 - heights) * noise


def turning_over(n_along: int, n_across: int):
    """The opened strip, the map reading closed-strip fields onto it, its spatial vertex offsets,
    the assembled operators, and the yielded stream function, temperature and energy budget
    over time from the perturbed conductive start."""
    closed, opened, closing, opening, places = core.layer(n_along, n_across, period)
    vertices = mv(core.Upright, places).dual().field()                          # Antivector-[V_opened]
    liquid = core.forms(closed, opened, vertices, density, prandtl)
    temperature = conducting(closing * (mv.y & vertices), disturbance)          # Scalar+[V]
    return opened, opening, vertices, liquid, core.convecting(
        liquid, closed.reading(closed.interior[0], 0), closing * vertices, density, expansion,
        np.array(rayleigh * prandtl / expansion), reference_temperature, heat_capacity, temperature, dt, n_steps,
        n_stride)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.convection import render

    opened, opening, vertices, liquid, states = turning_over(n_along, n_across)
    save_figure(render.draw_layer(opened, vertices, period), "convection_layer")
    # Skip the quiet initial growth in the animation while keeping every yielded state for the growth and budget charts.
    animation_states, kept_states = tee(states)
    animation_frames = islice(animation_states, n_unshown // n_stride, None)
    save_animation(render.animate(opened, vertices, opening, period, animation_frames), "convection", frame_duration_ms,
                   colors=n_animation_colours)
    streams, _, budgets = zip(*kept_states)
    times = np.arange(0, n_steps + 1, n_stride) * dt
    # Integrated squared stream function over the layer at each frame.
    sizes = stack(tuple((liquid.cells * stream.squared()).sites.sum() for stream in streams))   # [n_frames] Scalar+
    save_figure(render.draw_turnover(times, sizes), "convection_turnover")
    # Energy budget terms and normalized conservation residual relative to the initial state.
    kinetic, potential, dissipated, conducted, produced = map(stack, zip(*budgets))   # [n_frames] Scalar+, Scalar+, Scalar+, Scalar+, Scalar+
    accounted = kinetic + potential + dissipated - conducted                  # [n_frames] Scalar+
    save_figure(render.draw_budget(times, kinetic - kinetic[0], potential - potential[0], dissipated, conducted,
                                   (accounted - accounted[0]) / dissipated[-1], produced), "convection_budget")


if __name__ == "__main__":
    main()
