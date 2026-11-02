"""Rayleigh-Bénard convection: a layer turning over from a still, conducting start, and the growth of a disturbance
against the Rayleigh number around the onset of convection."""

from __future__ import annotations

from itertools import islice, tee

import numpy as np

from numga import stack
from examples.fluids.convection import core

mv = core.mv

# The liquid: its density, its viscosity relative to its conduction of heat, the Prandtl number, about water's, and the
# mass a unit volume loses per degree, small, as the Boussinesq approximation takes it.
density, prandtl, expansion = 1.0, 7.0, 1e-3
# The layer turning over: its Rayleigh number, its length, its vertices along and up, its time step, the steps, the
# steps between frames, and the steps before the rolls show, run but not shown; the disturbance's size and its seed;
# the frames' duration in milliseconds.
rayleigh = 2e4
period, n_along, n_across = 4.0, 96, 25
dt, n_steps, n_stride, n_unshown = 5e-4, 300, 2, 80
disturbance, seed = 1e-2, 1
frame_duration_ms = 40
# The onset: the Rayleigh numbers swept; the wavenumber of the first rolls to grow between rigid walls, in inverse
# depths, and the layer one wavelength of them long, with its vertices along; the time step, the steps, and the steps
# between the two readings of the growth; and the size of the rolls the still layer starts with.
onset_rayleighs = np.array([1500, 1600, 1700, 1800, 1900])
critical_wavenumber = 3.117
onset_period, n_onset_along = 2 * np.pi / critical_wavenumber, 48
onset_dt, n_onset_steps, n_onset_stride = 1e-3, 600, 300
rolls_size = 1e-4


def conducting(heights: core.Scalar, size: float) -> core.Scalar:
    """The still layer's temperature at vertices of these heights `Scalar+[vertices]`, a half degree above the
    middle's at the floor and below at the ceiling, with a disturbance of this size inside it."""
    rng = np.random.default_rng(seed)
    noise = heights.context.multivector.scalar(rng.standard_normal((len(heights.batch()), 1))).field()   # Scalar+[vertices]
    return 0.5 - heights + size * heights * (1 - heights) * noise


def turning_over():
    """The layer, closed and opened at its seam, with the map reading the closed layer's fields on the open one, the
    open one's vertices `Point-[vertices]`, each triangle's area `Scalar+[faces]`, and the layer's stream function and
    temperature over time `Scalar+[vertices]` each, from the still layer disturbed."""
    closed, opened, closing, opening, places = core.layer(n_along, n_across, period)
    vertices = (core.volume + mv(core.Upright, places).dual()).field()          # Point-[opened vertices]
    areas, sides, gradients = core.triangles(opened, vertices)
    temperature = conducting(closing * (mv.y & vertices), disturbance)          # Scalar+[vertices]
    return closed, opened, opening, vertices, areas, core.convecting(
        closed, areas, sides, gradients, closed.interior[0], density, prandtl, expansion,
        np.array(rayleigh * prandtl / expansion), temperature, dt, n_steps, n_stride)


def onset() -> core.Scalar:
    """The growth rate of a small disturbance of the still layer at each Rayleigh number `[rayleighs] Scalar+`: the
    logarithm of its stream function's growth between two readings, over the time between them."""
    closed, opened, closing, _, places = core.layer(n_onset_along, n_across, onset_period)
    vertices = (core.volume + mv(core.Upright, places).dual()).field()          # Point-[opened vertices]
    areas, sides, gradients = core.triangles(opened, vertices)
    heights, runs = closing * (mv.y & vertices), closing * (mv.x & vertices)    # Scalar+[vertices], Scalar+[vertices]
    # A disturbance shaped like the first rolls, a wavelength along and half a wave up.
    rolls = rolls_size * (critical_wavenumber * runs).sin() * (np.pi * heights).sin()   # Scalar+[vertices]
    frames = list(core.convecting(closed, areas, sides, gradients, closed.interior[0], density, prandtl, expansion,
                                  onset_rayleighs * prandtl / expansion, conducting(heights, 0.0) + rolls, onset_dt,
                                  n_onset_steps, n_onset_stride))
    # The stream function's squared size over the layer at the middle and at the end; it grows at twice the rate.
    cells = (~closed.incidence[2, 0] * (areas / 3)).sum(axis=0)                # Scalar+[vertices]
    sizes = [(cells * stream.squared()).sites.sum() for stream, _ in frames[1:]]   # [readings] [rayleighs] Scalar+
    return (sizes[1].log() - sizes[0].log()) / (2 * n_onset_stride * onset_dt)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.convection import render

    closed, opened, opening, vertices, areas, states = turning_over()
    save_figure(render.draw_layer(opened, vertices, period), "convection_layer")
    # Shown once the rolls begin to show, and kept whole for the flow's growth.
    animation_states, growth_states = tee(states)
    save_animation(render.animate(opened, vertices, opening, period, islice(animation_states, n_unshown // n_stride, None)),
                   "convection", frame_duration_ms)
    # The flow's squared size over the layer, frame after frame from the still start.
    cells = (~closed.incidence[2, 0] * (areas / 3)).sum(axis=0)                # Scalar+[vertices]
    sizes = stack(tuple((cells * stream.squared()).sites.sum() for stream, _ in growth_states))   # [frames] Scalar+
    save_figure(render.draw_turnover(np.arange(0, n_steps + 1, n_stride) * dt, sizes, n_unshown * dt), "convection_turnover")
    save_figure(render.draw_onset(onset_rayleighs, onset()), "convection_onset")


if __name__ == "__main__":
    main()
