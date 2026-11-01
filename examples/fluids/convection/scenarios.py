"""Rayleigh-Bénard convection: a layer turning over from a still, conducting start, and the growth of a disturbance
against the Rayleigh number around the onset of convection."""

from __future__ import annotations

from itertools import islice

import numpy as np

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
dt, n_steps, stride, n_unshown = 5e-4, 360, 3, 48
disturbance, seed = 1e-2, 1
frame_duration_ms = 40
# The onset: the Rayleigh numbers swept, the layer's length one wavelength of the first rolls to grow between rigid
# walls, its vertices along, the time step, the steps, and the steps between the two readings of the growth.
onset_rayleighs = np.array([1500.0, 1600.0, 1700.0, 1800.0, 1900.0])
onset_period, onset_along = 2 * np.pi / 3.117, 48
onset_dt, onset_steps, onset_stride = 1e-3, 600, 300


def conducting(places: np.ndarray, size: float) -> np.ndarray:
    """The still layer's temperature at each vertex `[vertices]`, a half degree above the middle's at the floor and
    below at the ceiling, with a disturbance of this size inside it."""
    rng = np.random.default_rng(seed)
    inside = places[:, 1] * (1 - places[:, 1])
    return 0.5 - places[:, 1] + size * inside * rng.standard_normal(len(places))


def turning_over():
    """The layer's stream function and temperature over time `Scalar+[vertices]` each, from the still layer disturbed,
    once its rolls begin to show."""
    topology, places, shifts, floor, ceiling = core.layer(n_along, n_across, period)
    vertices, edges = core.placed(topology, places, shifts)
    areas, sides, gradients, cells = core.triangles(topology, edges)
    temperature = mv.scalar(conducting(places, disturbance)[:, None]).field()
    return topology, vertices, islice(core.convecting(
        topology, areas, sides, gradients, cells, topology.interior[0], density, prandtl, expansion,
        np.array(rayleigh * prandtl / expansion), temperature, dt, n_steps, stride), n_unshown // stride, None)


def onset() -> core.Scalar:
    """The growth rate of a small disturbance of the still layer at each Rayleigh number `[rayleighs] Scalar+`: the
    logarithm of its stream function's growth between two readings, over the time between them."""
    topology, places, shifts, floor, ceiling = core.layer(onset_along, n_across, onset_period)
    vertices, edges = core.placed(topology, places, shifts)
    areas, sides, gradients, cells = core.triangles(topology, edges)
    still = mv.scalar(conducting(places, 0.0)[:, None]).field()
    # A disturbance shaped like the first rolls, a wavelength along and half a wave up.
    rolls = 1e-4 * np.sin(2 * np.pi * places[:, 0] / onset_period) * np.sin(np.pi * places[:, 1])
    frames = list(core.convecting(topology, areas, sides, gradients, cells, topology.interior[0], density, prandtl,
                                  expansion, onset_rayleighs * prandtl / expansion,
                                  still + mv.scalar(rolls[:, None]).field(), onset_dt, onset_steps, onset_stride))
    # The stream function's squared size over the layer at the middle and at the end; it grows at twice the rate.
    sizes = [(stream * cells * stream).sites.sum() for stream, _ in frames[1:]]
    return (sizes[1].log() - sizes[0].log()) / (2 * onset_stride * onset_dt)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.convection import render

    topology, vertices, frames = turning_over()
    save_figure(render.draw_layer(topology, vertices, n_across, period), "convection_layer")
    save_animation(render.animate(topology, vertices, n_across, period, frames), "convection", frame_duration_ms)
    save_figure(render.draw_onset(onset_rayleighs, onset()), "convection_onset")


if __name__ == "__main__":
    main()
