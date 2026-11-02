"""Salt fingers: warm salty water over cold fresh water fingering across the interface between them, and the growth of
a disturbance of evenly stratified water against the density ratio."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.fluids.saltfingers import core

mv = core.mv

# The water: its density, its viscosity relative to its conduction of heat, the Prandtl number, about water's, the mass
# a unit volume loses per degree, small, as the Boussinesq approximation takes it, and salt's diffusivity relative to
# heat's, ten times seawater's, so that the mesh resolves the fingers' salt.
density, prandtl, expansion = 1.0, 7.0, 1e-3
salt_diffusivity = 0.1
# The fingers: the heat's Rayleigh number, the density ratio, the tank's length, its vertices along and up, and the
# interface's thickness; the time step, the steps, the steps between frames, and the frames while the interface is
# still flat, not drawn; the disturbance's size and its seed; the frames' duration in milliseconds.
rayleigh, ratio = 5e5, 2.0
period, n_along, n_across = 2.0, 136, 69
thickness = 0.05
dt, n_steps, stride, n_flat = 2e-5, 1080, 6, 34
disturbance, seed = 4e-2, 1
frame_duration_ms = 40
# The growth: the density ratios swept, the tank's length one wavelength of the first rolls to grow between rigid
# walls, its vertices along and up, the disturbance's size, the time step, the steps, and the steps between the two
# readings of the growth.
growth_ratios = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
growth_period, growth_along, growth_across = 2 * np.pi / 3.117, 48, 25
growth_disturbance = 1e-6
growth_dt, growth_steps, growth_stride = 5e-4, 400, 200


def layered(places: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """The temperature and the salinity at each vertex `[vertices]`, each a half above the middle's at the top and below
    at the bottom, changing across an interface of this thickness, with a disturbance of the salinity of this size
    inside the interface."""
    rng = np.random.default_rng(seed)
    profile = np.tanh((places[:, 1] - 0.5) / thickness)
    return profile / 2, profile / 2 + size * (1 - profile ** 2) * rng.standard_normal(len(places))


def fingers():
    """The tank, its temperature and salinity at the start `Scalar+[vertices]`, the two layers at rest, and its stream
    function, temperature and salinity over time `Scalar+[vertices]` each, from just before the fingers show."""
    topology, places, shifts = core.tank(n_along, n_across, period)
    vertices, edges = core.placed(topology, places, shifts)
    areas, sides, gradients, cells = core.triangles(topology, edges)
    temperature, salinity = (mv.scalar(values[:, None]).field() for values in layered(places, disturbance))
    frames = core.fingering(topology, areas, sides, gradients, cells, topology.interior[0], density, prandtl, expansion,
                            np.array(expansion / ratio), salt_diffusivity, rayleigh * prandtl / expansion, temperature,
                            salinity, dt, n_steps, stride)
    return topology, vertices, temperature, salinity, islice(frames, n_flat, None)


def growth() -> core.Scalar:
    """The growth rate of a small disturbance of evenly stratified water at each density ratio `[ratios] Scalar+`: the
    logarithm of its stream function's growth between two readings, over the time between them."""
    topology, places, shifts = core.tank(growth_along, growth_across, growth_period)
    vertices, edges = core.placed(topology, places, shifts)
    areas, sides, gradients, cells = core.triangles(topology, edges)
    even = mv.scalar((places[:, 1] - 0.5)[:, None]).field()                    # Scalar+[vertices]
    # A disturbance of the salinity shaped like the first rolls, a wavelength along and half a wave up.
    rolls = growth_disturbance * np.sin(2 * np.pi * places[:, 0] / growth_period) * np.sin(np.pi * places[:, 1])
    frames = list(core.fingering(topology, areas, sides, gradients, cells, topology.interior[0], density, prandtl,
                                 expansion, expansion / growth_ratios, salt_diffusivity, rayleigh * prandtl / expansion,
                                 even, even + mv.scalar(rolls[:, None]).field(), growth_dt, growth_steps,
                                 growth_stride))
    # The stream function's squared size over the tank at the middle and at the end; it grows at twice the rate.
    sizes = [(stream * cells * stream).sites.sum() for stream, _, _ in frames[1:]]
    return (sizes[1].log() - sizes[0].log()) / (2 * growth_stride * growth_dt)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.saltfingers import render

    topology, vertices, temperature, salinity, frames = fingers()
    save_figure(render.draw_tank(topology, vertices, n_across, period, temperature, salinity), "saltfingers_tank")
    save_animation(render.animate(topology, vertices, n_across, period, frames), "saltfingers", frame_duration_ms)
    save_figure(render.draw_growth(growth_ratios, growth(), salt_diffusivity), "saltfingers_growth")


if __name__ == "__main__":
    main()
