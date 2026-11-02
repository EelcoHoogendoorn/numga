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
dt, n_steps, n_stride, n_flat = 2e-5, 1080, 6, 34
disturbance, seed = 4e-2, 1
frame_duration_ms = 40
# The growth: the density ratios swept; the wavenumber of the first rolls to grow between rigid walls, in inverse
# depths, and the tank one wavelength of them long, with its vertices along and up; the disturbance's size, the time
# step, the steps, and the steps between the two readings of the growth.
growth_ratios = np.array([6, 7, 8, 9, 10, 11, 12])
critical_wavenumber = 3.117
growth_period, n_growth_along, n_growth_across = 2 * np.pi / critical_wavenumber, 48, 25
growth_disturbance = 1e-6
growth_dt, n_growth_steps, n_growth_stride = 5e-4, 400, 200


def layered(heights: core.Scalar, size: float) -> tuple[core.Scalar, core.Scalar]:
    """The temperature and the salinity at vertices of these heights `Scalar+[vertices]`, each a half above the middle's
    at the top and below at the bottom, changing across an interface of this thickness, with a disturbance of the
    salinity of this size inside the interface."""
    rng = np.random.default_rng(seed)
    noise = heights.context.multivector.scalar(rng.standard_normal((len(heights.batch()), 1))).field()   # Scalar+[vertices]
    profile = ((heights - 0.5) / thickness).tanh()                            # Scalar+[vertices]
    return profile / 2, profile / 2 + size * (1 - profile.squared()) * noise


def fingers():
    """The tank, closed and opened at its seam, with the map reading the closed tank's fields on the open one, the open
    one's vertices `Point-[vertices]`, the temperature and salinity at the start `Scalar+[vertices]`, the two layers at
    rest, and the tank's stream function, temperature and salinity over time `Scalar+[vertices]` each, from just before
    the fingers show."""
    closed, opened, closing, opening, places = core.tank(n_along, n_across, period)
    vertices = (core.volume + mv(core.Upright, places).dual()).field()          # Point-[opened vertices]
    areas, sides, gradients = core.triangles(opened, vertices)
    temperature, salinity = layered(closing * (mv.y & vertices), disturbance)  # Scalar+[vertices], Scalar+[vertices]
    frames = core.fingering(closed, areas, sides, gradients, closed.interior[0], density, prandtl, expansion,
                            np.array(expansion / ratio), salt_diffusivity, rayleigh * prandtl / expansion, temperature,
                            salinity, dt, n_steps, n_stride)
    return closed, opened, opening, vertices, temperature, salinity, islice(frames, n_flat, None)


def growth() -> core.Scalar:
    """The growth rate of a small disturbance of evenly stratified water at each density ratio `[ratios] Scalar+`: the
    logarithm of its stream function's growth between two readings, over the time between them."""
    closed, opened, closing, _, places = core.tank(n_growth_along, n_growth_across, growth_period)
    vertices = (core.volume + mv(core.Upright, places).dual()).field()          # Point-[opened vertices]
    areas, sides, gradients = core.triangles(opened, vertices)
    heights, runs = closing * (mv.y & vertices), closing * (mv.x & vertices)    # Scalar+[vertices], Scalar+[vertices]
    stratification = heights - 0.5                                            # Scalar+[vertices]
    # A disturbance of the salinity shaped like the first rolls, a wavelength along and half a wave up.
    rolls = growth_disturbance * (critical_wavenumber * runs).sin() * (np.pi * heights).sin()   # Scalar+[vertices]
    frames = list(core.fingering(closed, areas, sides, gradients, closed.interior[0], density, prandtl, expansion,
                                 expansion / growth_ratios, salt_diffusivity, rayleigh * prandtl / expansion,
                                 stratification, stratification + rolls, growth_dt, n_growth_steps, n_growth_stride))
    # The stream function's squared size over the tank at the middle and at the end; it grows at twice the rate.
    cells = (~closed.incidence[2, 0] * (areas / 3)).sum(axis=0)                # Scalar+[vertices]
    sizes = [(cells * stream.squared()).sites.sum() for stream, _, _ in frames[1:]]   # [readings] [ratios] Scalar+
    return (sizes[1].log() - sizes[0].log()) / (2 * n_growth_stride * growth_dt)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.saltfingers import render

    closed, opened, opening, vertices, temperature, salinity, frames = fingers()
    save_figure(render.draw_tank(opened, vertices, opening, period, temperature, salinity), "saltfingers_tank")
    save_animation(render.animate(opened, vertices, opening, period, frames), "saltfingers", frame_duration_ms)
    save_figure(render.draw_growth(growth_ratios, growth(), salt_diffusivity), "saltfingers_growth")


if __name__ == "__main__":
    main()
