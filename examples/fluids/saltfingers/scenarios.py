"""Salt fingers: warm salty water over cold fresh water fingering across the interface between them."""

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
# still flat, not drawn; the disturbance's size and its seed; the frames' duration in milliseconds, and the colours of
# their palette.
rayleigh, ratio = 5e5, 2.0
period, n_along, n_across = 2.0, 136, 69
thickness = 0.05
dt, n_steps, n_stride, n_flat = 6e-5, 360, 2, 34
disturbance, seed = 4e-2, 1
frame_duration_ms = 40
n_animation_colours = 32


def layered(heights: core.Scalar, size: float) -> tuple[core.Scalar, core.Scalar]:
    """The temperature and the salinity at vertices of these heights `Scalar+[vertices]`, each a half above the middle's
    at the top and below at the bottom, changing across an interface of this thickness, with a disturbance of the
    salinity of this size inside the interface."""
    rng = np.random.default_rng(seed)
    noise = mv.scalar(rng.standard_normal((len(heights.batch()), 1))).field()   # Scalar+[vertices]
    profile = ((heights - 0.5) / thickness).tanh()                            # Scalar+[vertices]
    return profile / 2, profile / 2 + size * (1 - profile.squared()) * noise


def fingers(n_along: int = n_along, n_across: int = n_across, n_flat: int = n_flat):
    """The tank, `n_along` vertices along and `n_across` up, closed and opened at its seam, with the map reading the
    closed tank's fields on the open one, the open one's vertices `Point-[vertices]`, the temperature and salinity at
    the start `Scalar+[vertices]`, the two layers at rest, and the tank's stream function, temperature and salinity
    over time `Scalar+[vertices]` each, after the first `n_flat` frames, from just before the fingers show."""
    closed, opened, closing, opening, places = core.tank(n_along, n_across, period)
    vertices = (core.volume + mv(core.Upright, places).dual()).field()          # Point-[opened vertices]
    water = core.forms(closed, opened, vertices, density, prandtl, salt_diffusivity)
    temperature, salinity = layered(closing * (mv.y & vertices), disturbance)  # Scalar+[vertices], Scalar+[vertices]
    frames = core.fingering(water, closed.reading(closed.interior[0], 0), density, expansion, np.array(expansion / ratio),
                            rayleigh * prandtl / expansion, temperature, salinity, dt, n_steps, n_stride)
    return closed, opened, opening, vertices, temperature, salinity, islice(frames, n_flat, None)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.saltfingers import render

    closed, opened, opening, vertices, temperature, salinity, frames = fingers()
    save_figure(render.draw_tank(opened, vertices, opening, period, temperature, salinity), "saltfingers_tank")
    save_animation(render.animate(opened, vertices, opening, period, frames), "saltfingers", frame_duration_ms,
                   colors=n_animation_colours)


if __name__ == "__main__":
    main()
