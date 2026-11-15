"""Salt fingers: warm salty water over cold fresh water fingering across the interface between them."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.fluids.saltfingers import core, plumbing

mv = core.mv

# Reference density, Prandtl number, thermal expansion coefficient, and salt-to-heat diffusivity ratio.
density, prandtl, expansion = 1.0, 7.0, 1e-3
salt_diffusivity = 0.1
# Thermal Rayleigh number, density ratio, tank period and grid resolution, interface thickness, stepping schedule, and seed noise.
rayleigh, ratio = 5e5, 2.0
period, n_along, n_across = 2.0, 136, 69
thickness = 0.05
dt, n_steps, n_stride, n_flat = 6e-5, 360, 2, 34
disturbance, seed = 4e-2, 1
frame_duration_ms = 40
n_animation_colours = 32


def layer_profiles(heights: core.Scalar, noise_scale: float) -> tuple[core.Scalar, core.Scalar]:
    """Two-layer hyperbolic-tangent temperature and noisy salinity profiles across the mid-depth interface."""
    rng = np.random.default_rng(seed)
    noise = mv.scalar(rng.standard_normal((len(heights.batch()), 1))).field()  # Scalar+[V]
    profile = ((heights - 0.5) / thickness).tanh()                            # Scalar+[V]
    return profile / 2, profile / 2 + noise_scale * (1 - profile.squared()) * noise


def fingers(n_along: int = n_along, n_across: int = n_across, n_flat: int = n_flat):
    """Tank topologies, seam reading, opened vertices, initial layer profiles, and streaming states past the flat transient."""
    closed, opened, closing, opening, places = plumbing.tank(n_along, n_across, period)
    vertices = (mv(core.Upright, places) + mv.w).dual().field()               # Point-[V_opened]
    operators = core.tank_operators(closed, opened, vertices, density, prandtl, salt_diffusivity)
    temperature, salinity = layer_profiles(closing * (mv.y & vertices), disturbance)  # Scalar+[V], Scalar+[V]
    frames = core.fingering(
        operators,
        closed.reading(closed.interior[0], 0),
        density,
        expansion,
        expansion / ratio,
        rayleigh * prandtl / expansion,
        temperature,
        salinity,
        dt,
        n_steps,
        n_stride,
    )
    return closed, opened, opening, vertices, temperature, salinity, islice(frames, n_flat, None)


def main() -> None:
    from examples.fluids.saltfingers import render
    from examples.util.animation import save_animation, save_figure

    closed, opened, opening, vertices, temperature, salinity, frames = fingers()
    save_figure(render.draw_tank(opened, vertices, opening, period, temperature, salinity), "saltfingers_tank")
    kept = []

    def passing():
        for state in frames:
            kept.append(state)
            yield state

    save_animation(
        render.animate(opened, vertices, opening, period, passing()),
        "saltfingers",
        frame_duration_ms,
        colors=n_animation_colours,
    )

    # checks
    pull = -core.mv.ym                                                        # [] Drive+
    for part, expected in ((core.mass, -1.0), (core.volume, 0.0), (core.entropy, 0.0), (core.salt, 0.0)):
        assert (((core.mv.y & pull.commutator(part)) - expected).abs() < 1e-12).all()
    middle = ((mv.y & vertices.batch()) - 0.5).abs() < 1e-12
    _, last_temperature, last_salinity = kept[-1]
    salinities = (opening * last_salinity).batch()[middle]
    temperatures = (opening * last_temperature).batch()[middle]
    assert bool(salinities[salinities.argmax()] - salinities[salinities.argmin()] > 0.9)
    assert bool((temperatures - temperatures.mean()).squared().mean() < (salinities - salinities.mean()).squared().mean() / 4)


if __name__ == "__main__":
    main()
