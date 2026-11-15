"""Scenes for spherical synchronization gather a golden-angle population of directions and a circle grid
under one common Möbius transformation."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import concatenate, stack
from examples.math.spherical_synchronization import core
from examples.math.spherical_synchronization.core import Cut, Point, mv, time

# Population size spread over the sphere by the golden angle, plus meridian and parallel circle counts.
n_particles = 192
n_meridians = 4
n_parallels = 5
# Integration span, step count, coupling strength, shared rotation speed, initial boost, and frame delay.
span = 5.0
n_steps = 150
dt = span / n_steps
coupling = 1.2
rotation_speed = 0.25
initial_bias = 0.24
duration_ms = 40


# --- math -----------------------------------------------------------------------------
def synchronization(n_particles: int, n_steps: int) -> Iterator[tuple[Point, Cut]]:
    """Yields the population directions (`[n_particles] Point-`) and circle grid (`[n_cuts] Cut+`) across
    `n_steps + 1` frames under the population's shared Möbius motion."""
    n_cuts = n_meridians + n_parallels
    golden_angle = np.pi * (3 - np.sqrt(5))
    longitudes = golden_angle * np.arange(n_particles) % (2 * np.pi)
    latitudes = np.arcsin(np.linspace(-1 + 1 / n_particles, 1 - 1 / n_particles, n_particles))
    seeds = core.sphere_points(longitudes, latitudes) + time           # [n_particles] Point-
    # Great circles through the poles and latitude parallels are cuts whose sphere points pair to zero.
    meridian_turns = (-mv.xy * np.linspace(0, np.pi, n_meridians, endpoint=False) / 2).exp()   # [n_meridians] Rotor+
    meridians = meridian_turns >> mv.y                                 # [n_meridians] Cut+
    parallels = mv.z + mv.t * np.sin(np.linspace(-np.pi / 3, np.pi / 3, n_parallels))          # [n_parallels] Cut+
    grid = concatenate((meridians, parallels), axis=0)                 # [n_cuts] Cut+
    rotation = -mv.xy * rotation_speed                                 # [] Bivector+
    initial = (-mv.xt * initial_bias).exp()                            # [] Rotor+
    for rotor in islice(core.motion(initial, seeds, rotation, coupling, dt), n_steps + 1):
        # One rotor carries the population and the circle grid together.
        yield core.directions(rotor >> seeds), rotor >> grid


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.math.spherical_synchronization import render

    recorded: list[tuple[Point, Cut]] = []

    def streamed_states() -> Iterator[tuple[Point, Cut]]:
        for state in synchronization(n_particles, n_steps):
            recorded.append(state)
            yield state

    save_animation(render.frames(streamed_states()), "spherical_synchronization", duration_ms)

    # checks
    # Every direction stays a unit direction on the sphere, and the population gathers.
    population = stack([places for places, _ in recorded])             # [n_steps + 1, n_particles] Point-
    assert ((population + time).squared().abs() < 1e-10).all()
    spread = (population - population.mean(axis=-1)[:, None]).squared().mean(axis=-1).abs()   # [n_steps + 1] Scalar+
    assert bool(spread[-1] < 0.05 * spread[0])


if __name__ == "__main__":
    main()
