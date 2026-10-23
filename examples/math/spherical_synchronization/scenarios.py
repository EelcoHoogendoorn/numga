"""Scenes for spherical synchronization: a population of directions gathering under one Möbius motion,
and a grid of circles carried by the same motion."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import concatenate, stack
from examples.math.spherical_synchronization import core
from examples.math.spherical_synchronization.core import Cut, Point, mv, time

# The population, spread over the sphere by the golden angle, and the circles of the grid: meridians, and
# parallels at latitudes up to a third of a turn either way.
n_particles = 192
n_meridians = 4
n_parallels = 5
# The steps and their size, the strength of the attraction, the shared rotation's speed, the boost of the
# starting transformation, and the duration of a frame.
n_steps = 100
dt = 0.05
coupling = 1.2
rotation_speed = 0.25
initial_bias = 0.24
duration_ms = 60


# --- math -----------------------------------------------------------------------------
def synchronization() -> tuple[Point, Cut]:
    """The population's directions and the grid's cuts at every step `[steps + 1, ...]`, both carried by
    the one transformation the population's own attraction drives."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    longitudes = golden_angle * np.arange(n_particles) % (2 * np.pi)
    latitudes = np.arcsin(np.linspace(-1 + 1 / n_particles, 1 - 1 / n_particles, n_particles))
    seeds = core.sphere_points(longitudes, latitudes) + time           # [particles] Point
    # Great circles through the poles, and parallels: the cuts their points pair to nothing with.
    meridians = (-mv.xy * np.linspace(0, np.pi, n_meridians, endpoint=False) / 2).exp() >> mv.y   # [meridians] Cut
    parallels = mv.z + mv.t * np.sin(np.linspace(-np.pi / 3, np.pi / 3, n_parallels))             # [parallels] Cut
    grid = concatenate((meridians, parallels), axis=0)               # [circles] Cut
    rotation = -mv.xy * rotation_speed                              # [] Bivector
    initial = (-mv.xt * initial_bias).exp()                         # [] Rotor
    rotors = stack(list(islice(core.motion(initial, seeds, rotation, coupling, dt), n_steps + 1)))   # [steps + 1] Rotor
    # One rotor carries the population and the grid alike.
    return core.directions(rotors[:, None] >> seeds), rotors[:, None] >> grid


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.math.spherical_synchronization import render

    population, circles = synchronization()
    save_animation(render.frames(population, circles), "spherical_synchronization", duration_ms)

    # checks
    # Every direction stays a unit direction on the sphere, and the population gathers.
    np.testing.assert_allclose((population + time).squared().to_array(), 0, atol=1e-10)
    spread = (population - population.mean(axis=-1)[:, None]).squared().mean(axis=-1).to_array()   # [steps + 1]
    assert abs(spread[-1]) < 0.05 * abs(spread[0])


if __name__ == "__main__":
    main()
