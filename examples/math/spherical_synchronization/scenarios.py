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
N_PARTICLES = 192
N_MERIDIANS = 4
N_PARALLELS = 5
# The steps and their size, the strength of the attraction, the shared rotation's speed, the boost of the
# starting transformation, and the duration of a frame.
N_STEPS = 100
DT = 0.05
COUPLING = 1.2
ROTATION_SPEED = 0.25
INITIAL_BIAS = 0.24
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def synchronization() -> tuple[Point, Cut]:
    """The population's directions and the grid's cuts at every step `[steps + 1, ...]`, both carried by
    the one transformation the population's own attraction drives."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    longitudes = golden_angle * np.arange(N_PARTICLES) % (2 * np.pi)
    latitudes = np.arcsin(np.linspace(-1 + 1 / N_PARTICLES, 1 - 1 / N_PARTICLES, N_PARTICLES))
    seeds = core.sphere_points(longitudes, latitudes) + time           # [particles] Point
    # Great circles through the poles, and parallels: the cuts their points pair to nothing with.
    meridians = (-mv.xy * np.linspace(0, np.pi, N_MERIDIANS, endpoint=False) / 2).exp() >> mv.y   # [meridians] Cut
    parallels = mv.z + mv.t * np.sin(np.linspace(-np.pi / 3, np.pi / 3, N_PARALLELS))             # [parallels] Cut
    grid = concatenate((meridians, parallels), axis=0)               # [circles] Cut
    rotation = -mv.xy * ROTATION_SPEED                              # [] Bivector
    initial = (-mv.xt * INITIAL_BIAS).exp()                         # [] Rotor
    rotors = stack(list(islice(core.motion(initial, seeds, rotation, COUPLING, DT), N_STEPS + 1)))   # [steps + 1] Rotor
    # One rotor carries the population and the grid alike.
    return core.directions(rotors[:, None] >> seeds), rotors[:, None] >> grid


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.math.spherical_synchronization import render

    population, circles = synchronization()
    save_animation(render.frames(population, circles), "spherical_synchronization", DURATION_MS)

    # checks
    # Every direction stays a unit direction on the sphere, and the population gathers.
    np.testing.assert_allclose((population + time).squared().to_array(), 0, atol=1e-10)
    spread = (population - population.mean(axis=-1)[:, None]).squared().mean(axis=-1).to_array()   # [steps + 1]
    assert abs(spread[-1]) < 0.05 * abs(spread[0])


if __name__ == "__main__":
    main()
