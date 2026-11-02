"""A snowflake growing from a seed of ice in undercooled water, and the same seed growing with no anisotropy, which
grows no arms of its own: on a mesh of staggered rings, with no six-fold symmetry of its own, the six arms come from
the crystal's six-fold surface energy."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.snowflake import core
from examples.util import shapes
from examples.util.topology import Topology

mv = core.mv

# The dish of water's radius, and the radial intervals of its mesh, nearly equilateral triangles in staggered rings.
dish_radius, n_divisions = 4.5, 150
# The seed of ice in its middle: its radius and the width of its edge.
seed_radius, seed_edge = 0.1, 0.02
# The crystal's axis, a plane whose normal, up, one arm grows along.
crystal_axis = mv.y                                                           # [] Plane+
# The ice's relaxation time, its edge's mean width, the six-fold anisotropy of that width, and the steepness below
# which the anisotropy fades, where the ice fraction is flat.
relaxation_time, mean_width, anisotropy, regularization = 3e-4, 0.01, 0.02, 1.0
# The undercooling's tilt of the double well, its strength and its steepness; the melting temperature, with the water
# at nothing, and the latent heat's warming per unit of ice.
tilt_strength, tilt_steepness, melting_temperature, latent_heat = 0.9, 10.0, 1.0, 1.8
# The time step, the frames and the steps between them; the frames' duration in milliseconds.
dt, n_frames, stride = 2e-4, 100, 14
frame_duration_ms = 40


def dish() -> tuple[Topology, np.ndarray, core.Point]:
    """The dish of water: its topology, its vertices' places across and up `[vertices, 2]`, and its vertices, places of
    unit volume `Point-[vertices]`."""
    topology, coordinates = shapes.concentric_disk(np.array([dish_radius]), n_divisions)
    vertices = (core.volume + core.across * coordinates[:, 0] + core.up * coordinates[:, 1]).field()
    return topology, coordinates, vertices


def seed_fractions(coordinates: np.ndarray) -> np.ndarray:
    """The ice fraction at the start at places across and up `[vertices, 2]`, `[vertices]`: ice within the seed's
    radius of the middle, with a smooth edge."""
    return (1 + np.tanh((seed_radius - np.hypot(*coordinates.T)) / seed_edge)) / 2


def seeded(coordinates: np.ndarray) -> core.Point:
    """The water's parcels at the start `Point-[vertices]`: unit volume, the seed's ice, and no heat, the water at the
    cold water's temperature."""
    return (core.volume + core.ice * seed_fractions(coordinates)).field()


def frozen(topology: Topology, vertices: core.Point, parcels: core.Point, anisotropy: float, n_frames: int,
           stride: int) -> Iterator[core.Point]:
    """The water's parcels `Point-[vertices]` at each frame, from these, with this anisotropy."""
    areas, gradients, cells = core.triangles(topology, vertices)
    return core.growth(topology, areas, gradients, cells, parcels + 0 * core.heat, crystal_axis, relaxation_time,
                       mean_width, anisotropy, regularization, tilt_strength, tilt_steepness, melting_temperature,
                       latent_heat, dt, n_frames, stride)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.snowflake import render

    topology, coordinates, vertices = dish()
    parcels = seeded(coordinates)
    save_figure(render.draw_seed(topology, vertices, parcels + 0 * core.heat), "snowflake_seed")
    snowflake = render.animate(topology, vertices, frozen(topology, vertices, parcels, anisotropy, n_frames, stride))
    isotropic = render.animate(topology, vertices, frozen(topology, vertices, parcels, 0.0, n_frames, stride))
    pairs = list(zip(snowflake, isotropic))
    save_animation((flake for flake, _ in pairs), "snowflake", frame_duration_ms)
    save_animation(render.side_by_side(pairs), "snowflake_isotropic", frame_duration_ms)


if __name__ == "__main__":
    main()
