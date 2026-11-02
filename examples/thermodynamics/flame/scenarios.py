"""A cellular flame: sparks in a lean mixture whose fuel spreads faster than heat, the flames spreading from
them merging into one front that breaks into cells as it grows."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from examples.thermodynamics.flame import core
from examples.util import shapes

mv = core.mv

# The disk of mixture, in flame thicknesses: its radius and the rings of triangles across it.
radius, n_rings = 28.0, 187
# The mixture: its Lewis number, heat's diffusivity over fuel's, about lean hydrogen's in air; its Zeldovich number,
# how steeply burning grows with temperature near the burnt mixture's; and the unburnt mixture's temperature, in units
# of the rise burning brings.
lewis, zeldovich, unburnt_temperature = 0.3, 12.0, 0.25
# Arrhenius's law in those units: the activation temperature, and the prefactor that makes a flat flame's speed one
# where burning grows steeply.
burnt_temperature = unburnt_temperature + 1
activation = zeldovich * burnt_temperature ** 2
prefactor = zeldovich ** 2 / (2 * lewis) * np.exp(activation / burnt_temperature)
# The sparks: their places across and up and their radii, a spark lit earlier grown larger; the width of their edges,
# the noise roughening those edges, and the noise's seed.
spark_places = np.array([[-5.5, -2.5], [4.0, -4.5], [1.5, 5.5]])
spark_radii = np.array([3.0, 2.2, 1.5])
kernel_edge, roughness, seed = 0.5, 1.0, 0
# The time step, short enough that each step burns only a share of a hot parcel's fuel, and inside the limit the fuel's
# explicit spreading sets on triangles this small; the frames, until the cells near the rim, the steps between them,
# and the frames while the noise on the sparks' edges smooths out, run but not shown; the frames' duration in
# milliseconds.
dt, n_frames, n_stride, n_unshown = 0.001, 101, 83, 2
frame_duration_ms = 40


def disk() -> tuple[core.Topology, np.ndarray]:
    """The disk of mixture, triangles in staggered rings: its topology and its vertices' places across and up
    `[vertices, 2]`."""
    return shapes.concentric_disk(np.array([radius]), n_rings)


def hot_spot(coordinates: np.ndarray) -> np.ndarray:
    """The share of the mixture burnt at the start at each place `[vertices]`: all of it within any of the sparks, their
    edges roughened by noise, and none around them."""
    rng = np.random.default_rng(seed)
    edges = spark_radii + roughness * rng.standard_normal((len(coordinates), 1))   # [vertices, sparks]
    reaches = np.linalg.norm(coordinates[:, None] - spark_places, axis=-1)    # [vertices, sparks]
    return (0.5 * (1 - np.tanh((reaches - edges) / kernel_edge))).max(axis=-1)


def spreading() -> tuple[core.Topology, core.Point, Iterator[tuple[core.Point, core.Scalar]]]:
    """The disk's topology and vertices `Point-[vertices]`, and the mixture's parcels `Point-[vertices]` and the rate
    their fuel burns at `Scalar+[vertices]` at each frame, as the flame spreads from the sparks, once the noise on their
    edges has smoothed out."""
    topology, coordinates = disk()
    vertices = (core.volume + mv(core.Upright, coordinates).dual()).field()  # Point-[vertices]
    areas, gradients, cells = core.triangles(topology, vertices)
    burnt = mv.scalar(hot_spot(coordinates)[:, None]).field()                 # Scalar+[vertices]
    parcels = core.volume + core.heat * burnt + core.fuel * (1 - burnt)      # Point-[vertices]
    frames = core.burning(topology, areas, gradients, cells, 1 / lewis, prefactor, activation, unburnt_temperature,
                          parcels, dt, n_frames, n_stride)
    return topology, vertices, islice(frames, n_unshown, None)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.thermodynamics.flame import render

    topology, vertices, frames = spreading()
    save_animation(render.animate(topology, vertices, radius, core.energy, frames), "flame", frame_duration_ms)


if __name__ == "__main__":
    main()
