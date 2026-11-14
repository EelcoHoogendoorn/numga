"""Three sparks in a lean fuel-air mixture spread and merge into a cellular flame front on a triangulated disc."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from examples.thermodynamics.flame import core
from examples.util import shapes

mv = core.mv

# Disc radius in flame thicknesses and radial ring count of the triangulation.
radius, n_rings = 28.0, 187
# Lewis number (thermal over fuel diffusivity, near lean hydrogen in air), Zeldovich number, and unburnt temperature in
# units of the adiabatic temperature rise.
lewis, zeldovich, unburnt_temperature = 0.3, 12.0, 0.25
# Activation temperature and prefactor normalizing the flat-flame speed to unity in the high-activation-energy limit.
burnt_temperature = unburnt_temperature + 1
activation = zeldovich * burnt_temperature ** 2
prefactor = zeldovich ** 2 / (2 * lewis) * np.exp(activation / burnt_temperature)
# Initial spark centres, radii, smooth rim width, radial noise amplitude, and random seed.
spark_places = np.array([[-5.5, -2.5], [4.0, -4.5], [1.5, 5.5]])
spark_radii = np.array([3.0, 2.2, 1.5])
kernel_edge, roughness, seed = 0.5, 1.0, 0
# Time step within the explicit fuel-diffusion limit, total frames, steps per frame, initial smoothing frames omitted,
# and frame duration in milliseconds.
dt, n_frames, n_stride, n_unshown = 0.001, 101, 83, 2
frame_duration_ms = 40


def disc(n_rings: int) -> tuple[core.Topology, np.ndarray]:
    """Triangulated disc of staggered rings and its planar vertex coordinates."""
    return shapes.concentric_disk(np.array([radius]), n_rings)


def hot_spot(coordinates: np.ndarray) -> core.Scalar:
    """Initial burnt fraction on the vertices, equal to one inside the noise-roughened sparks and zero outside."""
    rng = np.random.default_rng(seed)
    edges = spark_radii + roughness * rng.standard_normal((len(coordinates), 1))   # [V, 3]
    reaches = np.linalg.norm(coordinates[:, None] - spark_places, axis=-1)    # [V, 3]
    burnt_fraction = (0.5 * (1 - np.tanh((reaches - edges) / kernel_edge))).max(axis=-1)   # [V]
    return mv.scalar(burnt_fraction[:, None]).field()


def flame_states(n_rings: int, n_frames: int,
                 n_stride: int) -> tuple[core.Topology, core.Point, Iterator[tuple[core.Point, core.Scalar]]]:
    """The triangulated disc, its vertex places, and the parcel and reaction-rate fields across the rendered frames
    after the initial spark-edge noise smooths out."""
    topology, coordinates = disc(n_rings)
    vertices = (mv(core.Upright, coordinates) + mv.w).dual().field()            # Antivector-[V]
    areas, gradients, cells = core.triangles(topology, vertices)
    burnt_fraction = hot_spot(coordinates)                                    # Scalar+[V]
    parcels = core.volume + core.heat * burnt_fraction + core.fuel * (1 - burnt_fraction)   # Antivector-[V]
    frames = core.burning(topology, areas, gradients, cells, 1 / lewis, prefactor, activation, unburnt_temperature,
                          parcels, dt, n_frames, n_stride)
    return topology, vertices, islice(frames, n_unshown, None)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.thermodynamics.flame import render

    topology, vertices, frames = flame_states(n_rings, n_frames, n_stride)
    save_animation(render.animate(topology, vertices, radius, core.energy, frames), "flame", frame_duration_ms)


if __name__ == "__main__":
    main()
