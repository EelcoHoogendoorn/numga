"""A rotating Bose-Einstein condensate settling into its lattice of vortices, from a still cloud stirred by noise.

Feynman's rule for the vortices' density, twice the rotation rate over the circulation quantum, is as Donnelly (2009)
gives it for a rotating bucket of superfluid helium.
"""

from __future__ import annotations

import numpy as np

from examples.quantum.condensate import core
from examples.util import shapes

mv = core.mv

# The trap, in oscillator lengths: its radius and the rings of triangles across it; the atoms' repulsion and the
# rotation rate, in the trap's frequency.
radius, n_rings = 8.0, 48
repulsion, rotation = 200.0, 0.9
# The still cloud's width, the noise stirring its amplitude and its angle, and the noise's seed.
cloud_width, amplitude_noise, angle_noise, seed = 4.5, 0.05, 0.3, 0
# The descent's step, its steps and the steps between frames; the frames' duration in milliseconds.
step, n_steps, stride = 0.05, 1200, 12
frame_duration_ms = 40
# The radii within which the vortices are counted, and the share of a whole turn that makes a triangle a vortex.
counting_radii = np.linspace(0.0, 6.0, 61)
vortex_share = 0.5


def trap() -> tuple[core.Topology, np.ndarray]:
    """The trap's disk of triangles in rings: its topology and its places across and up `[places, 2]`."""
    return shapes.concentric_disk(np.array([radius]), n_rings)


def still_cloud(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A still cloud's amplitude and angle at each place `[places]` each: a broad bell whose wave points nearly the same
    way everywhere, stirred by noise in both."""
    rng = np.random.default_rng(seed)
    distances_squared = (coordinates ** 2).sum(axis=-1)
    amplitudes = np.exp(-distances_squared / (2 * cloud_width ** 2)) * (1 + amplitude_noise * rng.standard_normal(len(coordinates)))
    return amplitudes, angle_noise * rng.standard_normal(len(coordinates))


def settled():
    """The trap's topology, its places `Point-[places]`, and the condensate at each frame of its descent `[frames]
    Wave+[places]`."""
    topology, coordinates = trap()
    places = (core.axis * coordinates[:, 0] + core.across * coordinates[:, 1]).field()   # Point-[places]
    potential = mv.scalar((0.5 * (coordinates ** 2).sum(axis=-1))[:, None]).field()   # Scalar+[places]
    amplitudes, angles = still_cloud(coordinates)
    wave = (amplitudes * ((mv.ab * (-angles / 2)).exp() >> mv.a)).field()    # Wave+[places]
    areas, gradients, cells = core.triangles(topology, places)
    return topology, places, core.settling(topology, places, areas, gradients, cells, potential, repulsion, rotation,
                                           wave, step, n_steps, stride)


def counted(topology: core.Topology, places: core.Point, wave: core.Wave) -> np.ndarray:
    """The vortices within each counting radius `[radii]`: the triangles around which the wave winds a whole turn, by the
    distance of their middles from the trap's middle."""
    middles = topology.incidence[2, 0].mean(axis=0) * places                  # Point-[faces]
    # Each triangle's winding in whole turns, read by the condensate's plane.
    turns = core.windings(topology, wave).scalar_product(~mv.ab)              # Scalar+[faces]
    vortices = (turns > vortex_share) | (turns < -vortex_share)               # [faces]
    return ((middles.norm() <= counting_radii[:, None]) & vortices).sum(axis=-1).reshape(len(counting_radii))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.condensate import render

    topology, places, frames = settled()
    frames = list(frames)
    save_animation(render.animate(topology, places, (wave.scalar_norm_squared() for wave in frames)), "condensate",
                   frame_duration_ms)
    save_figure(render.draw_count(counting_radii, counted(topology, places, frames[-1]),
                                  rotation * counting_radii ** 2), "condensate_count")


if __name__ == "__main__":
    main()
