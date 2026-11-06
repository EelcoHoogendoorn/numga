"""A rotating Bose-Einstein condensate settling into its lattice of vortices, from a still cloud stirred by noise.

Feynman's rule for the vortices' density, twice the rotation rate over the circulation quantum, is as Donnelly (2009)
gives it for a rotating bucket of superfluid helium.
"""

from __future__ import annotations

from itertools import islice

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
# The descent's step, its steps to the settled lattice the vortices are counted in, and the steps between frames; the
# frames drawn, over the first half of the descent, where the vortices come in and draw together into the lattice and
# then barely move; the frames' duration in milliseconds, and the colours of their palette, the density map's one hue
# family in a few dozen shades.
step, n_steps, stride = 0.05, 1200, 6
n_drawn_frames = n_steps // 2 // stride + 1
frame_duration_ms = 40
n_animation_colours = 64
# The radii within which the vortices are counted, and the share of a whole turn that makes a triangle a vortex.
counting_radii = np.linspace(0.0, 6.0, 61)
vortex_share = 0.5


def trap(n_rings: int = n_rings) -> tuple[core.Topology, np.ndarray]:
    """The trap's disk of triangles in rings: its topology and its vertices across and up `[V, 2]`."""
    return shapes.concentric_disk(np.array([radius]), n_rings)


def still_cloud(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A still cloud's amplitude and angle at each vertex `[V]` each: a broad bell whose wave points nearly the
    same way everywhere, stirred by noise in both."""
    rng = np.random.default_rng(seed)
    distances_squared = (coordinates ** 2).sum(axis=-1)
    amplitudes = np.exp(-distances_squared / (2 * cloud_width ** 2)) * (1 + amplitude_noise * rng.standard_normal(len(coordinates)))
    return amplitudes, angle_noise * rng.standard_normal(len(coordinates))


def settled(n_rings: int = n_rings, n_steps: int = n_steps, stride: int = stride):
    """The trap's topology, its vertices `Point-[V]`, and the condensate at each frame of its descent `[frames]
    Wave+[V]`."""
    topology, coordinates = trap(n_rings)
    vertices = (core.axis * coordinates[:, 0] + core.across * coordinates[:, 1]).field()   # Point-[V]
    potential = mv.scalar((0.5 * (coordinates ** 2).sum(axis=-1))[:, None]).field()   # Scalar+[V]
    amplitudes, angles = still_cloud(coordinates)
    wave = (amplitudes * ((mv.ab * (-angles / 2)).exp() >> mv.a)).field()    # Wave+[V]
    areas, gradients, cells = core.triangles(topology, vertices)
    return topology, vertices, core.settling(topology, vertices, areas, gradients, cells, potential, repulsion,
                                             rotation, wave, step, n_steps, stride)


def counted(topology: core.Topology, vertices: core.Point, wave: core.Wave) -> np.ndarray:
    """The vortices within each counting radius `[radii]`: the triangles around which the wave winds a whole turn, by the
    distance of their middles from the trap's middle."""
    middles = topology.incidence[2, 0].mean(axis=0) * vertices                # Point-[F]
    # Each triangle's winding in whole turns, read by the condensate's plane.
    turns = core.windings(topology, wave).scalar_product(~mv.ab)              # Scalar+[F]
    vortices = (turns > vortex_share) | (turns < -vortex_share)               # [F]
    return ((middles.norm() <= counting_radii[:, None]) & vortices).sum(axis=-1).reshape(len(counting_radii))


def main() -> None:
    from examples.util.animation import save_figure
    from examples.quantum.condensate import render

    topology, vertices, frames = settled()
    frames = list(frames)
    save_figure(render.draw_count(counting_radii, counted(topology, vertices, frames[-1]),
                                  rotation * counting_radii ** 2), "condensate_count")


def pinned() -> None:
    """The pictures the repository tracks: the stirred cloud settling into its lattice of vortices, for the gallery."""
    from examples.util.animation import pin_animation
    from examples.quantum.condensate import render

    topology, vertices, frames = settled()
    # The frames over the vortices' coming in and settling.
    drawn = islice(frames, n_drawn_frames)
    pin_animation(render.animate(topology, vertices, (wave.scalar_norm_squared() for wave in drawn)), "condensate",
                  frame_duration_ms, colors=n_animation_colours)


if __name__ == "__main__":
    main()
