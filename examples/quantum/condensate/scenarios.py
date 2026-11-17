"""Scenario parameters and entry points for a rotating Bose-Einstein condensate relaxing from a noise-stirred Gaussian
cloud into a triangular Abrikosov lattice of quantized vortices.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from examples.quantum.condensate import core
from examples.util import shapes
from numga.topology import Topology

mv = core.mv

# Trap radius in oscillator lengths, concentric triangle rings, contact repulsion, and rotation rate.
radius, n_rings = 8.0, 48
repulsion, rotation = 200.0, 0.9
# Initial Gaussian width, amplitude and phase noise scales, and PRNG seed.
cloud_width, amplitude_noise, angle_noise, seed = 4.5, 0.05, 0.3, 0
# Imaginary-time step, total steps to the settled lattice, stride between frames, and drawn prefix length.
step, n_steps, stride = 0.05, 1200, 6
n_drawn_frames = n_steps // 2 // stride + 1
frame_duration_ms = 40
n_animation_colours = 256
# Radial bins for enclosed vortex counts and winding threshold identifying a vortex triangle.
n_radii = 61
counting_radii = np.linspace(0.0, 6.0, n_radii)                               # [n_radii]
vortex_threshold = 0.5


def trap_disk(n_disk_rings: int) -> tuple[Topology, np.ndarray]:
    """A concentric triangulated disk of radius `radius` with `n_disk_rings` rings provides the trap topology and
    vertex coordinates (`[V, 2]`)."""
    return shapes.concentric_disk(np.array([radius]), n_disk_rings)


def initial_cloud(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A broad Gaussian amplitude profile (`[V]`) and small random phase angles (`[V]`) stir the initial cloud across
    the vertex coordinates (`[V, 2]`)."""
    rng = np.random.default_rng(seed)
    distances_squared = (coordinates ** 2).sum(axis=-1)                       # [V]
    amplitudes = np.exp(-distances_squared / (2 * cloud_width ** 2)) * (
        1 + amplitude_noise * rng.standard_normal(len(coordinates))
    )                                                                         # [V]
    angles = angle_noise * rng.standard_normal(len(coordinates))              # [V]
    return amplitudes, angles


def trap_relaxation(n_disk_rings: int, n_relaxation_steps: int,
                    frame_stride: int) -> tuple[Topology, core.Point, core.Scalar, Iterator[core.Wave]]:
    """The trap geometry (`Point-[V]`, `Scalar+[V]`) and initial stirred cloud feed the imaginary-time relaxation
    stream (`Wave+[V]`)."""
    topology, coordinates = trap_disk(n_disk_rings)
    vertices = (core.horizontal * coordinates[:, 0] + core.vertical * coordinates[:, 1]).field()   # Point-[V]
    trap_potential = mv.scalar((0.5 * (coordinates ** 2).sum(axis=-1))[:, None]).field()           # Scalar+[V]
    amplitudes, angles = initial_cloud(coordinates)
    phase_rotors = (mv.ab * (-angles / 2)).exp()                              # [V] Even+
    initial_wave = (amplitudes * (phase_rotors >> mv.a)).field()              # Wave+[V]
    areas, gradients, dual_areas = core.triangle_geometry(topology, vertices)
    states = core.relaxation_states(
        topology, vertices, areas, gradients, dual_areas, trap_potential,
        repulsion, rotation, initial_wave, step, n_relaxation_steps, frame_stride,
    )
    return topology, vertices, dual_areas, states


def main() -> None:
    from examples.quantum.condensate import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, dual_areas, states = trap_relaxation(n_rings, n_steps, stride)
    drawn_states: list[core.Wave] = []

    def drawn_prefix() -> Iterator[core.Wave]:
        for wave in islice(states, n_drawn_frames):
            drawn_states.append(wave)
            yield wave

    save_animation(render.wave_animation(topology, vertices, drawn_prefix()), "condensate",
                   frame_duration_ms, colors=n_animation_colours)
    settled_states = drawn_states + list(states)
    settled_wave = settled_states[-1]                                         # Wave+[V]

    counts = core.vortex_counts(topology, vertices, settled_wave, counting_radii, vortex_threshold)   # [n_radii]
    feynman_counts = rotation * counting_radii ** 2                           # [n_radii]
    save_figure(render.count_figure(counting_radii, counts, feynman_counts), "condensate_count")

    # checks
    atom_count = (dual_areas * settled_wave.scalar_norm_squared()).sites.sum()   # [] Scalar+
    assert (atom_count - 1).abs() < 1e-10
    turns = core.triangle_windings(topology, settled_wave).scalar_product(~mv.ab)   # Scalar+[F]
    vortices = (turns > vortex_threshold) | (turns < -vortex_threshold)       # [F]
    assert ((turns + 1).abs() < 1e-6)[vortices].all()
    assert (turns.abs() < 1e-6)[~vortices].all()
    bulk = counting_radii == 4.0                                              # [n_radii]
    assert np.allclose(counts[bulk], feynman_counts[bulk], rtol=0.2)


def pinned() -> None:
    """Regenerates the gallery animation of the stirred condensate relaxing into its vortex lattice."""
    from examples.quantum.condensate import render
    from examples.util.animation import pin_animation

    topology, vertices, _, states = trap_relaxation(n_rings, n_steps, stride)
    drawn = islice(states, n_drawn_frames)
    pin_animation(render.wave_animation(topology, vertices, drawn), "condensate",
                  frame_duration_ms, colors=n_animation_colours)


if __name__ == "__main__":
    main()
