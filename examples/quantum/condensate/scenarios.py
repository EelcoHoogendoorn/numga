"""Scenario parameters and entry points for a Bose-Einstein condensate stirred in real time by a rotating elliptical
harmonic trap into a triangular Abrikosov lattice of quantized vortices.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.quantum.condensate import core
from examples.util import shapes
from numga.topology import Topology

mv = core.mv

# Trap radius in oscillator lengths, concentric triangle rings, contact repulsion, rotation rate, ellipticity, and damping.
radius, n_rings = 8.0, 36
repulsion, rotation, ellipticity, damping = 200.0, 0.9, 0.1, 0.2
# Initial perturbation scales and PRNG seed breaking exact two-fold symmetry of the rest cloud.
amplitude_noise, angle_noise, seed = 0.02, 0.02, 0
# Real-time step, total steps to the settled lattice, and stride between animation frames.
step, n_steps, stride = 0.04, 700, 4
n_frames = n_steps // stride + 1
frame_time = stride * step
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
    """A non-rotating Thomas-Fermi amplitude profile (`[V]`) with a small random perturbation and phase angles (`[V]`)
    seeds the cloud at rest across the vertex coordinates (`[V, 2]`)."""
    rng = np.random.default_rng(seed)
    trap = 0.5 * ((1 + ellipticity) * coordinates[:, 0] ** 2 + (1 - ellipticity) * coordinates[:, 1] ** 2)   # [V]
    chemical_potential = np.sqrt(repulsion / np.pi)
    amplitudes = np.sqrt(np.maximum(chemical_potential - trap, 0.0) / repulsion) * (
        1 + amplitude_noise * rng.standard_normal(len(coordinates))
    )                                                                         # [V]
    angles = angle_noise * rng.standard_normal(len(coordinates))              # [V]
    return amplitudes, angles


def trap_evolution(n_disk_rings: int, n_evolution_steps: int,
                   frame_stride: int) -> tuple[Topology, core.Point, core.Scalar, Iterator[core.Wave]]:
    """The elliptical trap geometry (`Point-[V]`, `Scalar+[V]`) and rest Thomas-Fermi cloud feed the real-time damped
    Gross-Pitaevskii stream (`Wave+[V]`)."""
    topology, coordinates = trap_disk(n_disk_rings)
    vertices = (core.horizontal * coordinates[:, 0] + core.vertical * coordinates[:, 1]).field()   # Point-[V]
    trap_potential = 0.5 * ((1 + ellipticity) * (mv.x & vertices).squared() +
                            (1 - ellipticity) * (mv.y & vertices).squared())   # Scalar+[V]
    amplitudes, angles = initial_cloud(coordinates)
    phase_rotors = (mv.ab * (-angles / 2)).exp()                              # [V] Even+
    raw_wave = (amplitudes * (phase_rotors >> mv.a)).field()                  # Wave+[V]
    areas, gradients, dual_areas = core.triangle_geometry(topology, vertices)
    atom_count = (dual_areas * raw_wave.scalar_norm_squared()).sites.sum()    # [] Scalar+
    initial_wave = raw_wave / atom_count.square_root()                        # Wave+[V]
    states = core.condensate_states(
        topology, vertices, areas, gradients, dual_areas, trap_potential,
        repulsion, rotation, damping, initial_wave, step, n_evolution_steps, frame_stride,
    )
    return topology, vertices, dual_areas, states


def main() -> None:
    from examples.quantum.condensate import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, dual_areas, states = trap_evolution(n_rings, n_steps, stride)
    drawn_states = []

    def recorded() -> Iterator[core.Wave]:
        for wave in states:
            drawn_states.append(wave)
            yield wave

    save_animation(render.wave_animation(topology, vertices, rotation, frame_time, recorded()), "condensate",
                   frame_duration_ms, colors=n_animation_colours)
    settled_wave = drawn_states[-1]                                           # Wave+[V]

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
    """Regenerates the gallery animation of the stirred condensate evolving into its vortex lattice."""
    from examples.quantum.condensate import render
    from examples.util.animation import pin_animation

    topology, vertices, _, states = trap_evolution(n_rings, n_steps, stride)
    pin_animation(render.wave_animation(topology, vertices, rotation, frame_time, states), "condensate",
                  frame_duration_ms, colors=n_animation_colours)


if __name__ == "__main__":
    main()
