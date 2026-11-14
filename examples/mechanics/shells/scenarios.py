"""Two incompatible shell scenarios: a disc whose rest edge lengths grow quadratically toward its
rim buckles into a saddle, and a strip with the crossed natural curvatures of a seed pod's two
fibrous layers twists into a ribbon."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.shells import core
from numga.sparse import Sparse
from numga.topology import Topology

mv = core.mv
# Disc geometry and rim growth parameters.
disc_radius = 1.0
disc_spacing = 0.1
rim_growth = 0.3
disc_thickness = 0.03
roughness = 1e-3
# Seed-pod strip geometry and crossed natural curvature parameters.
strip_length = 2.0
strip_width = 0.3
strip_spacing = 0.08
strip_thickness = 0.05
natural_curvature = 6.0
pod_angle = np.pi / 4
# Damped relaxation time steps and frame counts.
disc_step = 0.02
strip_step = 0.012
damping = 0.01
default_n_frames = 100
default_steps_per_frame = 20
seed = 0


# --- plumbing -------------------------------------------------------------------------
def disc_domain(spacing: float, radius: float) -> tuple[Topology, core.Point, core.Point]:
    """Triangulating a circular lattice of `radius` at `spacing` gives `(topology, rest_points,
    perturbed_points)` (`Point-[V]`) with small random out-of-plane heights."""
    flat = core.triangular_lattice(spacing, radius)
    flat = flat[np.linalg.norm(flat, axis=-1) <= radius + 1e-9]                  # [V, 2]
    topology, points = core.flat_triangulation(flat, spacing)                    # Point-[V]
    heights = roughness * np.random.default_rng(seed).normal(size=(len(flat), 1))
    perturbed = mv.point(np.concatenate([flat, heights], axis=-1)).field()       # Point-[V]
    return topology, points, perturbed


def strip_domain(
    spacing: float, length: float, width: float,
) -> tuple[Topology, core.Point, core.Point]:
    """Triangulating a rectangular `length`-by-`width` lattice at `spacing` gives `(topology,
    rest_points, perturbed_points)` (`Point-[V]`) with small random out-of-plane heights."""
    flat = core.triangular_lattice(spacing, length)
    flat = flat[(np.abs(flat[:, 0]) <= length / 2) & (np.abs(flat[:, 1]) <= width / 2)]   # [V, 2]
    topology, points = core.flat_triangulation(flat, spacing)                    # Point-[V]
    heights = roughness * np.random.default_rng(seed).normal(size=(len(flat), 1))
    perturbed = mv.point(np.concatenate([flat, heights], axis=-1)).field()       # Point-[V]
    return topology, points, perturbed


# --- math -----------------------------------------------------------------------------
def disc(
    spacing: float, n_frames: int, n_steps_per_frame: int,
) -> tuple[Topology, Sparse, Sparse, core.Scalar, core.Chiral, float, core.Scalar, Iterator[core.Point]]:
    """Growing rest edge lengths quadratically toward the rim sets up the disc shell and yields its
    relaxation trajectory (`Iterator[Point-[V]]`)."""
    topology, points, perturbed = disc_domain(spacing, disc_radius)
    tails, heads = topology.incidence[1, 0] * points                             # Point-[E], Point-[E]
    # Scale rest edge lengths quadratically with midpoint distance from the origin.
    distance = (mv.w.dual() & ((tails + heads) / 2)).norm() / disc_radius        # Scalar+[E]
    lengths = (tails & heads).norm() * (1 + rim_growth * distance * distance)    # Scalar+[E]
    zero_curvature = 0 * (core.Line | core.Line)                                 # [] Scalar+ <- (Line-, Line-)
    hinge_ends, hinge_corners, angles, bending, masses = core.rest_geometry(
        topology, points, zero_curvature, mv.z, disc_thickness,
    )
    trajectory = core.relax(
        topology, hinge_ends, hinge_corners, lengths, angles, disc_thickness, bending, masses,
        perturbed, disc_step, damping, n_frames, n_steps_per_frame,
    )
    return topology, hinge_ends, hinge_corners, lengths, angles, disc_thickness, bending, trajectory


def strip(
    spacing: float, n_frames: int, n_steps_per_frame: int,
) -> tuple[Topology, Sparse, Sparse, core.Scalar, core.Chiral, float, core.Scalar, Iterator[core.Point]]:
    """Opposing principal curvatures at plus and minus `pod_angle` from the strip axis set up the
    seed-pod strip shell and yield its relaxation trajectory (`Iterator[Point-[V]]`)."""
    topology, points, perturbed = strip_domain(spacing, strip_length, strip_width)
    tails, heads = topology.incidence[1, 0] * points                             # Point-[E], Point-[E]
    lengths = (tails & heads).norm()                                             # Scalar+[E]
    # Rotate the strip's centreline either way in the xy-plane to align the two fibrous layers.
    axis = mv.wx.dual()                                                          # [] Line-
    first_turn = (mv.xy * (-pod_angle / 2)).exp()                                # [] Motor+
    second_turn = (mv.xy * (pod_angle / 2)).exp()                                # [] Motor+
    first_fibre, second_fibre = first_turn >> axis, second_turn >> axis          # [] Line-, [] Line-
    curvature = natural_curvature * (
        (second_fibre | core.Line) * (second_fibre | core.Line)
        - (first_fibre | core.Line) * (first_fibre | core.Line)
    )                                                                            # [] Scalar+ <- (Line-, Line-)
    hinge_ends, hinge_corners, angles, bending, masses = core.rest_geometry(
        topology, points, curvature, mv.z, strip_thickness,
    )
    trajectory = core.relax(
        topology, hinge_ends, hinge_corners, lengths, angles, strip_thickness, bending, masses,
        perturbed, strip_step, damping, n_frames, n_steps_per_frame,
    )
    return topology, hinge_ends, hinge_corners, lengths, angles, strip_thickness, bending, trajectory


def main() -> None:
    from examples.mechanics.shells import render
    from examples.util.animation import save_animation, save_figure

    def streamed(states: Iterator[core.Point], kept: list[core.Point]) -> Iterator[core.Point]:
        for state in states:
            kept.append(state)
            yield state

    disc_frames: list[core.Point] = []
    disc_topology, disc_ends, disc_corners, disc_lengths, disc_angles, disc_stretch, disc_bend, disc_run = disc(
        disc_spacing, default_n_frames, default_steps_per_frame,
    )
    save_animation(
        render.animate(disc_topology.elements[2], streamed(disc_run, disc_frames)),
        "shells_disc", render.frame_duration_ms,
    )

    strip_frames: list[core.Point] = []
    strip_topology, strip_ends, strip_corners, strip_lengths, strip_angles, strip_stretch, strip_bend, strip_run = strip(
        strip_spacing, default_n_frames, default_steps_per_frame,
    )
    save_animation(
        render.animate(strip_topology.elements[2], streamed(strip_run, strip_frames)),
        "shells_strip", render.frame_duration_ms,
    )

    save_figure(
        render.draw_settled([
            (disc_topology.elements[2], disc_frames[-1]),
            (strip_topology.elements[2], strip_frames[-1]),
        ]),
        "shells_settled",
    )

    # checks
    for topology, hinge_ends, hinge_corners, lengths, angles, stretching, bending, frames, min_span in (
        (disc_topology, disc_ends, disc_corners, disc_lengths, disc_angles, disc_stretch, disc_bend, disc_frames, 0.3),
        (strip_topology, strip_ends, strip_corners, strip_lengths, strip_angles, strip_stretch, strip_bend, strip_frames, 0.1),
    ):
        initial, settled = frames[0], frames[-1]
        assert core.shell_energy(
            topology, hinge_ends, hinge_corners, lengths, angles, stretching, bending, settled,
        ) < core.shell_energy(
            topology, hinge_ends, hinge_corners, lengths, angles, stretching, bending, initial,
        )
        # Internal forces produce zero net force and zero net moment (vanishing total action line).
        total = (settled & core.shell_forces(
            topology, hinge_ends, hinge_corners, lengths, angles, stretching, bending, settled,
        )).sites.sum()                                                           # [] Line-
        assert total.scalar_norm_squared() + total.dual().scalar_norm_squared() < 1e-24
        heights = (mv.z & settled).batch()                                       # [V] Scalar+
        assert heights.max() - heights.min() > min_span
    tails, heads = disc_topology.incidence[1, 0] * disc_frames[-1]               # Point-[E], Point-[E]
    strain = ((tails & heads).norm() / disc_lengths - 1).batch().abs()           # [E] Scalar+
    assert strain.mean() < 0.02


if __name__ == "__main__":
    main()
