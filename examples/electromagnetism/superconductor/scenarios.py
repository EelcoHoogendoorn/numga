"""Scenarios for Meissner field expulsion in a superconducting rod and fluxoid quantization in a superconducting ring.

Below the critical temperature, the superfluid electron fraction follows the empirical two-fluid temperature law from
the Stockholm superconductivity lecture notes, growing from zero at the transition to near unity as the rod cools.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.superconductor import core
from examples.util import shapes

mv = core.mv

outer_radius, n_divisions, penetration_depth = 2.0, 56, 0.12
n_normal, n_cooling, n_held, n_switching, n_rest = 10, 120, 20, 100, 20
n_steps = n_normal + n_cooling + n_held + n_switching + n_rest
base_temperature = 1 / 3
dt, n_substeps, frame_duration_ms = 2e-3, 3, 40
radial_width = 0.06

hole_radius, n_ring_divisions = 0.5, 40
trapped_quanta = np.array([1, 2, 3])                                          # [3]
n_candidates, n_cooling_fields = 6, 77
candidate_quanta = np.arange(n_candidates)                                    # [n_candidates]
cooling_fields = np.linspace(0.0, 1.9, n_cooling_fields)                      # [n_cooling_fields]


def cooling_schedule() -> tuple[np.ndarray, np.ndarray]:
    """Two-fluid temperature cooling and external field shutoff define the step schedules (`[n_steps]`)."""
    temperatures = np.concatenate([
        np.ones(n_normal),
        np.linspace(1, base_temperature, n_cooling),
        np.full(n_held + n_switching + n_rest, base_temperature),
    ])                                                                        # [n_steps]
    applied_fields = np.concatenate([
        np.ones(n_normal + n_cooling + n_held),
        np.linspace(1, 0, n_switching),
        np.zeros(n_rest),
    ])                                                                        # [n_steps]
    return 1 - np.minimum(temperatures, 1) ** 4, applied_fields


def rod_scene(n_divisions: int) -> tuple[core.Topology, np.ndarray, np.ndarray]:
    """Concentric triangle rings form the disk topology around the unit rod, `z = 0` coordinates (`[V, 3]`), and rod mask (`[V]`)."""
    topology, planar_coords = shapes.concentric_disk(np.array([1.0, outer_radius]), n_divisions)
    V, E, F = topology.n_elements
    coordinates = np.column_stack([planar_coords, np.zeros(V)])               # [V, 3]
    in_rod = np.linalg.norm(planar_coords, axis=-1) <= 1.0 + 1e-9             # [V]
    return topology, coordinates, in_rod


def cooling_run(
    n_divisions: int,
) -> tuple[core.Topology, core.Point, np.ndarray, Iterator[tuple[core.Scalar, core.Scalar]]]:
    """The triangulated rod scene pairs with the cooling generator (`[2] Scalar+[V]`) for both conductors."""
    topology, coordinates, in_rod = rod_scene(n_divisions)
    vertices = mv(core.Plane, coordinates).dual().field()                     # Point-[V]
    areas = core.triangle_areas(topology, vertices)                           # Scalar+[F]
    gradients = core.simplex_gradients(topology, vertices)                    # [3] Plane+[F]
    rod_mask = mv.scalar(in_rod[:, None]).field()                             # Scalar+[V]
    heights = mv.y & vertices                                                 # Scalar+[V]
    reference_potential = core.stack([0 * heights, heights])                  # [2] Scalar+[V]
    superfluid_shares, applied_fields = cooling_schedule()                    # [n_steps], [n_steps]
    states = core.cooling_states(
        topology, areas, gradients, rod_mask, heights, reference_potential,
        penetration_depth, superfluid_shares, applied_fields, dt, n_substeps,
    )
    return topology, vertices, in_rod, states


def ring_scene(n_divisions: int) -> tuple[core.Topology, np.ndarray, np.ndarray]:
    """Concentric triangle rings form the ring topology, `z = 0` coordinates (`[V, 3]`), and central hole mask (`[V]`)."""
    topology, planar_coords = shapes.concentric_disk(np.array([hole_radius, 1.0]), n_divisions)
    V, E, F = topology.n_elements
    coordinates = np.column_stack([planar_coords, np.zeros(V)])               # [V, 3]
    in_hole = np.linalg.norm(planar_coords, axis=-1) <= hole_radius + 1e-9    # [V]
    return topology, coordinates, in_hole


def main() -> None:
    """Scenario entry point writes the rod figures, ring figures, cooling animation, and verification checks."""
    from examples.electromagnetism.superconductor import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, in_rod, states = cooling_run(n_divisions)
    save_figure(render.rod_figure(topology, vertices, in_rod), "superconductor_rod")

    recorded_states: list[tuple[core.Scalar, core.Scalar]] = []
    save_animation(
        render.cooling_animation(
            topology, vertices, in_rod, penetration_depth,
            (recorded_states.append(state) or state for state in states),
        ),
        "superconductor",
        frame_duration_ms,
    )

    gradients = core.simplex_gradients(topology, vertices)                    # [3] Plane+[F]
    held_index = (n_normal + n_cooling + n_held - 1) // n_substeps
    held_potential = recorded_states[held_index][0][0]                        # Scalar+[V]
    held_field = core.magnetic_field(topology, gradients, held_potential)     # Field+[F]
    depths, strengths = core.radial_profile(topology, vertices, held_field, radial_width)
    save_figure(render.penetration_figure(depths, strengths, penetration_depth), "superconductor_depth")

    ring_topology, ring_coordinates, in_hole = ring_scene(n_ring_divisions)
    ring_vertices = mv(core.Plane, ring_coordinates).dual().field()           # Point-[V]
    ring_areas = core.triangle_areas(ring_topology, ring_vertices)            # Scalar+[F]
    ring_gradients = core.simplex_gradients(ring_topology, ring_vertices)     # [3] Plane+[F]
    trapped_fields = core.ring_fields(
        ring_topology, ring_areas, ring_gradients, in_hole,
        penetration_depth, trapped_quanta, 0 * trapped_quanta,
    )                                                                         # [3] Scalar+[V]
    save_figure(render.trapped_figure(ring_topology, ring_vertices, in_hole, trapped_fields), "superconductor_ring")

    retained_quanta = core.equilibrium_quanta(
        ring_topology, ring_areas, ring_gradients, in_hole,
        penetration_depth, candidate_quanta, cooling_fields,
    )                                                                         # [n_cooling_fields]
    ring_corners = ring_topology.incidence[2, 0]                              # [3] Scalar+[F <- V]
    ring_dual_areas = (ring_corners.adjugate() * (ring_areas / 3)).sum(axis=0)   # Scalar+[V]
    zero_applied = np.zeros_like(retained_quanta, dtype=float)                # [n_cooling_fields]
    retained_fields = core.ring_fields(
        ring_topology, ring_areas, ring_gradients, in_hole,
        penetration_depth, retained_quanta, zero_applied,
    )                                                                         # [n_cooling_fields] Scalar+[V]
    retained_flux = (retained_fields * ring_dual_areas).sites.sum()           # [n_cooling_fields] Scalar+
    save_figure(render.staircase_figure(cooling_fields, retained_flux), "superconductor_staircase")

    # checks
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    heights = mv.y & vertices                                                 # Scalar+[V]
    assert (((gradients * (corners * heights)).sum(axis=0) - mv.y).norm() < 1e-10).all()

    shallow_mask = (depths > 0.05) & (depths < 0.4)
    depth_offsets = depths[shallow_mask] - depths[shallow_mask].mean()
    log_offsets = strengths[shallow_mask].log() - strengths[shallow_mask].log().mean()
    decay_length = -(depth_offsets * depth_offsets).sum() / (depth_offsets * log_offsets).sum()
    assert ((decay_length - penetration_depth).abs() < 0.1 * penetration_depth).all()

    core_mask = vertices.antinorm().batch() < 0.6
    final_potentials = recorded_states[-1][0].batch()                         # [2, V] Scalar+
    assert (final_potentials[0, core_mask].abs() < 0.02).all()
    assert ((final_potentials[1, core_mask] - heights.batch()[core_mask]).abs() < 0.02).all()

    trapped_flux = (trapped_fields * ring_dual_areas).sites.sum()             # [3] Scalar+
    assert ((trapped_flux - trapped_quanta).abs() < 0.02 * trapped_quanta).all()
    assert set(np.unique(retained_quanta)) == {0, 1, 2}


def pinned() -> None:
    """Gallery entry point writes the pinned side-by-side cooling animation of both conductors."""
    from examples.electromagnetism.superconductor import render
    from examples.util.animation import pin_animation

    topology, vertices, in_rod, states = cooling_run(n_divisions)
    pin_animation(render.cooling_animation(topology, vertices, in_rod, penetration_depth, states), "superconductor", frame_duration_ms)


if __name__ == "__main__":
    main()
