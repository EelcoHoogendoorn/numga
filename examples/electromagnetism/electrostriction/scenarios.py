"""Hexagonal lattice parameters define the field-sweep and rotating-field scenes for electrostriction."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.sparse import Sparse
from numga.topology import Topology

from examples.electromagnetism.electrostriction import core

n_rings = 2
spacing = 1.0
polarizability = 0.035 * spacing**3
stiffness = 1.0
tether_stiffness = 0.15 * stiffness
field_strength = 3.05
n_newton_steps = 4
n_load_samples = 25
n_frames = 126
frame_duration_ms = 40
dipole_scale = 4.0
field_scale = 0.65 / field_strength


# --- math -----------------------------------------------------------------------------
def reference_lattice(n_rings: int, spacing: float) -> tuple[Topology, core.Point]:
    """Axial ring indices and nearest-neighbour edges define the bond topology and reference sites (`Point-[V]`) of a hexagonal patch."""
    indices, bonds = hexagonal_indices(n_rings)
    topology = Topology(len(indices), bonds)
    rest_positions = core.lattice_positions(indices, spacing)                   # Point-[V]
    return topology, rest_positions


def loading_curve(
    rest_positions: core.Point,
    bond_ends: Sparse,
    field_strength: float,
    polarizability: float,
    stiffness: float,
    tether_stiffness: float,
    n_load_samples: int,
    n_newton_steps: int,
) -> tuple[np.ndarray, core.Field, core.Point, core.Point]:
    """Increasing horizontal field strengths drive equilibrium positions and induced dipoles (`[n_load_samples] Point-[V]`) across the loading sweep."""
    strengths = np.linspace(0, field_strength, n_load_samples)
    load_fields = core.field_along(core.mv.x.dual()) * strengths                # [n_load_samples] Field+
    load_positions = core.mechanical_equilibrium(
        rest_positions, bond_ends, load_fields,
        polarizability, stiffness, tether_stiffness, n_newton_steps,
    )                                                                           # [n_load_samples] Point-[V]
    load_dipoles = core.dipole_response(load_positions, polarizability).solve(load_fields)
    return strengths, load_fields, load_positions, load_dipoles


# --- plumbing -------------------------------------------------------------------------
def hexagonal_indices(n_rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Hexagonal ring bounds select integer axial coordinates (`[V, 2]`) and nearest-neighbour edge pairs (`[E, 2]`)."""
    axis = np.arange(-n_rings, n_rings + 1)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    inside = np.maximum.reduce([np.abs(first), np.abs(second), np.abs(first + second)]) <= n_rings
    first, second = first[inside], second[inside]
    delta_first = first[:, None] - first[None, :]
    delta_second = second[:, None] - second[None, :]
    neighbours = np.maximum.reduce([np.abs(delta_first), np.abs(delta_second), np.abs(delta_first + delta_second)]) == 1
    bonds = np.stack(np.nonzero(np.triu(neighbours, 1)), axis=-1)
    return np.stack([first, second], axis=-1), bonds


def main() -> None:
    """The scenario entry point writes the electrostriction figures and rotating-field animation and checks equilibrium invariants."""
    from examples.electromagnetism.electrostriction import render
    from examples.util.animation import save_animation, save_figure

    topology, rest_positions = reference_lattice(n_rings, spacing)
    bond_ends = topology.d[0]                                                   # Scalar+[E <- V]
    save_figure(render.rest_figure(rest_positions, topology), "electrostriction_lattice")

    strengths, load_fields, load_positions, load_dipoles = loading_curve(
        rest_positions, bond_ends, field_strength,
        polarizability, stiffness, tether_stiffness, n_load_samples, n_newton_steps,
    )
    rigid_dipoles = core.dipole_response(rest_positions, polarizability).solve(load_fields[-1])
    save_figure(
        render.lattice_figure(
            rest_positions, topology, rest_positions, rigid_dipoles, load_fields[-1],
            dipole_scale, field_scale,
        ),
        "electrostriction_fixed",
    )
    save_figure(
        render.lattice_figure(
            rest_positions, topology, load_positions[-1], load_dipoles[-1], load_fields[-1],
            dipole_scale, field_scale,
        ),
        "electrostriction_relaxed",
    )

    parallel_strain = core.axial_strain(load_positions, rest_positions, core.mv.x)      # [n_load_samples] Scalar+
    transverse_strain = core.axial_strain(load_positions, rest_positions, core.mv.y)    # [n_load_samples] Scalar+
    save_figure(render.loading_figure(strengths, parallel_strain, transverse_strain), "electrostriction_loading")

    cycle_states = []

    def recorded_states() -> Iterator[tuple[core.Point, core.Point, core.Field]]:
        for state in core.turning_states(
            rest_positions, bond_ends, field_strength,
            polarizability, stiffness, tether_stiffness, n_frames, n_newton_steps,
        ):
            cycle_states.append(state)
            yield state

    frames = render.lattice_animation(rest_positions, topology, dipole_scale, field_scale, recorded_states())
    save_animation(frames, "electrostriction", frame_duration_ms)

    # checks
    assert ((load_positions[0] - rest_positions).antinorm() < 1e-6).all()
    assert parallel_strain[-1] < 0
    assert transverse_strain[-1] > 0
    assert (parallel_strain[2] / parallel_strain[1] - 4).abs() < 0.05
    initial_shape, initial_dipoles, _ = cycle_states[0]
    opposite_shape, opposite_dipoles, _ = cycle_states[n_frames // 2]
    assert ((opposite_shape - initial_shape).antinorm() < 1e-3).all()
    assert ((opposite_dipoles + initial_dipoles).antinorm() < 1e-3).all()


if __name__ == "__main__":
    main()
