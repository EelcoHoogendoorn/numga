"""Closed work cycles and overdamped waves in planar active-bond lattices."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from examples.mechanics.odd_elasticity import core
from numga.topology import Topology

mv = core.mv
Scalar, Displacement, Force, Shear = core.Scalar, core.Displacement, core.Force, core.Shear
x, y = core.x, core.y

# Bond parameters shared by the single-bond and shear-cycle demonstrations.
stiffness = 1
activity = 3
actuation_plane = mv.xy                                                       # Bivector+
n_cycle_steps = 126

# Single-bond elliptical endpoint excursion.
rest_length = 1
parallel_radius = 0.25
transverse_radius = 0.18

# Finite triangular patch driven through a circle of pure shears.
spacing = 1
shear_amplitude = 0.14
n_patch_columns, n_patch_rows = 7, 7

# Overdamped wavepacket released on a wider free triangular lattice.
wave_stiffness = 0.2
wave_activity = 4
drag = 1
wave_amplitude = 0.15
wave_width = 2.5
wave_number = 1.2
n_wave_columns, n_wave_rows = 21, 11
n_wave_frames = 180
n_wave_steps_per_frame = 2
time_step = 1 / 60


# --- plumbing -------------------------------------------------------------------------
def triangular_patch(n_columns: int, n_rows: int, grid_spacing: float) -> tuple[Topology, np.ndarray]:
    """An `n_columns` by `n_rows` triangular grid defines the free-boundary lattice topology and `[V, 2]` vertex coordinates."""
    columns, rows = np.meshgrid(np.arange(n_columns), np.arange(n_rows))
    horizontal = grid_spacing * (columns + (rows % 2) / 2 - (n_columns - 1) / 2)
    vertical = grid_spacing * (rows - (n_rows - 1) / 2) * np.sqrt(3) / 2
    coordinates = np.stack((horizontal, vertical), axis=-1).reshape(-1, 2)   # [V, 2]

    # Adjacent rows bound two counter-clockwise triangles per quad; alternating half-step row offsets flip the shared diagonal.
    indices = np.arange(n_columns * n_rows).reshape(n_rows, n_columns)
    lower_left, lower_right = indices[:-1, :-1], indices[:-1, 1:]
    upper_left, upper_right = indices[1:, :-1], indices[1:, 1:]
    odd_row_triangles = np.stack((lower_left, upper_right, upper_left, lower_left, lower_right, upper_right), axis=-1)
    even_row_triangles = np.stack((lower_left, lower_right, upper_left, lower_right, upper_right, upper_left), axis=-1)
    odd_row_mask = (np.arange(n_rows - 1) % 2 == 1)[:, None, None]
    triangles = np.where(odd_row_mask, odd_row_triangles, even_row_triangles).reshape(-1, 3)  # [F, 3]
    return Topology(n_columns * n_rows, triangles), coordinates


def streamed(states: Iterable, recorded: list) -> Iterator:
    """Yields each state from `states` while recording it into `recorded` during a single pass."""
    for state in states:
        recorded.append(state)
        yield state


# --- math -----------------------------------------------------------------------------
def unit_circle(n_steps: int) -> Displacement:
    """Uniform rotor phases sample `[n_steps + 1]` unit displacements counter-clockwise around `actuation_plane`."""
    phases = np.linspace(0, 2 * np.pi, n_steps + 1)
    turns = (-actuation_plane * (phases / 2)).exp()                          # [n_steps + 1] Rotor+
    return turns >> x                                                        # [n_steps + 1] Displacement-


def bond_cycle(n_steps: int) -> tuple[Displacement, Displacement, Displacement, Iterator[tuple[Displacement, Force, Scalar]]]:
    """An elliptical endpoint excursion around a single active bond defines the anchor, rest point, path, and `(position, force, work)` stream."""
    direction = x                                                            # Displacement-
    anchor = 0 * direction                                                   # Displacement-
    reference = rest_length * direction                                      # Displacement-
    response = core.bond_response(direction, actuation_plane, stiffness, activity)  # Force+ <- Displacement-
    across = direction | actuation_plane                                     # Displacement-
    # Stretch the unit circle independently along and across the bond axis.
    ellipse = (
        parallel_radius * direction * (direction.dual_inverse() & Displacement)
        + transverse_radius * across * (across.dual_inverse() & Displacement)
    )                                                                        # Displacement- <- Displacement-
    cycle = ellipse(unit_circle(n_steps))                                    # [n_steps + 1] Displacement-
    states = (
        (reference + displacement, force, work)
        for displacement, force, work in core.cycle_states(response, cycle)
    )
    return anchor, reference, reference + cycle, states


def shear_cycle(n_steps: int) -> tuple[Displacement, Topology, Iterator[tuple[Displacement, Force, Scalar]]]:
    """A circle of pure shears on a finite triangular patch defines the reference lattice and `(positions, forces, cell_work)` stream."""
    topology, coordinates = triangular_patch(n_patch_columns, n_patch_rows, spacing)
    reference = mv.vector(coordinates).dual().field()                        # Displacement-[V]

    # One bond from each of the three lattice families in a unit cell.
    bond_angles = np.arange(3) * np.pi / 3
    family_turns = (-actuation_plane * (bond_angles / 2)).exp()              # [3] Rotor+
    bond_directions = family_turns >> x                                      # [3] Displacement-
    cell_response = core.bond_response(bond_directions, actuation_plane, stiffness, activity)  # [3] Force+ <- Displacement-
    bond_motion = core.shear_strain(Shear, spacing * bond_directions)        # [3] Displacement- <- Shear-
    # Pulling bond forces back through the bond motion yields the macroscopic shear response per cell.
    shear_response = bond_motion.adjugate()(cell_response(bond_motion)).sum(axis=0)  # Force+ <- Shear-

    # Assembling the same bond responses across the finite patch exposes its boundary loads.
    edge_directions = (topology.d[0] * reference).normalized()               # Displacement-[E]
    edge_response = core.bond_response(edge_directions, actuation_plane, stiffness, activity)  # (Force+ <- Displacement-)[E]
    patch_response = core.lattice_response(topology.d[0], edge_response)     # (Force+ <- Displacement-)[V <- V]
    patch_motion = core.shear_strain(Shear, reference)                       # (Displacement- <- Shear-)[V]
    patch_forces = patch_response(patch_motion)                              # (Force+ <- Shear-)[V]
    cycle = shear_amplitude * unit_circle(n_steps)                           # [n_steps + 1] Shear-
    states = (
        (reference + patch_motion(shear), patch_forces(shear), work)
        for shear, _, work in core.cycle_states(shear_response, cycle)
    )
    return reference, topology, states


def lattice_wave(n_frames: int, n_steps_per_frame: int, dt: float) -> tuple[Displacement, Topology, Iterator[tuple[Displacement, Scalar]]]:
    """A Gaussian-modulated wavepacket on a free triangular lattice defines the reference lattice and `(positions, extensions)` stream."""
    topology, coordinates = triangular_patch(n_wave_columns, n_wave_rows, spacing)
    V, E, F = topology.n_elements
    reference = mv.vector(coordinates).dual().field()                        # Displacement-[V]
    bond_directions = (topology.d[0] * reference).normalized()               # Displacement-[E]
    response = core.bond_response(bond_directions, actuation_plane, wave_stiffness, wave_activity)  # (Force+ <- Displacement-)[E]
    drags = mv.scalar([[drag]] * V).field()                                  # Scalar+[V]
    velocities = core.velocity_map(topology.d[0], response, drags)           # (Displacement- <- Displacement-)[V <- V]
    envelope = (-reference.scalar_norm_squared() / (2 * wave_width**2)).exp()  # Scalar+[V]
    initial = wave_amplitude * x * envelope * (wave_number * (mv.x & reference)).cos()  # Displacement-[V]
    return reference, topology, core.wave_states(reference, topology.d[0], bond_directions, velocities, initial, n_frames, n_steps_per_frame, dt)


def main() -> None:
    """Streams all three odd-elasticity animations to `PLOT_DIR` and verifies cyclic work and wave invariants."""
    from examples.mechanics.odd_elasticity import render
    from examples.util.animation import save_animation

    frame_duration_ms = 40
    force_scale = 0.55
    patch_force_scale = 1.3
    displacement_scale = 4
    extension_scale = 0.04
    tracked_index = (n_wave_rows // 2) * n_wave_columns + n_wave_columns // 2

    bond_records: list[tuple[Displacement, Force, Scalar]] = []
    anchor, reference, loop, bond_stream = bond_cycle(n_cycle_steps)
    save_animation(render.bond_frames(anchor, reference, loop, force_scale, streamed(bond_stream, bond_records)),
                   "odd_elasticity_bond", frame_duration_ms)

    shear_records: list[tuple[Displacement, Force, Scalar]] = []
    patch, patch_topology, shear_stream = shear_cycle(n_cycle_steps)
    save_animation(render.patch_frames(patch, patch_topology, patch_force_scale, streamed(shear_stream, shear_records)),
                   "odd_elasticity_shear", frame_duration_ms)

    wave_records: list[tuple[Displacement, Scalar]] = []
    wave_reference, wave_topology, wave_stream = lattice_wave(n_wave_frames, n_wave_steps_per_frame, time_step)
    save_animation(render.wave_frames(wave_reference, wave_topology, displacement_scale, extension_scale,
                                      tracked_index, streamed(wave_stream, wave_records)),
                   "odd_elasticity_wave", frame_duration_ms)

    # checks
    ellipse = loop - reference                                               # [n_cycle_steps + 1] Displacement-
    ellipse_area = (ellipse[:-1] ^ ellipse[1:]).sum(axis=0) / 2              # Bivector+
    shear_loop = shear_amplitude * unit_circle(n_cycle_steps)                # [n_cycle_steps + 1] Shear-
    shear_area = (shear_loop[:-1] ^ shear_loop[1:]).sum(axis=0) / 2          # Bivector+
    bond_work = -activity * actuation_plane.scalar_product(ellipse_area)     # Scalar+
    shear_work = -3 * activity * actuation_plane.scalar_product(shear_area)  # Scalar+
    assert ((bond_records[-1][2] - bond_work).abs() < 1e-12).all()
    assert ((shear_records[-1][2] - shear_work).abs() < 1e-12).all()
    initial_centre = (wave_records[0][0] - wave_reference).batch().sum(axis=0)   # Displacement-
    final_centre = (wave_records[-1][0] - wave_reference).batch().sum(axis=0)    # Displacement-
    assert ((final_centre - initial_centre).norm() < 1e-12).all()
    assert (wave_records[-1][1].scalar_norm_squared().batch().sum()
            < wave_records[0][1].scalar_norm_squared().batch().sum()).all()


if __name__ == "__main__":
    main()
