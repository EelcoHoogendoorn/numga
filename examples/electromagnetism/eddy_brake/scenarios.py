"""Eddy-current braking scenarios for isotropic, radial-fibre, circumferential-fibre, and
straight-fibre conducting discs."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.sparse import SparseExtensor
from numga.topology import Topology
from examples.util import shapes
from examples.electromagnetism.eddy_brake import core

radius = 0.1                                       # m
thickness = 0.001                                  # m
bulk_conductivity = 5.8e7                          # S/m
# Isotropic, radial-fibre, and circumferential-fibre discs share the same mean conductivity.
labels = ("Isotropic", "Radial fibres", "Circumferential fibres")
radial_conductivity = bulk_conductivity * np.array([1.0, 1.9, 0.1])          # [3]
circumferential_conductivity = bulk_conductivity * np.array([1.0, 0.1, 1.9])  # [3]
angular_speed = 1.0                                # rad/s
field_strength = 0.2                               # T
field_width = 0.02                                 # m, Gaussian standard deviation
magnet_offset = 0.055                              # m, along x
n_divisions = 16
inertia = 0.005                                    # kg m^2, disc and attached rotor
initial_speed = 1.2                                # rad/s
initial_angle = 0.0                                # rad, straight fibres initially along x
along_conductivity = bulk_conductivity * 1.9
across_conductivity = bulk_conductivity * 0.1
# Poor cross-fibre conductivity requires a stronger magnet to match the isotropic braking time.
straight_field_strength = 0.35                     # T
elapsed = 6.0                                      # s
n_steps = 120
time_step = elapsed / n_steps
frame_duration_ms = 40
n_animation_colours = 64
n_fibre_lines = 12
n_fibre_samples = 64
fibre_inner_radius = 0.2 * radius
fibre_outer_radius = 0.85 * radius
material_radii = np.array([fibre_inner_radius, fibre_outer_radius, radius])  # [3]
permeability = 4 * np.pi * 1e-7                    # H/m, nonmagnetic conductor in free space
n_modes = 63
n_shown_modes = 4
n_harmonics = 16
inductive_field_strength = 1.0                     # T
# A thicker disc conducts more and retains modal currents longer across a turn.
inductive_thickness = 0.004                        # m
inductive_initial_speed = 0.0                      # rad/s
inductive_target_speed = 160.0                     # rad/s, unloaded motor pulse
spinup_time = 0.008                                # s
inductive_elapsed = 0.07                           # s
n_inductive_steps = 105
inductive_time_step = inductive_elapsed / n_inductive_steps
n_substeps = 4

centre = (core.mv.x * magnet_offset).dual()                                  # [] Point-
magnet_field = core.mv.xy * field_strength                                   # [] Bivector+
straight_magnet_field = core.mv.xy * straight_field_strength                 # [] Bivector+
inductive_magnet_field = core.mv.xy * inductive_field_strength               # [] Bivector+


# --- math -----------------------------------------------------------------------------
def disc_geometry(topology: Topology,
                  coordinates: np.ndarray) -> tuple[Topology, core.Point, core.Point, core.Point, core.Point, core.Point, core.Scalar, core.Point, SparseExtensor]:
    """Triangulated disc geometry builds vertex offsets `Point-[V]`, edge displacements and
    midpoints `Point-[E]`, triangle corners and opposite edges `[3] Point-[F]`, areas `Scalar+[F]`,
    centroids `Point-[F]`, and the Whitney field reconstruction `(Field+ <- Scalar+)[F <- E]`."""
    vertices = core.mv(core.Spatial, coordinates).dual().field()             # Point-[V]
    edge_displacements = topology.d[0] * vertices                            # Point-[E]
    edge_midpoints = topology.incidence[1, 0].mean(axis=0) * vertices        # Point-[E]
    corners = topology.incidence[2, 0] * vertices                            # [3] Point-[F]
    triangle_edges = topology.boundary[2, 1] * edge_displacements            # [3] Point-[F]
    double_areas = (triangle_edges[0] & triangle_edges[1]) & core.time       # Antiscalar-[F]
    triangle_areas = double_areas.norm() / 2                                 # Scalar+[F]
    face_centers = corners.mean(axis=0)                                      # Point-[F]
    reconstruction = core.disc_reconstruction(topology, vertices)            # (Field+ <- Scalar+)[F <- E]
    return (topology, vertices, edge_displacements, edge_midpoints, corners,
            triangle_edges, triangle_areas, face_centers, reconstruction)


def motor_pulse(n_frames: int, n_frame_substeps: int, step_duration: float) -> core.Scalar:
    """Half-sine motor torque pulse (`[n_frames, n_frame_substeps, 1] Scalar+`) accelerates an
    unloaded rotor to `inductive_target_speed` over `spinup_time` before releasing it."""
    substep_times = (np.arange(n_frames * n_frame_substeps) + 0.5) * step_duration / n_frame_substeps
    phase = np.clip(substep_times / spinup_time, 0, 1)
    acceleration = inductive_target_speed * np.pi / (2 * spinup_time) * np.sin(np.pi * phase) * (substep_times < spinup_time)
    return core.mv.scalar(inertia * acceleration.reshape(n_frames, n_frame_substeps, 1, 1))  # [n_frames, n_frame_substeps, 1] Scalar+


def materials_setup(n_ring_divisions: int) -> tuple[tuple, core.Conductivity]:
    """Concentric disc geometry and the three sheet conductivities (`[3] (Current- <- Field+)[F]`)
    share one mean conductance across isotropic, radial-fibre, and circumferential-fibre bands."""
    topology, coordinates = shapes.concentric_disk(material_radii, n_ring_divisions)
    geometry = disc_geometry(topology, coordinates)
    face_centers = geometry[7]                                               # Point-[F]
    principal = thickness * np.stack((radial_conductivity, circumferential_conductivity), axis=-1)  # [3, 2]
    conductivity = core.fibre_conductivity(face_centers, principal, fibre_inner_radius,
                                           fibre_outer_radius, thickness * bulk_conductivity)  # [3] (Current- <- Field+)[F]
    return geometry, conductivity


def steady_materials(geometry: tuple,
                     conductivity: core.Conductivity) -> tuple[core.Scalar, core.Amount, core.Current, core.Scalar, core.Moment]:
    """Steady rotation at `angular_speed` beneath `magnet_field` gives the potential, dual-edge
    fluxes, sheet currents, Joule heating, and braking torque of all three material discs."""
    topology, _, edge_displacements, edge_midpoints, _, _, triangle_areas, _, reconstruction = geometry
    magnetic = magnet_field * (-(edge_midpoints - centre).antinorm().squared() / (2 * field_width ** 2)).exp()  # Bivector+[E]
    spin = angular_speed * core.mv.xy                                        # [] Bivector+
    return core.steady_response(topology, edge_displacements, edge_midpoints, triangle_areas,
                                reconstruction, magnetic, spin, conductivity)


def straight_spin_down() -> tuple[tuple, Iterator[tuple]]:
    """Straight-fibre disc geometry and midpoint braking trajectory start at `initial_speed` beneath
    `straight_magnet_field`."""
    topology, coordinates = shapes.triangular_disk(radius, n_divisions)
    geometry = disc_geometry(topology, coordinates)
    _, vertices, edge_displacements, edge_midpoints, _, triangle_edges, triangle_areas, _, reconstruction = geometry
    directions = stack((core.mv.x.dual(), core.mv.y.dual()))                 # [2] Current-
    principal = thickness * np.array([along_conductivity, across_conductivity])  # [2]
    conductivity = (directions * (core.Field & (directions & core.time)) * principal).sum(axis=-1)[None]  # [1] Current- <- Field+
    orientation = (core.mv.xy * (-initial_angle / 2)).exp()[None]            # [1] Rotor+
    spin = (core.mv.xy * initial_speed)[None]                                # [1] Bivector+
    heat = (triangle_areas * 0)[None]                                        # [1] Scalar+[F]
    states = core.braking(topology, vertices, edge_displacements, edge_midpoints, triangle_edges,
                          triangle_areas, reconstruction, centre, core.mv.xy,
                          straight_field_strength, field_width, conductivity, inertia,
                          orientation, spin, heat, time_step, n_steps)
    return geometry, states


def materials_spin_down(geometry: tuple, conductivity: core.Conductivity) -> Iterator[tuple]:
    """Yields the midpoint braking states of the three material discs released at `initial_speed`."""
    topology, vertices, edge_displacements, edge_midpoints, _, triangle_edges, triangle_areas, _, reconstruction = geometry
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)           # [3] Rotor+
    spin = (core.mv.xy * initial_speed).broadcast_to(orientation.shape)      # [3] Bivector+
    heat = (triangle_areas * 0).broadcast_to(conductivity.shape)             # [3] Scalar+[F]
    return core.braking(topology, vertices, edge_displacements, edge_midpoints, triangle_edges,
                        triangle_areas, reconstruction, centre, core.mv.xy, field_strength,
                        field_width, conductivity, inertia, orientation, spin, heat,
                        time_step, n_steps)


def inductive_spin_up(geometry: tuple, conductivity: core.Conductivity, n_kept_modes: int,
                      n_drive_harmonics: int, n_frames: int,
                      n_frame_substeps: int) -> tuple[core.Scalar, SparseExtensor, Iterator[tuple]]:
    """Yields the coupled inductive spin-up and braking states across `n_frames` frames alongside
    the `n_kept_modes` slowest mode shapes and their vertex-embedding map."""
    topology, vertices, _, _, corners, triangle_edges, triangle_areas, face_centers, _ = geometry
    decay_rates, mode_shapes, current_map, embedder, resistivity = core.current_modes(
        topology, corners, triangle_edges, triangle_areas, conductivity, permeability, n_kept_modes)
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)           # [3] Rotor+
    spin = (core.mv.xy * inductive_initial_speed).broadcast_to(orientation.shape)  # [3] Bivector+
    heat = (triangle_areas * 0).broadcast_to(conductivity.shape)             # [3] Scalar+[F]
    amplitudes = decay_rates * 0                                             # [3, n_kept_modes] Scalar+
    forcing = core.periodic_drive(face_centers, triangle_areas, mode_shapes, current_map,
                                  centre, core.mv.xy, inductive_field_strength, field_width,
                                  orientation, n_drive_harmonics)
    torques = motor_pulse(n_frames, n_frame_substeps, inductive_time_step)   # [n_frames, n_frame_substeps, 1] Scalar+
    states = core.inductive_braking(vertices, triangle_areas, decay_rates, mode_shapes,
                                    current_map, embedder, resistivity, forcing, core.mv.xy,
                                    inertia, orientation, spin, heat, amplitudes, torques,
                                    inductive_time_step)
    return mode_shapes, embedder, states


# --- plumbing -------------------------------------------------------------------------
def record_states(states: Iterator[tuple], recorded: list[tuple]) -> Iterator[tuple]:
    """Yields each state from `states` while appending it to `recorded`."""
    for state in states:
        recorded.append(state)
        yield state


def main() -> None:
    """All eddy-current braking figures and animations save to `plots/` before power and energy
    balances are checked."""
    import matplotlib.pyplot as plt
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.eddy_brake import render

    geometry, conductivity = materials_setup(n_divisions)
    topology, vertices, _, _, _, triangle_edges, triangle_areas, _, _ = geometry
    _, _, steady_current, steady_heating, steady_torque = steady_materials(geometry, conductivity)
    steady_streams = core.streamfunction_projector(topology, triangle_edges, triangle_areas,
                                                   conductivity)(steady_current)  # [3] Scalar+[V]
    fig = render.steady_figure(topology, vertices, triangle_areas, steady_current, steady_streams,
                               centre, field_width, magnet_field)
    save_figure(fig, "eddy_brake_materials")
    plt.close(fig)

    straight_geometry, straight_stream = straight_spin_down()
    straight_topology, straight_vertices, _, _, _, _, straight_areas, _, _ = straight_geometry
    parallel = (render.parallel_fibres(radius, n_fibre_lines, n_fibre_samples),)
    straight_states: list[tuple] = []
    save_animation(render.animate_currents(straight_topology, straight_vertices, straight_areas,
                                           centre, field_width, straight_magnet_field, parallel,
                                           np.ones((1, 1), dtype=bool),
                                           record_states(straight_stream, straight_states)),
                   "eddy_brake_spin_down", frame_duration_ms, n_animation_colours)

    fibres = (render.radial_fibres(fibre_inner_radius, fibre_outer_radius, n_fibre_lines, n_fibre_samples),
              render.circular_fibres(fibre_inner_radius, fibre_outer_radius, n_fibre_lines, n_fibre_samples))
    shown = np.stack((radial_conductivity > circumferential_conductivity,
                      circumferential_conductivity > radial_conductivity), axis=-1)  # [3, 2]
    braked_states: list[tuple] = []
    save_animation(render.animate_currents(topology, vertices, triangle_areas, centre, field_width,
                                           magnet_field, fibres, shown,
                                           record_states(materials_spin_down(geometry, conductivity), braked_states)),
                   "eddy_brake_materials_spin_down", frame_duration_ms, n_animation_colours)
    save_animation(render.animate_heat(topology, vertices, triangle_areas, centre, field_width,
                                       magnet_field, fibres, shown, braked_states),
                   "eddy_brake_materials_heat", frame_duration_ms, n_animation_colours)

    inductive_conductivity = (inductive_thickness / thickness) * conductivity  # [3] (Current- <- Field+)[F]
    mode_shapes, embedder, inductive_stream = inductive_spin_up(
        geometry, inductive_conductivity, n_modes, n_harmonics, n_inductive_steps, n_substeps)
    modes_fig = render.modes_figure(topology, vertices, embedder(mode_shapes[:, :n_shown_modes]))
    save_figure(modes_fig, "eddy_brake_current_modes")
    plt.close(modes_fig)

    inductive_states: list[tuple] = []
    save_animation(render.animate_induction(topology, vertices, triangle_areas, centre, field_width,
                                            inductive_magnet_field, fibres, shown,
                                            record_states(inductive_stream, inductive_states)),
                   "eddy_brake_inductive_currents", frame_duration_ms, n_animation_colours)
    save_animation(render.animate_induction_heat(topology, vertices, triangle_areas, centre,
                                                 field_width, inductive_magnet_field, fibres, shown,
                                                 inductive_states),
                   "eddy_brake_inductive_heat", frame_duration_ms, n_animation_colours)

    # checks
    steady_spin = angular_speed * core.mv.xy                                 # [] Bivector+
    steady_power_loss = -(steady_spin & steady_torque)                       # [3] Scalar+
    steady_heat_rate = steady_heating.sites.sum()                            # [3] Scalar+
    assert ((steady_power_loss - steady_heat_rate).norm() < 1e-9 * steady_heat_rate).all()

    straight_energy = stack([state[3].sites.sum() + inertia / 2 * state[1].scalar_norm_squared()
                             for state in straight_states])                  # [n_steps + 1, 1] Scalar+
    assert ((straight_energy - straight_energy[0]).norm() < 1e-10 * straight_energy[0]).all()

    braked_energy = stack([state[3].sites.sum() + inertia / 2 * state[1].scalar_norm_squared()
                           for state in braked_states])                      # [n_steps + 1, 3] Scalar+
    assert ((braked_energy - braked_energy[0]).norm() < 1e-10 * braked_energy[0]).all()

    inductive_budget = stack([state[3].sites.sum() + state[6] - state[8]
                              + inertia / 2 * state[1].scalar_norm_squared()
                              for state in inductive_states])                # [n_inductive_steps + 1, 3] Scalar+
    peak_work = inductive_states[-1][8].norm().max()                         # [] Scalar+
    assert ((inductive_budget - inductive_budget[0]).norm() < 1e-7 * peak_work).all()


if __name__ == "__main__":
    main()
