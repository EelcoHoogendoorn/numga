"""Planetary cross-section parameters, shallow oblique impact setup, and cork normal-mode sweep."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain, repeat

import numpy as np

from numga import stack, where
from examples.mechanics.seismic_waves import core
from examples.util.shapes import concentric_disk
from numga.topology import Topology

mv = core.mv

core_radius = 0.55
surface_radius = 1.0
radial_boundaries = np.array([core_radius, surface_radius])
n_radial_divisions = 64

# Region zero is the inviscid-shear liquid core; region one is the viscoelastic solid shell.
layer_densities = np.array([5.0, 1.0])
layer_lambdas = np.array([5.0, 1.0])
layer_mus = np.array([0.0, 1.0])
layer_viscosities = np.array([0.001, 0.0])

impact_angle = np.radians(45)
impact_rotor = (mv.xy * (-impact_angle / 2)).exp()                            # [] Rotor+
impact_normal = impact_rotor >> mv.y.dual()                                   # [] Point-
source_depth = 0.05
source_centre = impact_normal * (surface_radius - source_depth)               # [] Point-
source_width = 0.07
source_support = 3.0
source_speed = 0.05
damping_time = 3e-4

moon_radius = 0.07
moon_speed = 2.0
approach_angle = np.radians(40)
restitution = 0.5
sliding = 0.6
n_approach_frames = 12
dt = 0.00448
n_frames = 100
n_substeps = 8
frame_duration_ms = 40

# Cork has zero first Lamé parameter, giving a low compressional-to-shear speed ratio on a uniform ball.
ball_lambda, ball_mu, ball_density = 0.0, 1.0, 1.0
n_rigid_motions = 3
n_modes = 40
pair_tolerance = 1e-3
n_travelling_waves = 12
n_cycle_frames = 24
mode_peak = 0.0006
displacement_scale = 80.0
dilation_scale = 0.1
vorticity_scale = 0.3


def planet_geometry(n_divisions: int) -> tuple[Topology, core.Point, np.ndarray]:
    """Triangulated concentric disk (`Point-[V]`) partitioned into liquid core (`0`) and solid shell (`1`) face regions."""
    topology, coordinates = concentric_disk(radial_boundaries, n_divisions)
    vertices = mv.vector(coordinates).dual().field()                          # Point-[V]
    centres = topology.incidence[2, 0].mean(axis=0) * vertices                # Point-[F]
    regions = (centres.antinorm() > core_radius).astype(int)                  # [F]
    return topology, vertices, regions


def impact_setup(vertices: core.Point) -> tuple[core.Point, core.Point]:
    """Oblique moon rebound trajectory (`[...] Point-`) and compactly tapered initial nodal velocity (`Point-[V]`)."""
    approach_rotor = (mv.xy * (approach_angle / 2)).exp()                     # [] Rotor+
    incoming = moon_speed * (approach_rotor >> -impact_normal)                # [] Point-
    normal_part = impact_normal * (incoming | impact_normal)                  # [] Point-
    outgoing = sliding * (incoming - normal_part) - restitution * normal_part  # [] Point-
    blow = (incoming - outgoing).normalized()                                 # [] Point-
    contact = impact_normal * (surface_radius + moon_radius)                  # [] Point-
    times = n_substeps * dt * np.arange(-n_approach_frames, n_frames + 1)     # [n_approach_frames + n_frames + 1]
    moon = contact + where(times < 0, incoming * times, outgoing * times)     # [n_approach_frames + n_frames + 1] Point-

    offsets = vertices - source_centre                                        # Point-[V]
    source_radius_squared = offsets.scalar_norm_squared() / source_width**2   # Scalar+[V]
    taper = (1 - source_radius_squared / source_support**2).squared()         # Scalar+[V]
    gaussian = (-source_radius_squared / 2).exp()                             # Scalar+[V]
    envelope = where(source_radius_squared < source_support**2, gaussian * taper, 0 * gaussian)  # Scalar+[V]
    velocity = source_speed * envelope * blow                                 # Point-[V]
    return moon, velocity


def main() -> None:
    import matplotlib.pyplot as plt
    from examples.mechanics.seismic_waves import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, regions = planet_geometry(n_radial_divisions)
    density = mv.scalar(layer_densities[regions, None]).field()               # Scalar+[F]
    lame_lambda = mv.scalar(layer_lambdas[regions, None]).field()             # Scalar+[F]
    lame_mu = mv.scalar(layer_mus[regions, None]).field()                     # Scalar+[F]
    viscosity_field = mv.scalar(layer_viscosities[regions, None]).field()     # Scalar+[F]

    areas = core.triangle_areas(topology, vertices)                           # Scalar+[F]
    gradients = core.simplex_gradients(topology, vertices)                    # [3] Plane+[F]
    stiffness = core.assembled_stiffness(topology, gradients, areas, core.constitutive_form(lame_lambda, lame_mu))
    damping = core.assembled_stiffness(topology, gradients, areas, core.viscous_form(viscosity_field))
    masses = core.vertex_masses(topology, areas, density)                     # Scalar+[V]
    inverse_inertia = (1 / masses) * core.Force.dual()                        # (Point- <- Force+)[V]
    acceleration_map = -inverse_inertia(stiffness)                            # (Point- <- Point-)[V <- V]
    drag_map = -inverse_inertia(damping)                                      # (Point- <- Point-)[V <- V]
    divergence = core.divergence_operator(topology, gradients)                # (Scalar+ <- Point-)[F <- V]
    rotation = core.rotation_operator(topology, gradients)                    # (Bivector+ <- Point-)[F <- V]

    figure = render.materials_figure(topology, vertices, regions)
    save_figure(figure, "seismic_waves_materials")
    plt.close(figure)

    moon, velocity = impact_setup(vertices)
    displacement = 0 * velocity                                               # Point-[V]
    recorded_states: list[tuple[core.Point, core.Point]] = []

    def streamed_states() -> Iterator[tuple[core.Point, core.Point]]:
        for state in core.wave_evolution(acceleration_map, drag_map, displacement, velocity,
                                         damping_time, dt, n_frames, n_substeps):
            recorded_states.append(state)
            yield state

    full_states = chain(repeat((displacement, displacement), n_approach_frames), streamed_states())
    wave_frames = core.wave_readings(divergence, rotation, full_states)
    save_animation(render.waves_animation(topology, vertices, regions, moon, moon_radius,
                                          displacement_scale, dilation_scale, vorticity_scale, wave_frames),
                   "seismic_waves_waves", frame_duration_ms)

    ball_regions = np.ones_like(regions)                                      # [F]
    ball_material = core.constitutive_form(mv.scalar([ball_lambda]), mv.scalar([ball_mu]))
    ball_stiffness = core.assembled_stiffness(topology, gradients, areas, ball_material)
    ball_masses = core.vertex_masses(topology, areas, mv.scalar([ball_density]))  # Scalar+[V]
    squared_frequencies, shapes = core.normal_modes(ball_stiffness, ball_masses, n_rigid_motions + n_modes)
    oscillation_frequencies = squared_frequencies[n_rigid_motions:]           # [n_modes] Scalar+
    oscillation_shapes = shapes[n_rigid_motions:]                             # [n_modes] Point-[V]
    frequency_gaps = oscillation_frequencies[1:] - oscillation_frequencies[:-1]
    pair_indices = np.flatnonzero(frequency_gaps < pair_tolerance * oscillation_frequencies[1:])[:n_travelling_waves]
    mode_frames = core.travelling_waves(divergence, rotation, oscillation_shapes[pair_indices],
                                        oscillation_shapes[pair_indices + 1], mode_peak, n_cycle_frames)
    save_animation(render.modes_animation(topology, vertices, ball_regions, displacement_scale, mode_frames),
                   "seismic_waves_modes", frame_duration_ms)

    # checks
    translations = 0 * vertices + mv.x.dual()                                 # Point-[V]
    assert (stiffness(translations).norm() < 1e-11).all()
    assert (stiffness(vertices.commutator(mv.xy)).norm() < 1e-11).all()
    energies = stack(tuple(core.field_energy(masses, stiffness, d, v) for d, v in recorded_states))  # [n_frames + 1] Scalar+
    assert ((energies[1:] - energies[:-1]) < 0).all()
    assert ((energies[-1] / energies[0]) > 0.5).all()
    initial_momentum = (masses * velocity).sites.sum()                        # [] Point-
    initial_angular_momentum = (vertices & (masses * velocity)).sites.sum()   # [] Patch-
    _, final_velocity = recorded_states[-1]
    assert (((masses * final_velocity).sites.sum() - initial_momentum).antinorm() < 1e-11).all()
    assert (((vertices & (masses * final_velocity)).sites.sum() - initial_angular_momentum).antinorm() < 1e-11).all()


if __name__ == "__main__":
    main()
