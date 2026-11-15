"""Planetary core geometry, Chandrasekhar–Kendall helical stirring, seed field, and run entry points."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.dynamo import core
from examples.util import shapes

mv = core.mv
# Spherical shell resolution and inner-core, outer-core, and mantle radii in outer-core units.
n_levels = 2
n_shells = 6
mantle_radius = 1.5
inner_radius = 0.25
mantle_conductance = 1e-2
# Magnetic Reynolds number, seed field, time-step schedule, and GIF palette sizes.
reynolds = 160.0
seed_field = mv.xy * 1e-2 + mv.yz * 3e-3                                       # Magnetic+
dt = 0.05
n_steps = 2400
n_skipped_steps = 1200
stride = 10
frame_duration_ms = 40
n_cutaway_colours = 64
n_lines_colours = 32
# Modal truncation counts, displayed mode indices, fluid viscosity, and stirring relaxation rate.
n_scalar_modes = 80
n_modes = 60
shown_modes = np.array([0, 3, 8, 40])
viscosity = 0.01
pull = 1.0


# --- plumbing -------------------------------------------------------------------------
def ball_complex(n_levels: int, n_shells: int) -> tuple[core.Topology, core.Point]:
    """Subdivided icosahedral shells scaled to `mantle_radius` form the tetrahedral ball topology and spatial vertices (`Point-[V]`)."""
    topology, coordinates = shapes.ball(n_levels, n_shells)
    return topology, mv(core.Plane, mantle_radius * coordinates).dual().field()


def outer_core(topology: core.Topology, vertices: core.Point) -> tuple[core.Topology, core.Point, core.Sparse]:
    """Tetrahedra between `inner_radius` and unit radius form the fluid outer-core subcomplex, its vertices (`Point-[V_fluid]`), and its tetrahedral embedding."""
    V, E, F, T = topology.n_elements
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    radii = centres.antinorm()                                                  # Scalar+[T]
    fluid_mask = (radii > inner_radius) & (radii < 1)                           # [T]
    simplices = topology.elements[3][fluid_mask]                                # [..., 4]
    first, last = simplices.min(), simplices.max()
    fluid_vertices = topology.reading(np.arange(first, last + 1), 0) * vertices # Point-[V_fluid]
    embedding = topology.reading(np.flatnonzero(fluid_mask), 3).adjugate()      # Scalar+[T <- T_fluid]
    return core.Topology(last + 1 - first, simplices - first), fluid_vertices, embedding


# --- math -----------------------------------------------------------------------------
def spheromak_flow(points: core.Point, degree: int, root: float) -> core.Point:
    """Chandrasekhar–Kendall Beltrami mode of `degree` and Bessel `root` about the z axis evaluates tangent velocities (`[...] Point-`) in the unit ball."""
    radii = points.antinorm()                                                   # [...] Scalar+
    cosines = (mv.z & points) / radii                                           # [...] Scalar+
    phases = (mv.xy * (root * radii)).exp()                                     # [...] Rotor+
    sines = phases.scalar_product(~mv.xy)                                       # [...] Scalar+
    cosine_phases = phases.scalar_product(mv.scalar([1]))                       # [...] Scalar+
    radial_phases = root * radii                                                # [...] Scalar+
    bessels = (
        sines / radial_phases,
        sines / (radial_phases * radial_phases) - cosine_phases / radial_phases,
        (3 / (radial_phases * radial_phases) - 1) * sines / radial_phases
        - 3 * cosine_phases / (radial_phases * radial_phases),
    )
    bessel = bessels[degree]                                                    # [...] Scalar+
    radial_slope = radial_phases * bessels[degree - 1] - degree * bessel        # [...] Scalar+
    legendre, meridian_slope = ((cosines, -1), ((3 * cosines * cosines - 1) / 2, -3 * cosines))[degree - 1]
    radial_directions, north_axis = points / radii, mv.z.dual()                 # [...] Point-, Point-
    poloidal = (
        radial_directions * (degree * (degree + 1) * bessel * legendre / radii)
        + (radial_directions * cosines - north_axis) * (meridian_slope * radial_slope / radii)
    )                                                                           # [...] Point-
    toroidal = points.commutator(mv.xy) * (-root * bessel * meridian_slope / radii)  # [...] Point-
    return poloidal + toroidal


def helical_flow(topology: core.Topology, vertices: core.Point) -> core.Point:
    """Orthogonal degree-1 and degree-2 spheromaks inside the unit core combine at unit peak speed (`Point-[T]`)."""
    first_root, second_root = 4.493409457909064, 5.763459196894550
    axis_rotor = (mv.zx * (np.pi / 4)).exp()                                    # Rotor+
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    turned_centres = ~axis_rotor >> centres                                     # Point-[T]
    tilted_mode = axis_rotor >> spheromak_flow(turned_centres, 2, second_root)  # Point-[T]
    core_flow = spheromak_flow(centres, 1, first_root) + tilted_mode            # Point-[T]
    core_mask = mv.scalar((centres.antinorm() < 1)[:, None]).field()            # Scalar+[T]
    masked_flow = core_flow * core_mask                                         # Point-[T]
    return masked_flow / masked_flow.antinorm().batch().max()


def seed_potential(topology: core.Topology, vertices: core.Point, field: core.Magnetic) -> core.Scalar:
    """Uniform magnetic flux through origin-to-edge triangles defines initial edge potentials (`Scalar+[E]`)."""
    tails, heads = topology.incidence[1, 0] * vertices                          # Point-[E], Point-[E]
    return (field & (tails & heads)) / 2


def field_frames(curl: core.SparseExtensor, mode_velocities: core.Point,
                 states: Iterable[tuple[core.Scalar, core.Scalar]]) -> Iterator[tuple[core.Magnetic, core.Point]]:
    """Yields tetrahedral magnetic planes (`Magnetic+[T]`) and fluid velocities (`Point-[T]`) for each streamed state."""
    for amplitudes, potential in states:
        yield curl(potential), (amplitudes * mode_velocities).sum(axis=0)


def main() -> None:
    from itertools import islice, tee
    from examples.electromagnetism.dynamo import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices = ball_complex(n_levels, n_shells)
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    volumes, whitney, curl = core.whitney_operators(topology, vertices)
    stiffness = core.magnetic_stiffness(volumes, curl)                          # (Antiscalar- <- Scalar+)[E <- E]
    fluid_topology, fluid_vertices, embedding = outer_core(topology, vertices)
    decay_rates, mode_velocities, mode_vorticities = core.flow_modes(
        fluid_topology, fluid_vertices, embedding, n_scalar_modes, n_modes
    )
    save_figure(render.modes_figure(centres, mantle_radius, mode_velocities[shown_modes]), "dynamo_modes")

    target_momentum = volumes * helical_flow(topology, vertices).dual()         # Plane+[T]
    target_amplitudes = (target_momentum & mode_velocities).sites.sum()         # [n_modes] Scalar+
    radii = centres.antinorm()                                                  # Scalar+[T]
    conductivity = mv.scalar(np.where(radii < 1, reynolds, reynolds * mantle_conductance)[:, None]).field()  # Scalar+[T]
    initial_potential = seed_potential(topology, vertices, seed_field)          # Scalar+[E]
    states = core.dynamo_states(
        volumes, whitney, curl, decay_rates, mode_velocities, mode_vorticities,
        conductivity, viscosity, pull, target_amplitudes, initial_potential, dt, n_steps,
    )
    cutaway_stream, energy_stream = tee(states, 2)
    save_animation(
        render.cutaway_animation(
            centres, mantle_radius,
            field_frames(curl, mode_velocities, islice(cutaway_stream, n_skipped_steps, None, stride)),
        ),
        "dynamo", frame_duration_ms, colors=n_cutaway_colours,
    )
    n_energy_frames = n_steps // stride + 1
    amplitudes, potentials = map(stack, zip(*islice(energy_stream, 0, None, stride)))  # [n_energy_frames, n_modes] Scalar+, [n_energy_frames] Scalar+[E]
    kinetic_energy = (amplitudes * amplitudes).sum(axis=-1) / 2                 # [n_energy_frames] Scalar+
    magnetic_energy = (potentials & stiffness(potentials)).sites.sum() / 2      # [n_energy_frames] Scalar+
    times = np.arange(0, n_steps + 1, stride) * dt                              # [n_energy_frames]
    save_figure(render.energies_figure(times, kinetic_energy, magnetic_energy), "dynamo_energies")

    # checks
    final_velocity = (amplitudes[-1] * mode_velocities).sum(axis=0)             # Point-[T]
    volume_energy = (volumes * (final_velocity.dual() & final_velocity)).sites.sum() / 2  # Scalar+
    assert (volume_energy - kinetic_energy[-1]).abs() < 1e-10 * kinetic_energy[-1]
    target_velocity = (target_amplitudes * mode_velocities).sum(axis=0)         # Point-[T]
    _, advection_rates = core.flow_advection(
        volumes, curl, mode_velocities, mode_vorticities, target_velocity, amplitudes[-1], potentials[-1]
    )
    power_scale = ((advection_rates * advection_rates).sum() * (amplitudes[-1] * amplitudes[-1]).sum()).square_root()
    assert (amplitudes[-1] * advection_rates).sum().abs() < 1e-10 * power_scale
    face_fluxes = topology.d[1] * potentials[-1]                                # Scalar+[F]
    assert ((topology.d[2] * face_fluxes).norm().batch() < 1e-10 * face_fluxes.norm().batch().max()).all()
    assert magnetic_energy[-1] > 100 * magnetic_energy[0]
    assert kinetic_energy.min() < 0.75 * kinetic_energy[0]


def pinned() -> None:
    """Gallery animation of saturated outer-core magnetic field lines and streamlines seen from above."""
    from itertools import islice
    from examples.electromagnetism.dynamo import render
    from examples.util.animation import pin_animation

    topology, vertices = ball_complex(n_levels, n_shells)
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    volumes, whitney, curl = core.whitney_operators(topology, vertices)
    fluid_topology, fluid_vertices, embedding = outer_core(topology, vertices)
    decay_rates, mode_velocities, mode_vorticities = core.flow_modes(
        fluid_topology, fluid_vertices, embedding, n_scalar_modes, n_modes
    )
    target_momentum = volumes * helical_flow(topology, vertices).dual()         # Plane+[T]
    target_amplitudes = (target_momentum & mode_velocities).sites.sum()         # [n_modes] Scalar+
    radii = centres.antinorm()                                                  # Scalar+[T]
    conductivity = mv.scalar(np.where(radii < 1, reynolds, reynolds * mantle_conductance)[:, None]).field()  # Scalar+[T]
    initial_potential = seed_potential(topology, vertices, seed_field)          # Scalar+[E]
    states = core.dynamo_states(
        volumes, whitney, curl, decay_rates, mode_velocities, mode_vorticities,
        conductivity, viscosity, pull, target_amplitudes, initial_potential, dt, n_steps,
    )
    pin_animation(
        render.lines_animation(centres, field_frames(curl, mode_velocities, islice(states, n_skipped_steps, None, stride))),
        "dynamo_lines", frame_duration_ms, colors=n_lines_colours,
    )


if __name__ == "__main__":
    main()
