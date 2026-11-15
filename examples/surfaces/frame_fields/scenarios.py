"""Reference 3D triangle, skyrmion disk, and curved saddle surface for smooth rotor frame fields."""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import VGA3D as ga
from examples.util import shapes
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Rotor = ga.gatype.rotor()                                                    # Rotor+
Point = ga.gatype.antivector()                                               # Antivector-
Upright = ga.gatype.vector().derive.within(Algebra("x+y+"))                  # Vector+

n_triangle_subdivisions = 6
n_fine_levels = 4
n_skyrmion_divisions = 12
n_saddle_divisions = 6
disk_radius = 1.0
skyrmion_twist = np.pi
saddle_height = 0.28
boundary_tilt_rates = np.array([0.35, 0.85, -0.75])                          # [3]
n_cases = len(boundary_tilt_rates)
boundary_twist_rates = np.array([0.0, 1.15, -1.15])                          # [3]
relaxation_step_size = 0.08
n_relaxation_steps = 45
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def triangle_samples(corners: Point, n_subdivisions: int) -> Point:
    """Interior simplex-grid points (`[...] Point-`) inside a 3D triangle (`[3] Point-`)."""
    steps = np.arange(1, n_subdivisions)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    inside = first + second < n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside],
                        first[inside], second[inside]), axis=-1) / n_subdivisions  # [..., 3]
    return (corners * weights).sum(axis=-1)                                  # [...] Point-


def single_triangle(n_subdivisions: int, n_levels: int) -> tuple[Point, Point, Topology, Point, Rotor]:
    """Tilted 3D triangle corners (`[3] Point-`), interior sample lattice (`[...] Point-`), subdivided surface topology and vertices (`Point-[V_fine]`), and three corner rotors (`[3] Rotor+`)."""
    corner_coordinates = np.array([[0.0, 0.0, 0.05], [1.15, 0.14, -0.12], [0.36, 0.98, 0.24]])  # [3, 3]
    corners = mv.vector(corner_coordinates).dual()                           # [3] Point-
    corner_rotors = stack((
        (mv.xy * (-0.35 / 2) + mv.yz * (0.25 / 2)).exp(),
        (mv.yz * (-1.25 / 2) + mv.zx * (0.55 / 2)).exp(),
        (mv.zx * (-1.35 / 2) - mv.xy * (0.65 / 2)).exp(),
    ))                                                                       # [3] Rotor+
    fine_topology, fine_coordinates = shapes.midpoint_split(Topology(3, [[0, 1, 2]]), corner_coordinates, n_levels)
    fine_vertices = mv.vector(fine_coordinates).dual().field()               # Point-[V_fine]
    return corners, triangle_samples(corners, n_subdivisions), fine_topology, fine_vertices, corner_rotors


def saddle_surface(
    radius: float, n_divisions: int, arch_height: float,
) -> tuple[Topology, Point, np.ndarray, Rotor, Rotor]:
    """Curved 3D saddle topology and vertices (`Point-[V]`), boundary vertex indices, three boundary rotor prescriptions (`[n_cases] Rotor+[V]`), and a twisted initial rotor field (`Rotor+[V]`)."""
    topology, flat_coordinates = shapes.triangular_disk(radius, n_divisions)
    flat_vertices = mv(Upright, flat_coordinates).dual().field()             # Point-[V]
    vertical = mv.z.dual()                                                   # [] Point-
    x_coord, y_coord = mv.x & flat_vertices, mv.y & flat_vertices            # Scalar+[V], Scalar+[V]
    vertices = flat_vertices + arch_height * (x_coord.squared() - y_coord.squared()) * vertical  # Point-[V]
    boundary_indices = topology.exterior[0]

    prescribed_rotors = (
        (-boundary_tilt_rates / 2) * flat_vertices.commutator(vertical)
        + (boundary_twist_rates / 2) * (x_coord * mv.yz + y_coord * mv.zx)
    ).exp()                                                                  # [n_cases] Rotor+[V]

    envelope = 1 - flat_vertices.norm_squared()                              # Scalar+[V]
    interior_twist = (
        envelope * (
            1.65 * flat_vertices.commutator(vertical)
            + (1.55 * x_coord - 1.25 * y_coord) * mv.xy
            + 2.25 * x_coord * y_coord * (mv.yz - mv.zx)
        )
    ).exp()                                                                  # Rotor+[V]
    initial_rotors = interior_twist * prescribed_rotors.batch()[1].field()   # Rotor+[V]
    return topology, vertices, boundary_indices, prescribed_rotors, initial_rotors


def main() -> None:
    from examples.surfaces.frame_fields.core import (
        harmonic_rotor_frames, rotor_laplacian_operators, rotor_relaxation,
        skyrmion_winding, triangle, triangle_darboux,
    )
    from examples.surfaces.frame_fields import render
    from examples.util.animation import save_animation, save_figure

    # 1. Mixed-grade rotor interpolation and Darboux–Whitney identity on a 3D triangle.
    corners, points, fine_topology, fine_vertices, corner_rotors = single_triangle(
        n_triangle_subdivisions, n_fine_levels,
    )
    _, unit_rotors, darboux, whitney_rates, chord_stretch, _, whitney_forms = triangle_darboux(
        corners, points, corner_rotors,
    )
    fine_chords, _, _, _, _, _, _ = triangle_darboux(corners, fine_vertices.batch(), corner_rotors)
    save_figure(
        render.darboux_figure(
            corners, points, fine_topology, fine_vertices, fine_chords.scalar_norm_squared().field(),
            corner_rotors, unit_rotors, whitney_rates,
        ),
        "frame_fields_darboux",
    )

    # 2. Planar disk wrapping the sphere of directors once.
    skyrmion_topology, skyrmion_coordinates = shapes.triangular_disk(disk_radius, n_skyrmion_divisions)
    skyrmion_vertices = mv(Upright, skyrmion_coordinates).dual().field()     # Point-[V]
    _, directors, scalar_parts, bivector_norms, _, solid_angles = skyrmion_winding(
        skyrmion_topology, skyrmion_vertices, skyrmion_twist,
    )
    save_figure(
        render.skyrmion_figure(skyrmion_topology, skyrmion_vertices, directors),
        "frame_fields_skyrmion",
    )

    # 3. Harmonic chord normalization and unit-rotor projected smoothing on a curved saddle.
    saddle_topology, saddle_vertices, boundary_indices, prescribed_rotors, initial_rotors = saddle_surface(
        disk_radius, n_saddle_divisions, saddle_height,
    )
    harmonic_chords, harmonic_rotors = harmonic_rotor_frames(
        saddle_topology, saddle_vertices, boundary_indices, prescribed_rotors,
    )
    save_figure(
        render.rotor_frames_figure(saddle_topology, saddle_vertices, harmonic_rotors),
        "frame_fields_frames",
    )

    streamed_rotors = []
    def tracked_states():
        for state in rotor_relaxation(
            saddle_topology, saddle_vertices, initial_rotors,
            relaxation_step_size, n_relaxation_steps,
        ):
            streamed_rotors.append(state)
            yield state

    save_animation(
        render.relaxation_animation(saddle_topology, saddle_vertices, tracked_states()),
        "frame_fields_relaxation",
        frame_duration_ms,
    )

    # checks
    probe_step = corners[1] - corners[0]                                     # [] Point-
    assert ((darboux(probe_step).select[2] - whitney_rates(probe_step)).scalar_norm_squared() < 1e-24).all()
    assert ((darboux(probe_step).select[0] - chord_stretch(probe_step)).abs() < 1e-12).all()
    assert ((unit_rotors.scalar_norm_squared() - 1).abs() < 1e-12).all()

    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]  # [3] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    radial_whitney = (((points[:, None] - corners) | patch.inverse()) / 2) * triangle.orientation[1]  # [..., 3] Plane+
    assert ((whitney_forms - radial_whitney).scalar_norm_squared() < 1e-24).all()

    skyrmion_charge = solid_angles.sites.sum() / (4 * np.pi)                 # [] Scalar+
    assert bool((skyrmion_charge - 1).abs() < 1.5e-2)
    assert ((scalar_parts.squared() + bivector_norms.squared() - 1).abs() < 1e-12).all()

    rotor_d0, rotor_laplacian, edge_weights, _, _ = rotor_laplacian_operators(saddle_topology, saddle_vertices)
    free_inclusion = saddle_topology.reading(saddle_topology.interior[0], 0).adjugate() * Rotor
    assert (free_inclusion.adjoint()(rotor_laplacian(harmonic_chords)).scalar_norm_squared() < 1e-20).all()

    trajectory_rotors = stack(streamed_rotors)                               # [n_relaxation_steps + 1] Rotor+[V]
    relaxation_energies = (edge_weights * rotor_d0(trajectory_rotors).scalar_norm_squared()).sites.sum() / 2  # [n_relaxation_steps + 1] Scalar+
    assert (relaxation_energies[1:] < relaxation_energies[:-1]).all()


if __name__ == "__main__":
    main()
