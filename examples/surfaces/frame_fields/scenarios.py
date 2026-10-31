"""Reference 3D triangle, Skyrmion disk, and curved saddle surface for smooth frame fields of interpolated rotors."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.algebras import VGA3D as ga
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Rotor = ga.gatype.rotor()                                                    # Rotor+
Point = ga.gatype.antivector()                                               # Antivector-

n_triangle_subdivisions = 6
n_fine_levels = 4
n_skyrmion_divisions = 12
n_saddle_divisions = 6
disk_radius = 1.0
skyrmion_twist = np.pi
saddle_height = 0.28
boundary_tilt_rates = np.array([0.35, 0.85, -0.75])                          # [cases]
boundary_twist_rates = np.array([0.0, 1.15, -1.15])                          # [cases]
relaxation_step_size = 0.08
n_relaxation_steps = 45
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def triangle_samples(corners: Point, n_subdivisions: int) -> Point:
    """Interior barycentric lattice points inside a 3D triangle."""
    steps = np.arange(1, n_subdivisions)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    inside = first + second < n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside],
                        first[inside], second[inside]), axis=-1) / n_subdivisions  # [samples, corners]
    return (corners * weights).sum(axis=-1)                                  # [samples] Point-


def single_triangle(n_subdivisions: int, n_fine_levels: int) -> tuple[Point, Point, Mesh, Point, Rotor]:
    """A tilted 3D triangle, its interior sample lattice, a fine surface mesh, and three corner rotors."""
    ex, ey, ez = mv.x.dual(), mv.y.dual(), mv.z.dual()                       # [] Point-, [] Point-, [] Point-
    corners = stack((
        ex * 0.0 + ey * 0.0 + ez * 0.05,
        ex * 1.15 + ey * 0.14 - ez * 0.12,
        ex * 0.36 + ey * 0.98 + ez * 0.24,
    ))                                                                       # [corners] Point-
    corner_rotors = stack((
        (mv.xy * (-0.35 / 2) + mv.yz * (0.25 / 2)).exp(),
        (mv.yz * (-1.25 / 2) + mv.zx * (0.55 / 2)).exp(),
        (mv.zx * (-1.35 / 2) - mv.xy * (0.65 / 2)).exp(),
    ))                                                                       # [corners] Rotor+
    # The triangle split at its edges' midpoints, for drawing the field smoothly across it.
    fine_mesh, fine_vertices = Mesh(3, np.array([[0, 1, 2]])).midpoint_split(corners.field(), n_fine_levels)
    return corners, triangle_samples(corners, n_subdivisions), fine_mesh, fine_vertices, corner_rotors


def saddle_surface(
    radius: float, n_divisions: int, arch_height: float,
) -> tuple[Mesh, Point, np.ndarray, Rotor, Rotor]:
    """A curved 3D saddle mesh with boundary indices, three boundary frame prescriptions, and a twisted initial field."""
    mesh, flat_vertices = Mesh.triangular_disk(radius, n_divisions)
    vertical = mv.z.dual()                                                   # [] Point-
    x_coord, y_coord = mv.x & flat_vertices, mv.y & flat_vertices            # Scalar+[vertices], Scalar+[vertices]
    vertices = flat_vertices + arch_height * (x_coord.squared() - y_coord.squared()) * vertical  # Point-[vertices]
    boundary = mesh.exterior(0)

    prescribed_rotors = (
        (-boundary_tilt_rates / 2) * flat_vertices.commutator(vertical)
        + (boundary_twist_rates / 2) * (x_coord * mv.yz + y_coord * mv.zx)
    ).exp()                                                                  # [cases] Rotor+[vertices]

    envelope = 1 - flat_vertices.norm_squared()                              # Scalar+[vertices]
    interior_twist = (
        envelope * (
            1.65 * flat_vertices.commutator(vertical)
            + (1.55 * x_coord - 1.25 * y_coord) * mv.xy
            + 2.25 * x_coord * y_coord * (mv.yz - mv.zx)
        )
    ).exp()                                                                  # Rotor+[vertices]
    initial_rotors = interior_twist * prescribed_rotors.batch()[1].field()   # Rotor+[vertices]
    return mesh, vertices, boundary, prescribed_rotors, initial_rotors


def main() -> None:
    from examples.surfaces.frame_fields.core import (
        harmonic_rotor_frames, relax_rotor_frames, rotor_laplacian_operators,
        skyrmion_winding, triangle, triangle_darboux,
    )
    from examples.surfaces.frame_fields import render
    from examples.util.animation import save_animation, save_figure

    # 1. Mixed-grade rotor interpolation and the Darboux-Whitney identity on a 3D triangle.
    corners, points, fine_mesh, fine_vertices, corner_rotors = single_triangle(
        n_triangle_subdivisions, n_fine_levels,
    )
    _, unit_rotors, darboux, whitney_rates, chord_stretch, _, whitney_forms = triangle_darboux(
        corners, points, corner_rotors,
    )
    fine_chords, _, _, _, _, _, _ = triangle_darboux(corners, fine_vertices.batch(), corner_rotors)
    save_figure(
        render.draw_darboux_whitney(
            corners, points, fine_mesh, fine_vertices, fine_chords.scalar_norm_squared().field(),
            corner_rotors, unit_rotors, whitney_rates,
        ),
        "frame_fields_darboux",
    )

    # 2. A disk wraps the sphere of directors once.
    skyrmion_mesh, skyrmion_vertices = Mesh.triangular_disk(disk_radius, n_skyrmion_divisions)
    _, directors, scalar_parts, bivector_norms, _, solid_angles = skyrmion_winding(
        skyrmion_mesh, skyrmion_vertices, skyrmion_twist,
    )
    save_figure(
        render.draw_skyrmion(skyrmion_mesh, skyrmion_vertices, directors),
        "frame_fields_skyrmion",
    )

    # 3. Normalize harmonic chords, then smooth a separate field under the unit-rotor constraint.
    saddle_mesh, saddle_vertices, boundary, prescribed_rotors, initial_rotors = saddle_surface(
        disk_radius, n_saddle_divisions, saddle_height,
    )
    harmonic_chords, harmonic_rotors = harmonic_rotor_frames(
        saddle_mesh, saddle_vertices, boundary, prescribed_rotors,
    )
    save_figure(
        render.draw_rotor_frames(saddle_mesh, saddle_vertices, harmonic_rotors),
        "frame_fields_frames",
    )

    relaxation_states = relax_rotor_frames(
        saddle_mesh, saddle_vertices, initial_rotors, boundary,
        relaxation_step_size, n_relaxation_steps,
    )
    save_animation(
        render.animate_relaxation(saddle_mesh, saddle_vertices, relaxation_states),
        "frame_fields_relaxation",
        frame_duration_ms,
        forward_then_back=True,
    )

    # checks
    np.testing.assert_allclose((darboux.select[2] - whitney_rates).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((darboux.select[0] - chord_stretch).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(unit_rotors.scalar_norm_squared().kernel, 1.0, atol=1e-14)

    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]
    patch = (sides[0] & sides[1]) / 2
    radial_whitney = (((points[:, None] - corners) | patch.inverse()) / 2) * triangle.face_edge_orientation
    np.testing.assert_allclose((whitney_forms - radial_whitney).kernel, 0.0, atol=1e-14)

    skyrmion_charge = solid_angles.sites.sum() / (4 * np.pi)
    np.testing.assert_allclose(skyrmion_charge.kernel, 1.0, rtol=1.5e-2)
    np.testing.assert_allclose(
        (scalar_parts.squared() + bivector_norms.squared()).kernel, 1.0, atol=1e-14,
    )

    _, rotor_laplacian, _, _, _ = rotor_laplacian_operators(saddle_mesh, saddle_vertices)
    free_vertices = saddle_mesh.interior(0)
    free = ~saddle_mesh.reading(free_vertices) * Rotor
    np.testing.assert_allclose(free.adjoint()(rotor_laplacian(harmonic_chords)).kernel, 0.0, atol=1e-11)

    relaxation_energies = stack([
        energy for _, energy in relax_rotor_frames(
            saddle_mesh, saddle_vertices, initial_rotors, boundary,
            relaxation_step_size, n_relaxation_steps,
        )
    ])                                                                      # [steps + 1] Scalar+
    assert np.all(relaxation_energies[1:] < relaxation_energies[:-1])


if __name__ == "__main__":
    main()
