"""Mixed-grade rotor Whitney forms: Darboux grade split, director winding, and harmonic frame fields."""

import subprocess
import sys

import numpy as np

from numga import stack
from numga.sparse import spdiag
from examples.surfaces.frame_fields import core, scenarios


def test_frame_fields_math_does_not_import_plotting():
    probe = (
        "import sys, examples.surfaces.frame_fields.core, examples.surfaces.frame_fields.scenarios; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_darboux_whitney_grade_decomposition_and_maurer_cartan():
    corners, points, _, _, corner_rotors = scenarios.single_triangle(8, 8)
    (
        chord_rotors, unit_rotors, darboux, whitney_rates, chord_stretch, angular_velocity, whitney_forms,
    ) = core.triangle_darboux(corners, points, corner_rotors)

    assert chord_rotors.gatype == core.Even
    assert unit_rotors.gatype == core.Rotor
    assert darboux.gatype == core.EvenMap
    assert whitney_rates.gatype == core.RateMap
    assert chord_stretch.gatype == core.StretchMap
    assert angular_velocity.gatype == core.RateMap

    # Grade-2 and Grade-0 parts of 2 * ~R(x) * dR match the Whitney 1-forms and Regge symmetric differentials.
    np.testing.assert_allclose((darboux.select[2] - whitney_rates).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((darboux.select[0] - chord_stretch).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(unit_rotors.scalar_norm_squared().kernel, 1.0, atol=1e-14)

    # At the three triangle corners, the chord rotor is unit and matches the prescribed corner rotors.
    corner_chords, _, corner_darboux, corner_rates, corner_stretch, _, _ = core.triangle_darboux(
        corners, corners, corner_rotors,
    )
    np.testing.assert_allclose((corner_chords - corner_rotors).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(corner_chords.scalar_norm_squared().kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose((corner_darboux.select[2] - corner_rates).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((corner_darboux.select[0] - corner_stretch).kernel, 0.0, atol=1e-14)

    # In the interior, the 4D chord cuts strictly inside S^3 so its norm squared is strictly below 1.
    assert np.all(chord_rotors.scalar_norm_squared().to_array() < 1.0 - 1e-3)

    # Antisymmetric combination lambda_i g_j - lambda_j g_i matches the radial contraction with the inverse patch.
    sides = (core.triangle.boundary[2, 1] * (core.triangle.d[0] * corners.field())).batch()[:, 0]
    patch = (sides[0] & sides[1]) / 2
    radial_whitney = (((points[:, None] - corners) | patch.inverse()) / 2) * core.triangle.face_edge_orientation
    np.testing.assert_allclose((whitney_forms - radial_whitney).kernel, 0.0, atol=1e-14)

    # Dividing the grade-2 Whitney rate by |R(x)|^2 gives the exact Maurer-Cartan 1-form 2 * ~R_unit * dR_unit.
    step = 1e-7 * (0.6 * sides[0] - 0.35 * sides[1])
    _, shifted_units, _, _, _, _, _ = core.triangle_darboux(corners, points + step, corner_rotors)
    finite_diff_mc = 2 * (unit_rotors.reverse() * (shifted_units - unit_rotors)).select[2]
    exact_mc = angular_velocity(step)
    np.testing.assert_allclose((finite_diff_mc - exact_mc).kernel, 0.0, atol=1e-12)

    # Constant corner rotors yield zero Whitney rate, zero chord stretch, and unit norm everywhere.
    constant_rotors = stack((corner_rotors[0], corner_rotors[0], corner_rotors[0]))
    const_chords, _, _, const_rates, const_stretch, _, _ = core.triangle_darboux(
        corners, points, constant_rotors,
    )
    np.testing.assert_allclose(const_chords.scalar_norm_squared().kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose(const_rates.kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(const_stretch.kernel, 0.0, atol=1e-14)


def test_skyrmion_topological_charge_and_cotangent_trace():
    mesh, vertices = core.Mesh.triangular_disk(1.0, 6)
    (
        rotors, directors, scalar_parts, bivector_norms, energy_density, solid_angles,
    ) = core.skyrmion_winding(mesh, vertices, np.pi)

    assert rotors.batch().gatype == core.Rotor
    assert directors.batch().gatype == core.Point
    assert scalar_parts.batch().gatype == core.Scalar
    assert energy_density.batch().gatype == core.Scalar

    # Unit norm decomposition <R>_0^2 + |<R>_2|^2 = 1 at every vertex, +1 at core, 0 scalar at rim.
    np.testing.assert_allclose(
        (scalar_parts.squared() + bivector_norms.squared()).kernel, 1.0, atol=1e-14,
    )
    np.testing.assert_allclose(scalar_parts.batch()[0].kernel, 1.0, atol=1e-14)
    boundary = mesh.exterior(0)
    np.testing.assert_allclose(scalar_parts.batch()[boundary].kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((core.mv.z & directors.batch()[0]).kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose((core.mv.z & directors.batch()[boundary]).kernel, -1.0, atol=1e-13)

    # Basis-free cotangent trace of Maurer-Cartan 1-form matches orthonormal ambient trace on every face.
    edges = mesh.d[0] * vertices
    sides = mesh.boundary[2, 1] * edges
    patches = (sides[0] & sides[1]) / 2
    corner_gradients = (sides | patches.inverse()) / 2
    face_rotors = mesh.incidence[2, 0] * rotors
    chord_rotors = face_rotors.sum(axis=0) / 3
    face_differential = (face_rotors * (corner_gradients & core.Point)).sum(axis=0)
    face_rates = (2 * chord_rotors.reverse() * face_differential).select[2] / chord_rotors.scalar_norm_squared()
    basis_points = stack((core.mv.x.dual(), core.mv.y.dual(), core.mv.z.dual()))
    orthonormal_trace = face_rates(basis_points).scalar_norm_squared().sum(axis=0)
    np.testing.assert_allclose((energy_density - orthonormal_trace).kernel, 0.0, atol=1e-13)

    # Topological Skyrmion charge integrates to Q = 1 and refines with mesh resolution.
    coarse_mesh, coarse_vertices = core.Mesh.triangular_disk(1.0, 4)
    _, _, _, _, _, coarse_solid_angles = core.skyrmion_winding(coarse_mesh, coarse_vertices, np.pi)
    coarse_charge = float((coarse_solid_angles.sites.sum() / (4 * np.pi)).to_array())
    fine_charge = float((solid_angles.sites.sum() / (4 * np.pi)).to_array())
    assert abs(fine_charge - 1.0) < abs(coarse_charge - 1.0)
    np.testing.assert_allclose(fine_charge, 1.0, rtol=1.5e-2)

    # A full turn reaches the opposite rotor sign at the rim while returning the director to the north pole.
    rotors_2pi, directors_2pi, scalars_2pi, _, _, _ = core.skyrmion_winding(mesh, vertices, 2 * np.pi)
    np.testing.assert_allclose(scalars_2pi.batch()[boundary].kernel, -1.0, atol=1e-13)
    np.testing.assert_allclose((core.mv.z & directors_2pi.batch()[boundary]).kernel, 1.0, atol=1e-13)


def test_harmonic_rotor_frames_and_projected_relaxation():
    mesh, vertices, boundary, prescribed_rotors, initial_rotors = scenarios.saddle_surface(1.0, 5, 0.28)
    rotor_d0, rotor_laplacian, edge_weights, areas, sides = core.rotor_laplacian_operators(mesh, vertices)

    # Whitney 0-form gradient pullback matches the cotangent Laplacian on the curved 3D saddle surface.
    patches = (sides[0] & sides[1]) / 2
    corner_gradients = (sides | patches.inverse()) / 2
    gradient = (spdiag(corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * core.Scalar
    whitney_stiffness = gradient.adjoint()(spdiag(areas) * gradient)
    scalar_laplacian = ~mesh.d[0] * (spdiag(edge_weights) * mesh.d[0])
    test_scalar = (core.mv.x & vertices).squared() - 0.4 * (core.mv.z & vertices)
    np.testing.assert_allclose(
        (whitney_stiffness(test_scalar) - scalar_laplacian * test_scalar).kernel,
        0.0, atol=1e-14,
    )

    harmonic_chords, harmonic_rotors = core.harmonic_rotor_frames(
        mesh, vertices, boundary, prescribed_rotors,
    )
    assert harmonic_chords.batch().shape == (len(scenarios.boundary_tilt_rates), mesh.n_vertices)
    assert harmonic_rotors.batch().gatype == core.Rotor

    # The chord field satisfies the linear Laplace equation; normalized frames preserve boundary values.
    free_vertices = mesh.interior(0)
    free = ~mesh.reading(free_vertices) * core.Rotor
    np.testing.assert_allclose(free.adjoint()(rotor_laplacian(harmonic_chords)).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(
        (harmonic_rotors.batch()[:, boundary] - prescribed_rotors.batch()[:, boundary]).kernel,
        0.0, atol=1e-13,
    )
    np.testing.assert_allclose(harmonic_rotors.scalar_norm_squared().kernel, 1.0, atol=1e-14)

    # Projected relaxation keeps unit norm at every vertex and strictly reduces Dirichlet energy.
    trajectory = list(core.relax_rotor_frames(mesh, vertices, initial_rotors, boundary, 0.18, 12))
    energies = [float(step_energy.to_array()) for _, step_energy in trajectory]
    assert all(energies[i + 1] < energies[i] for i in range(len(energies) - 1))
    for step_rotors, _ in trajectory:
        np.testing.assert_allclose(step_rotors.scalar_norm_squared().kernel, 1.0, atol=1e-14)
        np.testing.assert_allclose(
            (step_rotors.batch()[boundary] - initial_rotors.batch()[boundary]).kernel,
            0.0, atol=1e-14,
        )
