"""Mixed-grade rotor Whitney forms, harmonic frame fields, plotting isolation, and minimal scene renders."""

import subprocess
import sys

import matplotlib.pyplot as plt

from numga import stack
from numga.sparse import spdiag
from examples.util import shapes
from examples.surfaces.frame_fields import core, render, scenarios


def test_frame_fields_math_does_not_import_plotting():
    probe = (
        "import sys, examples.surfaces.frame_fields.core, examples.surfaces.frame_fields.scenarios; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_harmonic_rotor_frames_and_projected_relaxation():
    topology, vertices, boundary_indices, prescribed_rotors, initial_rotors = scenarios.saddle_surface(1.0, 4, 0.28)
    V, E, F = topology.n_elements
    rotor_d0, rotor_laplacian, edge_weights, areas, sides = core.rotor_laplacian_operators(topology, vertices)

    # Simplex-gradient pullback matches the scalar cotangent Laplacian on the curved 3D saddle surface.
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    gradients = (sides | patches.inverse()) / 2                              # [3] Plane+[F]
    gradient = (spdiag(gradients) * topology.incidence[2, 0]).sum(axis=0) * core.Scalar
    whitney_stiffness = gradient.adjugate()((areas * core.Plane.dual())(gradient))
    scalar_laplacian = topology.d[0].adjugate() * (spdiag(edge_weights * core.Scalar.dual()) * topology.d[0])
    test_scalar = (core.mv.x & vertices).squared() - 0.4 * (core.mv.z & vertices)  # Scalar+[V]
    assert ((whitney_stiffness(test_scalar) - scalar_laplacian(test_scalar)).scalar_norm_squared() < 1e-24).all()

    harmonic_chords, harmonic_rotors = core.harmonic_rotor_frames(
        topology, vertices, boundary_indices, prescribed_rotors,
    )
    assert harmonic_chords.batch().shape == (scenarios.n_cases, V)
    assert harmonic_rotors.batch().gatype == core.Rotor

    # Harmonic chords satisfy the interior Laplace equation while normalized rotors preserve boundary frames.
    free_inclusion = topology.reading(topology.interior[0], 0).adjugate() * core.Rotor
    assert (free_inclusion.adjoint()(rotor_laplacian(harmonic_chords)).scalar_norm_squared() < 1e-20).all()
    assert ((topology.reading(boundary_indices, 0) * (harmonic_rotors - prescribed_rotors)).scalar_norm_squared() < 1e-24).all()
    assert ((harmonic_rotors.scalar_norm_squared() - 1).abs() < 1e-12).all()

    # Projected relaxation keeps unit norm at every vertex, holds the boundary, and strictly lowers Dirichlet energy.
    trajectory_rotors = stack(list(core.rotor_relaxation(topology, vertices, initial_rotors, 0.18, 6)))  # [7] Rotor+[V]
    energies = (edge_weights * rotor_d0(trajectory_rotors).scalar_norm_squared()).sites.sum() / 2        # [7] Scalar+
    assert (energies[1:] < energies[:-1]).all()
    assert ((trajectory_rotors.scalar_norm_squared() - 1).abs() < 1e-12).all()
    assert ((topology.reading(boundary_indices, 0) * (trajectory_rotors - initial_rotors)).scalar_norm_squared() < 1e-24).all()


def test_the_scenes_run_and_draw():
    corners, points, fine_topology, fine_vertices, corner_rotors = scenarios.single_triangle(3, 1)
    _, unit_rotors, _, whitney_rates, _, _, _ = core.triangle_darboux(corners, points, corner_rotors)
    fine_chords, _, _, _, _, _, _ = core.triangle_darboux(corners, fine_vertices.batch(), corner_rotors)
    plt.close(render.darboux_figure(corners, points, fine_topology, fine_vertices,
                                    fine_chords.scalar_norm_squared().field(), corner_rotors, unit_rotors,
                                    whitney_rates))
    topology, coordinates = shapes.triangular_disk(scenarios.disk_radius, 2)
    vertices = scenarios.mv(scenarios.Upright, coordinates).dual().field()   # Point-[V]
    _, directors, _, _, _, _ = core.skyrmion_winding(topology, vertices, scenarios.skyrmion_twist)
    plt.close(render.skyrmion_figure(topology, vertices, directors))
    topology, vertices, boundary_indices, prescribed_rotors, initial_rotors = scenarios.saddle_surface(
        scenarios.disk_radius, 2, scenarios.saddle_height)
    _, harmonic_rotors = core.harmonic_rotor_frames(topology, vertices, boundary_indices, prescribed_rotors)
    plt.close(render.rotor_frames_figure(topology, vertices, harmonic_rotors))
    states = core.rotor_relaxation(topology, vertices, initial_rotors, scenarios.relaxation_step_size, 1)
    assert len(list(render.relaxation_animation(topology, vertices, states))) >= 1
