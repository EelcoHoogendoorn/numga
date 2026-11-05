"""Mixed-grade rotor Whitney forms: harmonic frame fields, math that imports no plotting, and the Darboux triangle,
the skyrmion and the saddle's frames on the smallest meshes, drawn."""

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

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
    topology, vertices, boundary, prescribed_rotors, initial_rotors = scenarios.saddle_surface(1.0, 5, 0.28)
    rotor_d0, rotor_laplacian, edge_weights, areas, sides = core.rotor_laplacian_operators(topology, vertices)

    # Whitney 0-form gradient pullback matches the cotangent Laplacian on the curved 3D saddle surface.
    patches = (sides[0] & sides[1]) / 2
    corner_gradients = (sides | patches.inverse()) / 2
    gradient = (spdiag(corner_gradients) * topology.incidence[2, 0]).sum(axis=0) * core.Scalar
    whitney_stiffness = gradient.adjugate()((areas * core.Plane.dual())(gradient))         # (Amount- <- Scalar+)[vertices <- vertices]
    scalar_laplacian = ~topology.d[0] * (spdiag(edge_weights * core.Scalar.dual()) * topology.d[0])   # (Amount- <- Scalar+)[vertices <- vertices]
    test_scalar = (core.mv.x & vertices).squared() - 0.4 * (core.mv.z & vertices)
    np.testing.assert_allclose(
        (whitney_stiffness(test_scalar) - scalar_laplacian(test_scalar)).kernel,
        0.0, atol=1e-14,
    )

    harmonic_chords, harmonic_rotors = core.harmonic_rotor_frames(
        topology, vertices, boundary, prescribed_rotors,
    )
    assert harmonic_chords.batch().shape == (len(scenarios.boundary_tilt_rates), topology.n_vertices)
    assert harmonic_rotors.batch().gatype == core.Rotor

    # The chord field satisfies the linear Laplace equation; normalized frames preserve boundary values.
    free_vertices = topology.interior[0]
    free = ~topology.reading(free_vertices, 0) * core.Rotor
    np.testing.assert_allclose(free.adjoint()(rotor_laplacian(harmonic_chords)).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(
        (topology.reading(boundary, 0) * (harmonic_rotors - prescribed_rotors)).kernel,
        0.0, atol=1e-13,
    )
    np.testing.assert_allclose(harmonic_rotors.scalar_norm_squared().kernel, 1.0, atol=1e-14)

    # Projected relaxation keeps unit norm at every vertex and strictly reduces Dirichlet energy.
    trajectory = list(core.relax_rotor_frames(topology, vertices, initial_rotors, boundary, 0.18, 12))
    energies = [float(step_energy.to_array()) for _, step_energy in trajectory]
    assert all(energies[i + 1] < energies[i] for i in range(len(energies) - 1))
    for step_rotors, _ in trajectory:
        np.testing.assert_allclose(step_rotors.scalar_norm_squared().kernel, 1.0, atol=1e-14)
        np.testing.assert_allclose(
            (topology.reading(boundary, 0) * (step_rotors - initial_rotors)).kernel,
            0.0, atol=1e-14,
        )


def test_the_scenes_run_and_draw():
    corners, points, fine_topology, fine_vertices, corner_rotors = scenarios.single_triangle(3, 1)
    _, unit_rotors, _, whitney_rates, _, _, _ = core.triangle_darboux(corners, points, corner_rotors)
    fine_chords, _, _, _, _, _, _ = core.triangle_darboux(corners, fine_vertices.batch(), corner_rotors)
    plt.close(render.draw_darboux_whitney(corners, points, fine_topology, fine_vertices,
                                          fine_chords.scalar_norm_squared().field(), corner_rotors, unit_rotors,
                                          whitney_rates))
    topology, coordinates = shapes.triangular_disk(scenarios.disk_radius, 2)
    vertices = scenarios.mv(scenarios.Upright, coordinates).dual().field()      # Point-[vertices]
    _, directors, _, _, _, _ = core.skyrmion_winding(topology, vertices, scenarios.skyrmion_twist)
    plt.close(render.draw_skyrmion(topology, vertices, directors))
    topology, vertices, boundary, prescribed_rotors, initial_rotors = scenarios.saddle_surface(
        scenarios.disk_radius, 2, scenarios.saddle_height)
    _, harmonic_rotors = core.harmonic_rotor_frames(topology, vertices, boundary, prescribed_rotors)
    plt.close(render.draw_rotor_frames(topology, vertices, harmonic_rotors))
    states = core.relax_rotor_frames(topology, vertices, initial_rotors, boundary, scenarios.relaxation_step_size, 1)
    assert len(list(render.animate_relaxation(topology, vertices, states))) >= 1
