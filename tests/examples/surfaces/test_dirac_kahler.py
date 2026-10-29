"""Mixed-grade Whitney reconstruction, Dirac–Kähler products, Hodge decomposition of a velocity, and wave dynamics."""

import numpy as np

from numga import stack
from examples.surfaces.dirac_kahler import core, scenarios


def test_mixed_grade_whitney_reconstruction_and_dirac_kahler_product_on_disk():
    mesh, vertices = scenarios.disk(1.0, 5)
    operators = core.Operators(mesh, vertices)

    assert operators.whitney0.gatype == core.ga.gatype((core.Scalar, core.Scalar))
    assert operators.whitney1.gatype == core.ga.gatype((core.Plane, core.Scalar))
    assert operators.whitney2.gatype == core.ga.gatype((core.Bivector, core.Scalar))
    assert operators.dirac.gatype == core.ga.gatype((core.Full, core.Full))
    assert operators.even_gradient.gatype == core.ga.gatype((core.Plane, core.Even))
    assert operators.even_divergence.gatype == core.ga.gatype((core.Even, core.Plane))

    x, y = core.mv.x & vertices, core.mv.y & vertices
    scalar_potential = 0.6 * (x.squared() - y.squared()) + 0.4 * x - 0.25 * y
    vertex_1form = (0.7 * x - 0.5 * y) * core.mv.x + (0.4 * x + 0.9 * y) * core.mv.y
    stream_bivector = (0.8 * x * y - 0.3 * x) * core.mv.xy
    vertex_multivector = scalar_potential + vertex_1form + stream_bivector
    assert vertex_multivector.batch().gatype == core.Full

    face_dirac = operators.dirac(vertex_multivector)
    assert face_dirac.batch().gatype == core.Full

    u_corners = mesh.incidence[2, 0] * vertex_1form
    omega_corners = mesh.incidence[2, 0] * stream_bivector
    expected_div = (operators.corner_gradients | u_corners).sum(axis=0)
    expected_grad_plus_rot = operators.gradient(scalar_potential) + (operators.corner_gradients | omega_corners).sum(axis=0)
    expected_curl = (operators.corner_gradients ^ u_corners).sum(axis=0)

    np.testing.assert_allclose((face_dirac.select[0] - expected_div).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((face_dirac.select[1] - expected_grad_plus_rot).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        (face_dirac.select[1] - operators.even_gradient(scalar_potential + stream_bivector)).kernel,
        0.0, atol=1e-14,
    )
    np.testing.assert_allclose((face_dirac.select[2] - expected_curl).kernel, 0.0, atol=1e-14)

    # Cochain reconstruction into a single Full multivector field on faces.
    edge_midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices
    mx, my = core.mv.x & edge_midpoints, core.mv.y & edge_midpoints
    midpoint_1form = (0.7 * mx - 0.5 * my) * core.mv.x + (0.4 * mx + 0.9 * my) * core.mv.y
    edge_cochain = midpoint_1form & (mesh.d[0] * vertices)
    face_cochain = operators.d1(edge_cochain)
    whitney_multivector = (
        operators.whitney0(scalar_potential)
        + operators.whitney1(edge_cochain)
        + operators.whitney2(face_cochain)
    )
    assert whitney_multivector.batch().gatype == core.Full
    np.testing.assert_allclose(
        (whitney_multivector.select[0] - operators.whitney0(scalar_potential)).kernel, 0.0, atol=1e-15,
    )
    np.testing.assert_allclose(
        (whitney_multivector.select[1] - operators.whitney1(edge_cochain)).kernel, 0.0, atol=1e-15,
    )
    np.testing.assert_allclose(
        (whitney_multivector.select[2] - face_dirac.select[2]).kernel, 0.0, atol=1e-13,
    )


def test_interior_decoupling_and_hodge_helmholtz_decomposition_of_a_velocity_on_annulus():
    mesh, vertices, inner_ring, interior = scenarios.annulus(0.32, 1.0, 7)
    operators = core.Operators(mesh, vertices)

    # An annulus has Euler characteristic V - E + F = 0.
    assert mesh.n_vertices - len(mesh.edges) + len(mesh.faces) == 0
    assert operators.potential_velocity.gatype == core.ga.gatype((core.Point, core.Scalar))
    assert operators.stream_velocity.gatype == core.ga.gatype((core.Point, core.Scalar))

    # On every interior vertex star, the exterior wedge of two hat gradients integrates to zero,
    # so a stream velocity carries no net divergence and a potential velocity no net circulation there.
    x, y = core.mv.x & vertices, core.mv.y & vertices
    arbitrary_potential = x.squared() - 0.4 * x * y + 0.7 * y
    arbitrary_stream = 0.9 * x.squared() + 0.5 * y.squared() - 0.3 * x
    stream_divergence = operators.gradient.adjugate()(operators.stream_velocity(arbitrary_stream))
    potential_circulation = operators.stream_velocity.adjugate()(
        operators.isotropic_resistance(operators.potential_velocity(arbitrary_potential)),
    )
    np.testing.assert_allclose(stream_divergence.batch()[interior].kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(potential_circulation.batch()[interior].kernel, 0.0, atol=1e-13)

    # Harmonic circulation has zero interior stream Laplacian.
    harmonic_stream, true_harmonic = core.harmonic_circulation(operators, inner_ring, interior)
    np.testing.assert_allclose(
        operators.stream_laplacian(harmonic_stream).batch()[interior].kernel, 0.0, atol=1e-13,
    )

    # Recovery of the potential, stream and harmonic parts of a velocity on the perforated disk.
    r2 = vertices.norm_squared()
    bubble = (r2 - 0.32**2) * (1.0 - r2)
    true_potential = bubble * (1.9 * x - 1.1 * y)
    true_stream = bubble * (1.4 * x + 1.7 * y)
    total_flow = (operators.potential_velocity(true_potential) + operators.stream_velocity(true_stream)
                  + 0.7 * true_harmonic)

    potential, stream, potential_flow, stream_flow, harmonic_flow = core.hodge_decomposition(
        operators, total_flow, interior,
    )
    np.testing.assert_allclose((potential - true_potential).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((stream - true_stream).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((harmonic_flow - 0.7 * true_harmonic).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        ((potential_flow + stream_flow + harmonic_flow) - total_flow).kernel, 0.0, atol=1e-14,
    )

    # Pairwise energy orthogonality of the three Hodge-Helmholtz components.
    resistance = operators.isotropic_resistance
    np.testing.assert_allclose((resistance(potential_flow) & stream_flow).sites.sum().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((resistance(potential_flow) & harmonic_flow).sites.sum().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((resistance(stream_flow) & harmonic_flow).sites.sum().kernel, 0.0, atol=1e-12)


def test_skew_adjoint_dirac_kahler_wave_operators_and_boundary_mode_conversion():
    mesh, vertices, _, _ = scenarios.annulus(0.32, 1.0, 6)
    operators = core.Operators(mesh, vertices)

    # Exact skew-adjoint power cancellation between even_gradient and even_divergence.
    x, y = core.mv.x & vertices, core.mv.y & vertices
    test_even = (0.8 * x - 0.5 * y) + (0.6 * x * y - 0.2 * y) * core.mv.xy
    test_field = operators.even_gradient((0.3 * x + 0.9 * y) + (-0.4 * x + 0.7 * y) * core.mv.xy)
    power_even = (operators.vertex_areas * operators.even_divergence(test_field).reverse().scalar_product(test_even)).sites.sum()
    power_field = (operators.areas * test_field.scalar_product(operators.even_gradient(test_even))).sites.sum()
    np.testing.assert_allclose((power_even + power_field).kernel, 0.0, atol=1e-14)

    # A purely compressional initial pulse (Grade 0 only) converts partially into Grade-2 vorticity at the
    # boundary while conserving total Dirac-Kahler wave energy.
    source = -0.58 * core.mv.x.dual() + 0.18 * core.mv.y.dual()
    envelope = (-(vertices - source).norm_squared() / (2 * 0.16**2)).exp()
    initial_even = envelope.cast(core.Even)
    initial_field = operators.even_gradient(initial_even * 0.0)

    total_energies, vortical_energies = [], []
    for even_state, face_field in core.propagate(operators, initial_even, initial_field, 0.01, 45):
        e_phi = (operators.vertex_areas * even_state.select[0].squared()).sites.sum()
        e_psi = (operators.vertex_areas * even_state.select[2].scalar_norm_squared()).sites.sum()
        e_field = (operators.areas * face_field.scalar_norm_squared()).sites.sum()
        total_energies.append((e_phi + e_psi + e_field) / 2)
        vortical_energies.append(e_psi / 2)
    totals = stack(total_energies)
    vorticals = stack(vortical_energies)

    np.testing.assert_allclose(vorticals[0].kernel, 0.0, atol=1e-16)
    assert vorticals[-1] > 1e-3
    np.testing.assert_allclose((totals / totals[0]).kernel, 1.0, rtol=1e-4)
