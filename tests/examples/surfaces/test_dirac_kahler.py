"""Mixed-grade Whitney reconstruction, Dirac–Kähler derivatives, Hodge decomposition, and wave dynamics."""

import matplotlib.pyplot as plt

from numga import stack
from examples.surfaces.dirac_kahler import core, render, scenarios
from examples.util import shapes


def test_mixed_grade_whitney_reconstruction_and_dirac_kahler_product_on_disk():
    topology, coordinates = shapes.hexagonal_disk(1.0, 5)
    vertices = core.mv.vector(coordinates).dual().field()                    # Point-[V]
    _, patches, _, gradients = core.simplex_geometry(topology, vertices)
    exterior_curl, whitney0, whitney1, whitney2 = core.whitney_maps(topology, patches, gradients)
    gradient, dirac, even_gradient = core.dirac_derivatives(topology, gradients)

    assert whitney0.gatype == core.ga.gatype((core.Scalar, core.Scalar))
    assert whitney1.gatype == core.ga.gatype((core.Plane, core.Scalar))
    assert whitney2.batch().gatype.derive.structural == core.ga.gatype((core.Bivector, core.Scalar))
    assert dirac.gatype == core.ga.gatype((core.Full, core.Full))
    assert even_gradient.gatype == core.ga.gatype((core.Plane, core.Even))

    x, y = core.mv.x & vertices, core.mv.y & vertices                        # Scalar+[V], Scalar+[V]
    scalar_potential = 0.6 * (x.squared() - y.squared()) + 0.4 * x - 0.25 * y  # Scalar+[V]
    vertex_1form = (0.7 * x - 0.5 * y) * core.mv.x + (0.4 * x + 0.9 * y) * core.mv.y  # Plane+[V]
    stream_bivector = (0.8 * x * y - 0.3 * x) * core.mv.xy                   # Bivector+[V]
    vertex_multivector = scalar_potential + vertex_1form + stream_bivector   # Full+[V]
    assert vertex_multivector.batch().gatype == core.Full

    scalar_gradient = gradient(scalar_potential)                             # Plane+[F]
    face_dirac = dirac(vertex_multivector)                                   # Full+[F]
    assert face_dirac.batch().gatype == core.Full

    corner_1form = topology.incidence[2, 0] * vertex_1form                   # [3] Plane+[F]
    corner_stream = topology.incidence[2, 0] * stream_bivector               # [3] Bivector+[F]
    div_1form = (gradients | corner_1form).sum(axis=0)                       # Scalar+[F]
    grad_plus_rot = scalar_gradient + (gradients | corner_stream).sum(axis=0)  # Plane+[F]
    curl_1form = (gradients ^ corner_1form).sum(axis=0)                      # Bivector+[F]

    assert ((face_dirac.select[0] - div_1form).abs() < 1e-12).all()
    assert ((face_dirac.select[1] - grad_plus_rot).scalar_norm_squared() < 1e-24).all()
    assert ((face_dirac.select[1] - even_gradient(scalar_potential + stream_bivector)).scalar_norm_squared() < 1e-24).all()
    assert ((face_dirac.select[2] - curl_1form).scalar_norm_squared() < 1e-24).all()

    edge_midpoints = topology.incidence[1, 0].mean(axis=0) * vertices        # Point-[E]
    midpoint_x, midpoint_y = core.mv.x & edge_midpoints, core.mv.y & edge_midpoints  # Scalar+[E], Scalar+[E]
    midpoint_1form = (0.7 * midpoint_x - 0.5 * midpoint_y) * core.mv.x + (0.4 * midpoint_x + 0.9 * midpoint_y) * core.mv.y  # Plane+[E]
    edge_cochain = midpoint_1form & (topology.d[0] * vertices)               # Scalar+[E]
    face_cochain = exterior_curl(edge_cochain)                               # Scalar+[F]
    whitney_multivector = (
        whitney0(scalar_potential)
        + whitney1(edge_cochain)
        + whitney2(face_cochain)
    )                                                                        # Full+[F]
    assert whitney_multivector.batch().gatype == core.Full
    assert ((whitney_multivector.select[0] - whitney0(scalar_potential)).abs() < 1e-12).all()
    assert ((whitney_multivector.select[1] - whitney1(edge_cochain)).scalar_norm_squared() < 1e-24).all()
    assert ((whitney_multivector.select[2] - face_dirac.select[2]).scalar_norm_squared() < 1e-24).all()

    figure = render.grades_figure(
        topology, vertices, vertex_multivector, scalar_gradient, face_dirac, whitney_multivector,
    )
    assert len(figure.axes) == 4
    plt.close(figure)


def test_interior_decoupling_and_hodge_helmholtz_decomposition_of_a_velocity_on_annulus():
    topology, vertices, inner_ring, interior = scenarios.perforated_annulus(0.32, 1.0, 6)
    V, E, F = topology.n_elements
    assert V - E + F == 0

    _, patches, areas, gradients = core.simplex_geometry(topology, vertices)
    gradient, _, even_gradient = core.dirac_derivatives(topology, gradients)
    (
        _, isotropic_resistance, potential_velocity,
        stream_velocity, even_velocity, stream_laplacian, even_laplacian,
    ) = core.medium_operators(topology, patches, areas, gradient, even_gradient)

    assert potential_velocity.gatype == core.ga.gatype((core.Point, core.Scalar))
    assert stream_velocity.gatype == core.ga.gatype((core.Point, core.Scalar))

    # On every interior vertex star, the exterior wedge of two simplex gradients sums to zero,
    # so a stream velocity carries no net divergence and a potential velocity no net circulation there.
    x, y = core.mv.x & vertices, core.mv.y & vertices                        # Scalar+[V], Scalar+[V]
    arbitrary_potential = x.squared() - 0.4 * x * y + 0.7 * y                # Scalar+[V]
    arbitrary_stream = 0.9 * x.squared() + 0.5 * y.squared() - 0.3 * x       # Scalar+[V]
    stream_divergence = gradient.adjugate()(stream_velocity(arbitrary_stream))  # Antiscalar-[V]
    potential_circulation = stream_velocity.adjugate()(
        isotropic_resistance(potential_velocity(arbitrary_potential)),
    )                                                                        # Antiscalar-[V]
    assert ((topology.reading(interior, 0) * stream_divergence).scalar_norm_squared() < 1e-24).all()
    assert ((topology.reading(interior, 0) * potential_circulation).scalar_norm_squared() < 1e-24).all()

    harmonic_stream, true_harmonic = core.harmonic_circulation(
        topology, stream_velocity, stream_laplacian, inner_ring, interior,
    )
    assert ((topology.reading(interior, 0) * stream_laplacian(harmonic_stream)).scalar_norm_squared() < 1e-24).all()

    squared_radii = vertices.norm_squared()                                  # Scalar+[V]
    bubble = (squared_radii - 0.32**2) * (1.0 - squared_radii)               # Scalar+[V]
    true_potential = bubble * (1.9 * x - 1.1 * y)                            # Scalar+[V]
    true_stream = bubble * (1.4 * x + 1.7 * y)                               # Scalar+[V]
    total_flow = (
        potential_velocity(true_potential)
        + stream_velocity(true_stream)
        + 0.7 * true_harmonic
    )                                                                        # Point-[F]

    potential, stream, potential_flow, stream_flow, harmonic_flow = core.hodge_decomposition(
        topology, even_gradient, even_velocity, even_laplacian, total_flow, interior,
    )
    assert ((potential - true_potential).abs() < 1e-11).all()
    assert ((stream - true_stream).abs() < 1e-11).all()
    assert ((harmonic_flow - 0.7 * true_harmonic).scalar_norm_squared() < 1e-22).all()
    assert (((potential_flow + stream_flow + harmonic_flow) - total_flow).scalar_norm_squared() < 1e-24).all()

    assert (isotropic_resistance(potential_flow) & stream_flow).sites.sum().abs() < 1e-11
    assert (isotropic_resistance(potential_flow) & harmonic_flow).sites.sum().abs() < 1e-11
    assert (isotropic_resistance(stream_flow) & harmonic_flow).sites.sum().abs() < 1e-11

    figure = render.hodge_figure(
        topology, vertices, total_flow, potential, potential_flow, stream_flow, harmonic_flow,
    )
    assert len(figure.axes) == 4
    plt.close(figure)


def test_skew_adjoint_dirac_kahler_wave_operators_and_boundary_mode_conversion():
    topology, vertices, _, _ = scenarios.perforated_annulus(0.32, 1.0, 5)
    _, patches, areas, gradients = core.simplex_geometry(topology, vertices)
    gradient, _, even_gradient = core.dirac_derivatives(topology, gradients)
    isotropic_medium, *_ = core.medium_operators(topology, patches, areas, gradient, even_gradient)
    dual_areas, even_divergence = core.wave_divergence(topology, areas, even_gradient, isotropic_medium)

    assert even_divergence.gatype == core.ga.gatype((core.Even, core.Plane))

    x, y = core.mv.x & vertices, core.mv.y & vertices                        # Scalar+[V], Scalar+[V]
    test_even = (0.8 * x - 0.5 * y) + (0.6 * x * y - 0.2 * y) * core.mv.xy   # Even+[V]
    test_field = even_gradient((0.3 * x + 0.9 * y) + (-0.4 * x + 0.7 * y) * core.mv.xy)  # Plane+[F]
    power_even = (dual_areas * even_divergence(test_field).reverse().scalar_product(test_even)).sites.sum()  # [] Scalar+
    power_field = (areas * test_field.scalar_product(even_gradient(test_even))).sites.sum()  # [] Scalar+
    assert (power_even + power_field).abs() < 1e-12

    source = -0.58 * core.mv.x.dual() + 0.18 * core.mv.y.dual()              # [] Point-
    envelope = (-(vertices - source).norm_squared() / (2 * 0.16**2)).exp()   # Scalar+[V]
    initial_even = envelope.cast(core.Even)                                  # Even+[V]
    initial_field = even_gradient(initial_even * 0.0)                        # Plane+[F]

    states = list(core.wave_states(even_gradient, even_divergence, initial_even, initial_field, 0.01, 20, 20))
    energies = stack([
        ((dual_areas * even_state.scalar_norm_squared()).sites.sum()
         + (areas * face_field.scalar_norm_squared()).sites.sum()) / 2
        for even_state, face_field in states
    ])                                                                       # [2] Scalar+
    vorticals = stack([
        (dual_areas * even_state.select[2].scalar_norm_squared()).sites.sum() / 2
        for even_state, _ in states
    ])                                                                       # [2] Scalar+

    assert vorticals[0].abs() < 1e-12
    assert vorticals[-1] > 1e-4
    assert ((energies / energies[0] - 1.0).abs() < 1e-4).all()

    frame = next(render.waves_animation(topology, vertices, states[:1]))
    assert frame.shape == (render.n_panel_pixels, 3 * render.n_panel_pixels + 2 * render.panel_gap, 3)
    assert frame.dtype == "uint8"
