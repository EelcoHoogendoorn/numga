"""Projective readings in two signatures, and one physical diffusion problem under different lifts."""

import numpy as np
import pytest

from numga import NumpyContext, stack
from numga.algebras import VGA3D, PGA2D
from examples.fluids.projective_sphere_flow import scenarios


@pytest.mark.parametrize("ga", [VGA3D, PGA2D])
def test_projective_readings_and_quotient_derivative(ga):
    mv = NumpyContext(ga).multivector
    corners = mv.antivector([[1, 0, 1], [0, 1, 1], [-0.5, -0.2, 1]])
    scales = np.array([0.35, 2.2, 1.1])
    lifted = corners * scales
    first, second, third = lifted
    reciprocals = stack((second & third, third & first, first & second)) / (first & second & third)
    section = reciprocals.sum(axis=0)
    point = (corners * np.array([0.2, 0.35, 0.45])).sum(axis=0)
    displacement = corners[1] - corners[0]
    denominator = section & point
    hats = (reciprocals & point) / denominator
    gradients = (reciprocals - hats * section) / denominator
    edge_planes = reciprocals[[1, 2, 0]] ^ reciprocals[[2, 0, 1]]
    forms = (edge_planes & point) / denominator.squared()
    alternating = hats[[1, 2, 0]] * gradients[[2, 0, 1]] - hats[[2, 0, 1]] * gradients[[1, 2, 0]]
    np.testing.assert_allclose(hats.sum(axis=0).to_array(), 1, atol=1e-12)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((gradients & point).to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((forms - alternating).kernel, 0, atol=1e-12)

    # A varying scale along a path adds radial velocity; neither scalar nor edge reading changes.
    scale, scale_rate = 2.6, -0.7
    other_point = scale * point
    other_displacement = scale * displacement + scale_rate * point
    other_denominator = section & other_point
    other_hats = (reciprocals & other_point) / other_denominator
    other_forms = (edge_planes & other_point) / other_denominator.squared()
    np.testing.assert_allclose((other_hats - hats).to_array(), 0, atol=1e-12)
    np.testing.assert_allclose(((other_forms & other_displacement) - (forms & displacement)).to_array(), 0, atol=1e-12)

    # Compensating the corner reciprocals keeps a chosen chart, and therefore the entire basis, fixed.
    first, second, third = corners
    unweighted = stack((second & third, third & first, first & second)) / (first & second & third)
    chart = unweighted.sum(axis=0)
    compensated = reciprocals * (chart & lifted)
    np.testing.assert_allclose((compensated - unweighted).kernel, 0, atol=1e-12)


def test_fixed_sphere_diffusion_with_unequal_vertex_scales():
    core = scenarios.core
    mesh, vertices = core.Mesh.icosphere(scenarios.n_levels)
    barycentric, triangles = scenarios.triangle_grid(scenarios.n_quadrature_divisions)
    quadrature = barycentric[triangles].mean(axis=1)
    corners = mesh.incidence[2, 0] * vertices
    first, second, third = corners
    chords = (corners[None, :] * quadrature).sum(axis=1)
    samples = chords.normalized()
    measures = (first & second & third).norm() / (2 * len(quadrature) * chords.norm() ** 3)
    source = 2 * (core.mv.z & samples)
    reference, _ = core.diffusion(mesh, corners, samples, measures, source, core.Plane.dual())
    common_scale, _ = core.diffusion(mesh, 3.2 * corners, samples, measures, source, core.Plane.dual())
    np.testing.assert_allclose((common_scale - reference).to_array(), 0, atol=1e-11)

    for radius in [0.4, 2.5]:
        scales = np.ones(mesh.n_vertices)
        scales[0] = radius
        lifted = vertices * core.as_scalar(scales).field()
        potential, numerator = core.diffusion(mesh, mesh.incidence[2, 0] * lifted, samples,
                                              measures, source, core.Plane.dual())
        # The height is the known physical solution, an eigenfunction of the sphere's Laplacian, at this coarse resolution.
        np.testing.assert_allclose((potential - (core.mv.z & vertices)).to_array(), 0, atol=0.08)
        assert np.max(np.abs((potential - reference).to_array())) > 1e-3
    zero_source = source * 0
    equilibrium, _ = core.diffusion(mesh, corners, samples, measures, zero_source, core.Plane.dual())
    np.testing.assert_allclose(equilibrium.to_array(), 0, atol=1e-12)
