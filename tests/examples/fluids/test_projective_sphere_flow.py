"""Projective readings in two signatures, and one physical diffusion problem under different lifts."""

import numpy as np
import pytest

from numga import NumpyContext
from numga.algebras import VGA3D, PGA2D
from examples import instantiate
from examples.fluids.projective_sphere_flow import scenarios


@pytest.mark.parametrize("ga", [VGA3D, PGA2D])
def test_projective_readings_and_quotient_derivative(ga):
    mv = NumpyContext(ga).multivector
    Mesh = instantiate("examples.util.mesh", ga).Mesh
    triangle = Mesh(3, [[0, 1, 2]])
    corners = mv.antivector([[1, 0, 1], [0, 1, 1], [-0.5, -0.2, 1]])
    scales = np.array([0.35, 2.2, 1.1])
    lifted = (corners * scales).field()
    tails, heads = triangle.incidence[1, 0] * lifted
    reciprocals = ((triangle.boundary[2, 1] * (tails & heads)) / (lifted.batch()[0] & lifted.batch()[1] & lifted.batch()[2])).batch()[:, 0]
    section = reciprocals.sum(axis=0)
    point = (corners * np.array([0.2, 0.35, 0.45])).sum(axis=0)
    displacement = corners[1] - corners[0]
    denominator = section & point
    hats = (reciprocals & point) / denominator
    gradients = (reciprocals - hats * section) / denominator
    edge_tails, edge_heads = triangle.corners[2, 1]
    edge_planes = reciprocals[edge_tails] ^ reciprocals[edge_heads]
    forms = (edge_planes & point) / denominator.squared()
    alternating = hats[edge_tails] * gradients[edge_heads] - hats[edge_heads] * gradients[edge_tails]
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
    corner_field = corners.field()
    unweighted_tails, unweighted_heads = triangle.incidence[1, 0] * corner_field
    unweighted = ((triangle.boundary[2, 1] * (unweighted_tails & unweighted_heads)) / (corners[0] & corners[1] & corners[2])).batch()[:, 0]
    chart = unweighted.sum(axis=0)
    compensated = reciprocals * (chart & lifted.batch())
    np.testing.assert_allclose((compensated - unweighted).kernel, 0, atol=1e-12)


def test_fixed_sphere_diffusion_with_unequal_vertex_scales():
    core = scenarios.core
    mesh, vertices = core.Mesh.icosphere(scenarios.n_levels)
    grid, barycentric = scenarios.triangle_grid(scenarios.n_quadrature_divisions)
    quadrature = (grid.incidence[2, 0].mean(axis=0) * core.as_scalar(barycentric).field()).batch()
    corners = mesh.incidence[2, 0] * vertices
    chords = (corners[:, None] * quadrature).sum(axis=0)
    samples = chords.normalized()
    measures = (corners[0] & corners[1] & corners[2]).norm() / (2 * quadrature.shape[-1] * chords.norm() ** 3)
    source = 2 * (core.mv.z & samples)
    reference, _, _ = core.diffusion(mesh, vertices, samples, measures, source, core.Plane.dual())
    common_scale, _, _ = core.diffusion(mesh, 3.2 * vertices, samples, measures, source, core.Plane.dual())
    np.testing.assert_allclose((common_scale - reference).to_array(), 0, atol=1e-11)

    for radius in [0.4, 2.5]:
        scales = np.ones(mesh.n_vertices)
        scales[0] = radius
        lifted = vertices * core.as_scalar(scales).field()
        potential, _, _ = core.diffusion(mesh, lifted, samples, measures, source, core.Plane.dual())
        # The height is the known physical solution, an eigenfunction of the sphere's Laplacian, at this coarse resolution.
        np.testing.assert_allclose((potential - (core.mv.z & vertices)).to_array(), 0, atol=0.08)
        assert np.max(np.abs((potential - reference).to_array())) > 1e-3
    zero_source = source * 0
    equilibrium, _, _ = core.diffusion(mesh, vertices, samples, measures, zero_source, core.Plane.dual())
    np.testing.assert_allclose(equilibrium.to_array(), 0, atol=1e-12)
