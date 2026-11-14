"""Projective readings in two signatures, and one physical diffusion problem under different lifts."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import NumpyContext
from numga.algebras import VGA3D, PGA2D
from examples.fluids.projective_sphere_flow import render, scenarios
from examples.util import shapes
from numga.topology import Topology


@pytest.mark.parametrize("ga", [VGA3D, PGA2D])
def test_projective_readings_and_quotient_derivative(ga):
    mv = NumpyContext(ga).multivector
    triangle = Topology(3, [[0, 1, 2]])
    corners = mv.vector([[1, 0, 1], [0, 1, 1], [-0.5, -0.2, 1]]).dual()
    scales = np.array([0.35, 2.2, 1.1])
    representatives = (corners * scales).field()
    tails, heads = triangle.incidence[1, 0] * representatives
    reciprocals = ((triangle.boundary[2, 1] * (tails & heads)) / (representatives.batch()[0] & representatives.batch()[1] & representatives.batch()[2])).batch()[:, 0]
    section = reciprocals.sum(axis=0)
    point = (corners * np.array([0.2, 0.35, 0.45])).sum(axis=0)
    displacement = corners[1] - corners[0]
    denominator = section & point
    barycentrics = (reciprocals & point) / denominator
    gradients = (reciprocals - barycentrics * section) / denominator
    edge_tails, edge_heads = triangle.corners[2, 1]
    edge_planes = reciprocals[edge_tails] ^ reciprocals[edge_heads]
    forms = (edge_planes & point) / denominator.squared()
    alternating = barycentrics[edge_tails] * gradients[edge_heads] - barycentrics[edge_heads] * gradients[edge_tails]
    np.testing.assert_allclose(barycentrics.sum(axis=0).to_array(), 1, atol=1e-12)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((gradients & point).to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((forms - alternating).kernel, 0, atol=1e-12)

    # A varying scale along a path adds radial velocity; neither scalar nor edge reading changes.
    scale, scale_rate = 2.6, -0.7
    other_point = scale * point
    other_displacement = scale * displacement + scale_rate * point
    other_denominator = section & other_point
    other_barycentrics = (reciprocals & other_point) / other_denominator
    other_forms = (edge_planes & other_point) / other_denominator.squared()
    np.testing.assert_allclose((other_barycentrics - barycentrics).to_array(), 0, atol=1e-12)
    np.testing.assert_allclose(((other_forms & other_displacement) - (forms & displacement)).to_array(), 0, atol=1e-12)

    # Compensating the corner reciprocals keeps a chosen chart, and therefore the entire basis, fixed.
    corner_field = corners.field()
    unit_tails, unit_heads = triangle.incidence[1, 0] * corner_field
    unit_reciprocals = ((triangle.boundary[2, 1] * (unit_tails & unit_heads)) / (corners[0] & corners[1] & corners[2])).batch()[:, 0]
    chart = unit_reciprocals.sum(axis=0)
    chart_reciprocals = reciprocals * (chart & representatives.batch())
    np.testing.assert_allclose((chart_reciprocals - unit_reciprocals).kernel, 0, atol=1e-12)


def test_fixed_sphere_diffusion_with_unequal_vertex_scales():
    core = scenarios.core
    topology, coordinates = shapes.sphere(scenarios.n_levels)
    vertices = scenarios.mv.vector(coordinates).dual().field()                  # Point-[V]
    grid, barycentric = scenarios.triangle_grid(scenarios.n_quadrature_divisions)
    quadrature = (grid.incidence[2, 0].mean(axis=0) * core.mv.scalar(barycentric[..., None]).field()).batch()
    corners = topology.incidence[2, 0] * vertices
    chords = (corners[:, None] * quadrature).sum(axis=0)
    samples = chords.normalized()
    measures = (corners[0] & corners[1] & corners[2]).norm() / (2 * quadrature.shape[-1] * chords.norm() ** 3)
    source = 2 * (core.mv.z & samples)
    reference, _, _ = core.diffusion(topology, vertices, samples, measures, source, core.Plane.dual())
    common_scale, _, _ = core.diffusion(topology, 3.2 * vertices, samples, measures, source, core.Plane.dual())
    np.testing.assert_allclose((common_scale - reference).to_array(), 0, atol=1e-11)

    for radius in [0.4, 2.5]:
        scales = np.ones(topology.n_vertices)
        scales[0] = radius
        representatives = vertices * core.mv.scalar(scales[..., None]).field()
        potential, _, _ = core.diffusion(topology, representatives, samples, measures, source, core.Plane.dual())
        # The height is the known physical solution, an eigenfunction of the sphere's Laplacian, at this coarse resolution.
        np.testing.assert_allclose((potential - (core.mv.z & vertices)).to_array(), 0, atol=0.08)
        assert np.max(np.abs((potential - reference).to_array())) > 1e-3
    zero_source = source * 0
    equilibrium, _, _ = core.diffusion(topology, vertices, samples, measures, zero_source, core.Plane.dual())
    np.testing.assert_allclose(equilibrium.to_array(), 0, atol=1e-12)


def test_render_draws_sphere_and_planar():
    topology, coordinates = shapes.sphere(1)
    vertices = scenarios.mv.vector(coordinates).dual().field()
    plt.close(render.draw_triangulation(topology, vertices))
    quadrature_grid, quadrature_weights = scenarios.triangle_grid(2)
    quadrature = (quadrature_grid.incidence[2, 0].mean(axis=0) * scenarios.mv.scalar(quadrature_weights[..., None]).field()).batch()
    display_grid, display = scenarios.triangle_grid(2)
    frames = scenarios.spherical_frames(topology, vertices, [np.ones(topology.n_vertices)], quadrature, display)
    next(render.animate_sphere(topology, display_grid, frames))

    points = (scenarios.planar_corners[:, None] * display).sum(axis=0)
    potential = scenarios.planar.mv.x & points.normalized()
    current = scenarios.planar.mv.x.dual() * np.ones(len(points))
    plt.close(render.draw_planar(points, display_grid, potential, current))
