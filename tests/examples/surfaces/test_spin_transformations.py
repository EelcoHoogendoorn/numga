"""Spin transformations: both scenes pass their checks, the deformation converges to a conformal
one under refinement, and the figures draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga.sparse import SparseExtensor, spdiag

from examples.surfaces.spin_transformations import core, render, scenarios


def test_conformal_error_falls_with_refinement():
    """The discretization is first order: each split of the faces roughly halves the largest change
    in a corner's cosine."""
    errors = []
    for levels in (2, 3):
        mesh, vertices = core.Mesh.icosphere(levels)
        centroids = mesh.incidence[2, 0].mean(axis=0) * vertices
        transformed_vertices = core.spin_transform_deform(mesh, vertices, (centroids & core.context.multivector.y) * 0.5)
        errors.append(np.abs((core.corner_cosines(mesh, transformed_vertices) - core.corner_cosines(mesh, vertices)).to_array()).max())
    assert errors[1] < 0.6 * errors[0]


def test_a_dipole_change_keeps_angles_and_raises_curvature_where_asked():
    mesh, vertices, curvature_change, transformed_vertices = scenarios.dipole()
    np.testing.assert_allclose((core.corner_cosines(mesh, transformed_vertices) - core.corner_cosines(mesh, vertices)).to_array(), 0.0, atol=0.05)
    curvature_rise = core.mean_curvature(mesh, core.recentred(transformed_vertices)) - core.mean_curvature(mesh, vertices)
    assert np.corrcoef(curvature_rise.to_array().ravel(), curvature_change.to_array().ravel())[0, 1] > 0.5
    plt.close(render.draw_dipole(mesh, vertices, curvature_change, transformed_vertices))


def test_the_dirac_spheres_have_fields_of_zero_energy():
    mesh, vertices, gallery, energies = scenarios.dirac()
    assert np.all(np.abs(energies) < 0.05)
    plt.close(render.draw_dirac(mesh, gallery, vertices))


def test_the_rounding_animation_draws():
    mesh, vertices = core.spot()
    frames = core.conformal_smooth(mesh, vertices, 1, scenarios.RATE)
    assert len(list(render.animate_rounding(mesh, frames))) == 2


def test_the_dirac_operator_is_the_geometric_derivative_times_the_face_normals():
    """Crane's operator, each corner taken by the edge it faces over twice the area, is the geometric
    derivative with each face's unit normal on the right, where it anticommutes with the in-plane
    gradients; and the geometric derivative of position is twice the volume element on every face."""
    mesh, vertices = core.Mesh.icosphere(3)
    edges = mesh.d[0] * vertices                                              # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                       # [3] Point-[faces]
    corners = (spdiag(sides) * mesh.incidence[2, 0]).sum(axis=0)
    areas = sides[1].commutator(sides[0]).norm() / 2                          # Scalar+[faces]
    dirac = spdiag(1 / areas) * corners * 0.5
    derivative = core.geometric_derivative(mesh, vertices)
    normals = (sides[0] & sides[1]) / (2 * areas)                             # Vector-[faces]
    np.testing.assert_allclose((spdiag(normals) * derivative + dirac).cells.kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((derivative * vertices).kernel, np.broadcast_to([0, 0, 0, 2], (len(mesh.faces), 4)), atol=1e-12)
