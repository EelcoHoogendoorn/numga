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
        mesh = core.icosphere(levels)
        centroids = core.as_ga_sparse(mesh.faces, core.as_scalar(np.ones_like(mesh.faces) / 3)) * mesh.vertices
        deformed = core.spin_transform_deform(mesh, (centroids | core.context.multivector.y) * 0.5)
        errors.append(np.abs((deformed.corner_cosines() - mesh.corner_cosines()).to_array()).max())
    assert errors[1] < 0.6 * errors[0]


def test_scenes_and_figures():
    figure = render.draw_dipole(*scenarios.dipole())
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    mesh, gallery = scenarios.dirac()
    figure = render.draw_dirac(gallery, mesh)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    frames = render.animate_rounding(scenarios.rounding())
    assert len(frames) == scenarios.STEPS + 1


def test_the_dirac_operator_is_the_geometric_derivative_divided_by_the_face_planes():
    """Crane's operator, each corner taken by the edge it faces over minus twice the area, is the
    geometric derivative divided by each face's plane; and the geometric derivative of position is 2
    on every face, the surface's dimension."""
    mesh = core.icosphere(3)
    boundary = core.as_ga_sparse(mesh.edges, core.as_scalar(np.ones_like(mesh.edges) * [-1, 1]))
    edges = boundary * mesh.vertices                                          # Vector[E]
    orientation = core.as_scalar(mesh.face_edge_orientation.T).field()        # [3] Scalar[F]
    facing = SparseExtensor.selection(core.context, mesh.face_edges.T, len(mesh.edges))   # [3] [F, E] Scalar
    corners = mesh.at_corners(facing * edges * orientation)
    dirac = spdiag(1 / mesh.triangle_areas) * corners * -0.5
    derivative = core.geometric_derivative(mesh)
    np.testing.assert_allclose((spdiag(mesh.face_planes.inverse()) * derivative - dirac).cells.kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((derivative * mesh.vertices - 2).kernel, 0.0, atol=1e-12)
