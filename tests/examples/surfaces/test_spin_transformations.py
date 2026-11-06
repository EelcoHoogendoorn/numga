"""Spin transformations: the Dirac operator is the geometric derivative times the face normals, and the three scenes run
on coarse spheres and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga.sparse import SparseExtensor, spdiag

from examples.util import shapes
from examples.surfaces.spin_transformations import core, render, scenarios


def test_the_dirac_operator_is_the_geometric_derivative_times_the_face_normals():
    """Crane's operator, each corner taken by the edge it faces over twice the area, is the geometric
    derivative with each face's unit normal on the right, where it anticommutes with the in-plane
    gradients; and the geometric derivative of position is twice the volume element on every face."""
    topology, coordinates = shapes.sphere(3)
    vertices = core.context.multivector.vector(coordinates).dual().field()      # Point-[vertices]
    edges = topology.d[0] * vertices                                          # Point-[edges]
    sides = topology.boundary[2, 1] * edges                                   # [3] Point-[faces]
    corners = (spdiag(sides) * topology.incidence[2, 0]).sum(axis=0)
    areas = sides[1].commutator(sides[0]).norm() / 2                          # Scalar+[faces]
    dirac = spdiag(1 / areas) * corners * 0.5
    derivative = core.geometric_derivative(topology, vertices)
    normals = (sides[0] & sides[1]) / (2 * areas)                             # Vector-[faces]
    np.testing.assert_allclose((spdiag(normals) * derivative + dirac).cells.kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((derivative * vertices).kernel, np.broadcast_to([0, 0, 0, 2], (topology.n_elements[2], 4)), atol=1e-12)


def test_the_scenes_run_and_draw():
    topology, vertices, curvature_change, transformed_vertices = scenarios.dipole(n_subdivisions=1)
    plt.close(render.draw_dipole(topology, vertices, curvature_change, transformed_vertices))
    sphere, sphere_vertices, gallery, _ = scenarios.dirac(n_subdivisions=1, n_states=1)
    plt.close(render.draw_dirac(sphere, gallery, sphere_vertices))
    # The rounding flow, one step on the coarse sphere rather than on Spot.
    frames = list(core.conformal_smooth(topology, vertices, 1, scenarios.RATE))
    assert len(list(render.animate_rounding(topology, frames))) == 2
