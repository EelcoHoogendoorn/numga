"""Spin transformations: the Dirac operator equals the geometric derivative times the face normals, and
the scenes run and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt

from numga.sparse import spdiag

from examples.surfaces.spin_transformations import core, render, scenarios
from examples.util import shapes


def test_the_dirac_operator_is_the_geometric_derivative_times_the_face_normals():
    """Crane's operator, coupling each corner by the opposite edge over twice the face area, equals the
    geometric derivative multiplied on the left by the negative unit face normal, and the geometric
    derivative of position equals twice the intrinsic pseudoscalar on every face."""
    topology, coordinates = shapes.sphere(1)
    vertices = core.mv.vector(coordinates).dual().field()                      # Point-[V]
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)               # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                        # Patch-[F]
    areas = patches.norm()                                                     # Scalar+[F]
    dirac = (spdiag(sides / (2 * areas)) * topology.incidence[2, 0]).sum(axis=0)  # Point-[F <- V]
    derivative = core.geometric_derivative(topology, vertices)                 # Plane+[F <- V]
    normals = patches / areas                                                  # Patch-[F]
    difference = spdiag(normals) * derivative + dirac                          # Even-[F <- V]
    assert ((~difference * difference).diagonal().scalar_norm_squared() < 1e-24).all()
    assert (((derivative * vertices).dual() - 2).scalar_norm_squared() < 1e-24).all()
    # Zero prescribed curvature change leaves a centred sphere invariant.
    unchanged = core.deformed_surface(topology, vertices, areas * 0) - (vertices - vertices.sites.mean())
    assert (unchanged.scalar_norm_squared() < 1e-22).all()


def test_the_scenes_run_and_draw():
    topology, vertices, curvature_change, transformed_vertices = scenarios.dipole_scene(n_splits=1)
    plt.close(render.dipole_figure(topology, vertices, curvature_change, transformed_vertices))
    plt.close(render.surface_figure(topology, vertices, curvature_change))
    plt.close(render.textured_figure(topology, transformed_vertices, vertices))
    sphere, sphere_vertices, gallery, _ = scenarios.dirac_scene(n_splits=1, n_modes=1)
    plt.close(render.dirac_figure(sphere, gallery, sphere_vertices))
    # Run one step of the rounding flow on the coarse sphere.
    frames = core.curvature_flow_states(topology, vertices, 1, scenarios.flow_rate)
    assert len(list(render.rounding_animation(topology, frames))) == 2
