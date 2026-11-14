"""Tests polar dual planes, spherical circumcentres, and single-frame rendering for weighted duals."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.weighted_dual import core, numerical, render, scenarios


def test_face_planes_pair_to_one_with_their_vertices():
    states = list(islice(scenarios.breath(), 2))
    assert all(np.array_equal(topology.elements[2], states[0][0].elements[2]) for topology, _, _ in states)
    for topology, points, duals in states:
        assert (((duals.dual_inverse() & (topology.incidence[2, 0] * points)) - 1).abs() < 1e-10).all()


def test_unit_sphere_poles_project_to_triangle_circumcentres():
    points = scenarios.directions                                              # Point-[V]
    topology = numerical.convex_hull(points)
    duals = core.face_poles(topology, points)                                  # Point-[F]
    corners = topology.incidence[2, 0] * points                                # [3] Point-[F]
    centres = duals.normalized()                                               # Point-[F]
    distances = (corners - centres).norm()                                     # [3] Scalar+[F]
    assert ((distances - distances[:1]).abs() < 1e-10).all()
    assert (core.pole_sides(topology, points, duals) > 0).all(axis=0).sum() > 0
    assert (core.dual_lengths(topology, points) > 0).all()


def test_figure_and_single_animation_frames_render():
    states = list(islice(scenarios.breath(), 1))
    plt.close(render.complex_figure(*states[0]))
    assert len(list(render.complex_frames(states))) == 1
    assert len(list(render.complex_frames(islice(scenarios.rescue(), 1)))) == 1
