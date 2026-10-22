"""Quadrics of the conformal model, the Dupin cyclide among them, and the GPU trace through glass."""

from __future__ import annotations

import numpy as np
import pytest

from examples.quadrics.glass_cyclide import core, scenarios

mv = core.mv
ROUND_OFF = 1e-10


def test_inverted_torus_lies_on_the_cyclide():
    inverted = scenarios.inversion >> core.torus_points(scenarios.torus_radius, scenarios.torus_tube, 16, 8)
    points = inverted.dual()
    surface = core.cyclide(scenarios.torus_radius, scenarios.torus_tube, scenarios.inversion)
    form = (points & surface(points)) / (core.infinity | inverted).squared()
    np.testing.assert_allclose(form.kernel, 0, atol=ROUND_OFF)


def test_tangent_sphere_gives_the_outward_direction():
    centre = mv.x * 1.0 - mv.z * 0.5
    solid = core.ball(core.sphere(centre, 0.8))
    on_surface = centre + (mv.x * 0.6 + mv.y * 0.48 + mv.z * 0.64) * 0.8       # [] Euclidean
    point = core.up(on_surface)
    direction = core.outward(solid(point), on_surface)
    radial = on_surface - centre
    # Square to the sphere, pointing out of it.
    np.testing.assert_allclose((direction ^ radial).kernel, 0, atol=ROUND_OFF)
    assert (direction | radial).kernel > 0
    # Inside the ball the form is negative, outside positive, and a negated quadric swaps them.
    inside, outside = core.up(centre), core.up(centre + mv.x * 2)
    assert (inside & solid(inside)).kernel < 0 < (outside & solid(outside)).kernel
    assert (inside & (-solid)(inside)).kernel > 0


def test_trace_shows_the_ball_through_the_ring():
    pytest.importorskip("wgpu")
    from examples.quadrics.glass_cyclide import render

    quadrics, solids = scenarios.solids()
    turns, eyes = scenarios.cameras(1)
    frame, = render.frames(render.shader(), quadrics, solids, turns, eyes, scenarios.lamp, scenarios.half_view, (90, 144))
    centre = frame[32:44, 74:88].reshape(-1, 3).astype(float)
    # The red ball, seen through the ring's hole.
    assert np.median(centre[:, 0] - centre[:, 2]) > 80
