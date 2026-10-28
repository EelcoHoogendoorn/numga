"""Quadrics of the conformal model, the Dupin cyclide among them, and the GPU trace through glass."""

from __future__ import annotations

import numpy as np
import pytest

from examples.quadrics.glass_cyclide import core, scenarios

mv = core.mv
ROUND_OFF = 1e-10


def test_inverted_torus_lies_on_the_cyclide():
    points = scenarios.inversion >> core.torus_points(scenarios.torus_radius, scenarios.torus_tube, 16, 8)
    inversion = scenarios.inversion
    surface = inversion >> core.torus(scenarios.torus_radius, scenarios.torus_tube)(inversion << core.Point)
    form = (points & surface(points)) / (core.horizon & points).squared()
    np.testing.assert_allclose(form.kernel, 0, atol=ROUND_OFF)


def test_tangent_sphere_gives_the_outward_direction():
    centre = core.translator(mv.x.dual() * 1.0 - mv.z.dual() * 0.5) >> core.origin     # [] Point-
    solid = core.ball(core.sphere(centre, 0.8))
    radial = (mv.x.dual() * 0.6 + mv.y.dual() * 0.48 + mv.z.dual() * 0.64) * 0.8       # [] Direction-
    point = core.translator(radial) >> centre                                          # [] Point-
    direction = core.outward(solid(point), point)                                      # [] Direction-
    # Square to the sphere, along the radius, and pointing out of it: a step along it leaves the ball.
    np.testing.assert_allclose((direction & radial).kernel, 0, atol=ROUND_OFF)
    beyond = core.translator(direction * 0.01) >> point                                # [] Point-
    assert (beyond & solid(beyond)).kernel > 0
    # Inside the ball the form is negative, outside positive, and a negated quadric swaps them.
    inside, outside = centre, core.translator(mv.x.dual() * 2) >> centre
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
