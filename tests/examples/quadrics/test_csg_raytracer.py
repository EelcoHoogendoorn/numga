"""Solids bounded by quadrics, their placement, and the GPU trace of them."""

from __future__ import annotations

import numpy as np
import pytest

from examples.quadrics.csg_raytracer import core, scenarios

mv = core.mv
ROUND_OFF = 1e-12


def test_hyperboloid_rulings_lie_in_its_surface():
    waist, rising, falling = core.ruling_lines(scenarios.waist_axes, 8)
    offsets = np.linspace(-2, 2, 5)[:, None]
    surface = core.hyperboloid(scenarios.waist_axes)
    for direction in (rising, falling):
        on_ruling = waist + direction * offsets
        np.testing.assert_allclose((surface(on_ruling) & on_ruling).to_array(), 0, atol=ROUND_OFF)


def test_placed_quadrics_and_half_spaces_follow_their_motor():
    shift = ((mv.w ^ mv.direction([1.0, -2.0, 0.5]).dual()) / 2).exp()
    motor = shift * (mv.xz * -0.3).exp()
    ball = core.ellipsoid(np.array([0.5, 0.7, 0.9]))
    floor = core.half_space(mv.z)
    points = mv.point([[0.5, 0, 0], [0, 0.7, 0], [0, 0, -2], [0.1, 0.2, 0.3]])
    moved = motor >> points
    for quadric in (ball, floor):
        placed = motor >> quadric(motor << core.Point)
        np.testing.assert_allclose((placed(moved) & moved).to_array(), (quadric(points) & points).to_array(),
                                   atol=ROUND_OFF)
    # The half-space is negative below its plane, by the height times the weight.
    np.testing.assert_allclose((floor(points) & points).to_array(), [0, 0, -2, 0.3], atol=ROUND_OFF)


def test_orbit_cameras_look_at_the_target():
    cameras = scenarios.orbit(6)
    eyes = cameras >> core.origin
    looks = cameras >> -mv.z.dual()
    np.testing.assert_allclose((eyes + looks * scenarios.distance - scenarios.target).kernel, 0, atol=ROUND_OFF)


def test_trace_shows_the_solids():
    pytest.importorskip("wgpu")
    from examples.quadrics.csg_raytracer import render

    quadrics, solids = scenarios.solids()
    frame, = render.frames(render.shader(), quadrics, solids, scenarios.orbit(1), scenarios.lamp,
                           scenarios.half_view, (90, 144))
    # The bowl's teal shows right of centre, and the floor's grey below everything.
    right = frame[30:60, 90:130].reshape(-1, 3).astype(float)
    assert np.any((right[:, 2] > right[:, 0] + 30))
    assert np.all(np.abs(np.diff(frame[-1].astype(float), axis=-1)) < 12)
