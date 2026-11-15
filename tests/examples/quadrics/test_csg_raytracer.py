"""Coverage tests for quadric-solid placement, hyperboloid rulings, camera orbits, and GPU tracing."""

from __future__ import annotations

import numpy as np
import pytest

from examples.quadrics.csg_raytracer import core, scenarios

mv = core.mv
TOLERANCE = 1e-10


def test_hyperboloid_rulings_lie_on_surface():
    n_rulings = 8
    n_offsets = 5
    waist, rising, falling = core.ruling_lines(scenarios.waist_axes, n_rulings)  # [n_rulings] Point-, Point-, Point-
    offsets = np.linspace(-2, 2, n_offsets)[:, None]
    surface = core.hyperboloid(scenarios.waist_axes)                             # [] Plane+ <- Point-
    ruling_points = waist + rising * offsets                                     # [n_offsets, n_rulings] Point-
    falling_points = waist + falling * offsets                                   # [n_offsets, n_rulings] Point-
    assert ((surface(ruling_points) & ruling_points).abs() < TOLERANCE).all()
    assert ((surface(falling_points) & falling_points).abs() < TOLERANCE).all()


def test_placed_quadrics_and_half_spaces_follow_motor():
    shift = ((mv.w ^ mv.direction([1, -2, 0.5]).dual()) / 2).exp()               # [] Motor+
    motor = shift * (mv.xz * -0.3).exp()                                         # [] Motor+
    ball = core.ellipsoid(np.array([0.5, 0.7, 0.9]))                             # [] Plane+ <- Point-
    floor = core.half_space(mv.z)                                                # [] Plane+ <- Point-
    points = mv.point([[0.5, 0, 0], [0, 0.7, 0], [0, 0, -2], [0.1, 0.2, 0.3]])   # [4] Point-
    moved = motor >> points                                                      # [4] Point-
    for quadric in (ball, floor):
        placed = motor >> quadric(motor << core.Point)                           # [] Plane+ <- Point-
        residual = (placed(moved) & moved) - (quadric(points) & points)          # [4] Scalar+
        assert (residual.abs() < TOLERANCE).all()
    heights = mv.scalar([[0], [0], [-2], [0.3]])                                 # [4] Scalar+
    assert (((floor(points) & points) - heights).abs() < TOLERANCE).all()


def test_orbit_cameras_aim_at_target():
    n_cameras = 4
    cameras = core.camera_orbit(scenarios.target, scenarios.distance, scenarios.elevation,
                                np.linspace(0, 2 * np.pi, n_cameras, endpoint=False))  # [n_cameras] Motor+
    eyes = cameras >> core.origin                                                # [n_cameras] Point-
    aim = cameras >> (-mv.z.dual())                                              # [n_cameras] Point-
    assert ((eyes + aim * scenarios.distance - scenarios.target).antinorm() < TOLERANCE).all()


def test_one_frame_traces():
    pytest.importorskip("wgpu")
    from examples.quadrics.csg_raytracer import render

    quadrics, solids = scenarios.scene_solids()
    frame, = render.frames(render.shader(), quadrics, solids, scenarios.orbit_cameras(1),
                           scenarios.lamp, scenarios.half_view, (32, 48))
    assert frame.shape == (32, 48, 3)
