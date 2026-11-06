"""Unit tests for the scenegraph, robot kinematics, and compound optics example, and one frame of the robot's sweep
drawn."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.scenegraph import core, render, scenarios

mv = core.mv


def euclidean(points: core.Point) -> np.ndarray:
    """The Euclidean coordinates of points are their pairings with the coordinate planes divided by their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def test_anisotropic_scale_extensor():
    """The anisotropic scaling extensor scales each coordinate and keeps the weight."""
    scaled = core.anisotropic_scale(2.0, 0.5, 3.0)(mv.point([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(euclidean(scaled), [6.0, 2.0, 15.0], atol=1e-12)
    np.testing.assert_allclose((mv.w & scaled).to_array(), 1.0, atol=1e-12)


def test_robot_arm_kinematics():
    """Forward kinematics yields five rigidly placed bodies."""
    unit_box = core.canonical_unit_box()
    bodies_to_world, _ = core.robot_arm((0.2, -0.3, 0.5, -0.2))
    assert bodies_to_world.shape == (5,)
    verts = bodies_to_world[:, None](unit_box[None, :])
    np.testing.assert_allclose((mv.w & verts).to_array(), 1.0, atol=1e-10)


def test_compound_optics_lands_on_the_sensor():
    """The multi-lens camera sends world points onto the sensor plane z = -1.25."""
    pose = core.look_at(mv.point([0.0, -4.0, 1.5]), mv.point([0.0, 0.0, 1.0]))
    front_lens, rear_lens, _ = core.lens_train(1.0, 0.8, -0.3)
    pupil = (mv.xw * 0.02).exp() >> core.origin
    camera = core.lens_camera(pose, front_lens, rear_lens, pupil, mv.z + 1.25 * mv.w)
    sensor_hit = camera(mv.point([0.1, 0.2, 1.0]))
    np.testing.assert_allclose(euclidean(sensor_hit)[2], -1.25, atol=1e-10)


def test_one_frame_of_the_sweep_draws():
    next(render.animate_scenegraph(scenarios.robot_sweep(1)))
