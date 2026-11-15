"""Coverage and mathematical checks for the articulated scenegraph and compound two-lens camera example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.geometry.scenegraph import core, render, scenarios

mv = core.mv


def euclidean(points: core.Point) -> np.ndarray:
    """Euclidean `(x, y, z)` coordinates of projective `points` from pairings with the coordinate planes."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def test_anisotropic_scale_extensor():
    """Anisotropic scaling stretches each spatial coordinate while preserving unit point weight."""
    scaled = core.anisotropic_scale(2.0, 0.5, 3.0)(mv.point([3.0, 4.0, 5.0]))
    assert ((scaled - mv.point([6.0, 2.0, 15.0])).antinorm() < 1e-9).all()
    assert ((mv.w & scaled) - 1).abs() < 1e-9


def test_robot_arm_kinematics():
    """Forward kinematics places five rigid bodies and four joint pivots while preserving unit weights."""
    unit_box = core.canonical_unit_box()
    bodies_to_world, pivots = core.robot_arm((0.2, -0.3, 0.5, -0.2))
    assert bodies_to_world.shape == (5,)
    assert pivots.shape == (4,)
    vertices = bodies_to_world[:, None](unit_box[None, :])
    assert (((mv.w & vertices) - 1).abs() < 1e-9).all()


def test_compound_optics_lands_on_sensor():
    """Compound two-lens refraction lands world points on the sensor plane `z == -1.25`."""
    pose = core.aim_motor(mv.point([0.0, -4.0, 1.5]), mv.point([0.0, 0.0, 1.0]))
    front_lens, rear_lens, _ = core.lens_train(1.0, 0.8, -0.3)
    pupil_shift = (mv.xw * 0.02).exp()
    pupil = pupil_shift >> core.origin
    sensor_plane = mv.z + 1.25 * mv.w
    camera = core.lens_camera(pose, front_lens, rear_lens, pupil, sensor_plane)
    sensor_hit = camera(mv.point([0.1, 0.2, 1.0]))
    assert (sensor_plane & sensor_hit).abs() < 1e-9
    np.testing.assert_allclose(euclidean(sensor_hit)[2], -1.25, atol=1e-9)


def test_figures_and_one_sweep_frame_render():
    """Static scenegraph figures and one animation frame render without saving files."""
    world_vertices, projected_pixels, camera_pose, rays = scenarios.still_scenegraph()
    for fig in (
        render.scenegraph_figure(world_vertices, projected_pixels, camera_pose, rays),
        render.world_scene_figure(world_vertices, camera_pose, rays),
        render.sensor_photograph_figure(world_vertices, projected_pixels, camera_pose),
    ):
        assert fig.axes
        plt.close(fig)

    frame = next(render.scenegraph_animation(scenarios.arm_sweep_states(1)))
    assert frame.ndim == 3 and frame.shape[-1] == 3
