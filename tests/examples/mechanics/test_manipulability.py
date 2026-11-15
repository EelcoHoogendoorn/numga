"""Coverage tests for the articulated arm's velocity and force manipulability ellipsoids."""

from __future__ import annotations

import matplotlib.pyplot as plt

from numga import stack
from examples.mechanics.manipulability import core, render, scenarios


def test_ellipsoid_reciprocity_and_unit_torque_loading() -> None:
    """The force ellipsoid is the polar of the velocity ellipsoid in the tip's unit sphere and measures squared joint torques."""
    _, tip, axes = core.arm_axes((0.1, 0.7, 0.9, -0.3))
    velocity = core.velocity_ellipsoid(tip, axes)                               # [] Point- <- Plane+
    force = core.force_ellipsoid(tip, axes)                                     # [] Plane+ <- Point-
    sphere = core.unit_sphere(tip)                                              # [] Plane+ <- Point-
    basis_points = stack([core.mv.w, core.mv.x, core.mv.y, core.mv.z]).dual()   # [4] Point-
    polar_residual = (force - sphere(velocity(sphere(core.Point))))(basis_points[None, :]) & basis_points[:, None]  # [4, 4] Scalar+
    assert (polar_residual.abs() < 1e-8).all()

    force_direction = core.mv.direction([[0.0, 0.6, 0.8]])                      # [1] Point-
    torques = axes[:, None] & (tip & force_direction)                           # [4, 1] Scalar+
    torque_norm_sq = torques.squared().sum(axis=0)                              # [1] Scalar+
    loaded_point = tip + force_direction                                        # [1] Point-
    assert (((force(loaded_point) & loaded_point) - (torque_norm_sq - 1)).abs() < 1e-8).all()


def test_figure_and_loop_frames_render() -> None:
    """Both the multi-pose figure and the streaming joint-space loop render RGB images."""
    pose_stream = core.joint_loop_poses(
        1,
        scenarios.loop_centres,
        scenarios.loop_amplitudes,
        scenarios.loop_frequencies,
        scenarios.loop_phases,
        scenarios.velocity_scale,
        scenarios.force_scale,
    )
    poses = list(pose_stream)
    figure = render.ellipsoids_figure(poses, scenarios.view_degrees)
    plt.close(figure)
    frames = list(
        render.animate_poses(
            scenarios.view_degrees,
            scenarios.workspace_centre,
            scenarios.workspace_extent,
            40,
            poses,
        )
    )
    assert len(frames) == 1 and frames[0].ndim == 3 and frames[0].shape[2] == 3
