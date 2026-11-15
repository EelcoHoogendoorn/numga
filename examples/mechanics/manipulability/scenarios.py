"""Scene parameters and verification run for the articulated arm's velocity and force ellipsoids."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.mechanics.manipulability import core

# Joint angles (base yaw, shoulder, elbow, and wrist pitch) for a comfortable reach and a nearly straight arm.
reaching_angles = (0.35, 0.5, 1.2, 0.4)
straight_angles = (0.35, 0.2, 0.15, 0.1)
# Reciprocal display scales that enlarge the velocity ellipsoid and shrink the force ellipsoid.
velocity_scale = 0.25
force_scale = 2.5
# Sinusoidal closed loop in joint space keeping the elbow bent by at least half a radian.
loop_centres = np.array([0.35, 0.35, 1.25, 0.4])
loop_amplitudes = np.array([0.9, 0.35, 0.75, 0.8])
loop_frequencies = np.array([1, 1, 1, 2])
loop_phases = np.array([0.0, 1.5, 0.3, 0.0])
# Camera elevation/azimuth in degrees and fixed workspace view centre and half-width.
view_degrees = (18, -60)
workspace_centre = core.mv.point([-0.9, 0.25, 1.3])                         # Point-
workspace_extent = 2.2
n_pixels = 300
n_frames = 144
frame_duration_ms = 40
# Sample force directions at the gripper tip for virtual-work verification.
sample_directions = core.mv.direction([[0.3, -0.2, 0.5], [1.0, 0.4, -0.7]])  # [2] Point-


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Renders the static two-pose figure and the joint-space loop animation, then verifies polar reciprocity."""
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.manipulability import render

    static_poses = [
        core.pose_ellipsoids(angles, velocity_scale, force_scale)
        for angles in (reaching_angles, straight_angles)
    ]
    save_figure(render.ellipsoids_figure(static_poses, view_degrees), "manipulability")

    streamed_poses: list[tuple[core.PointMap, core.Point, core.Quadric, core.Quadric]] = []

    def recorded_poses() -> Iterator[tuple[core.PointMap, core.Point, core.Quadric, core.Quadric]]:
        for pose in core.joint_loop_poses(
            n_frames,
            loop_centres,
            loop_amplitudes,
            loop_frequencies,
            loop_phases,
            velocity_scale,
            force_scale,
        ):
            streamed_poses.append(pose)
            yield pose

    frames = render.animate_poses(view_degrees, workspace_centre, workspace_extent, n_pixels, recorded_poses())
    save_animation(frames, "manipulability_sweep", frame_duration_ms)

    # checks
    basis_points = stack([core.mv.w, core.mv.x, core.mv.y, core.mv.z]).dual()   # [4] Point-
    unit_directions = sample_directions / sample_directions.antinorm()          # [2] Point-
    for _, pose_tip, scaled_velocity, scaled_force in [*static_poses, streamed_poses[0], streamed_poses[n_frames // 2]]:
        # Removing the display scales recovers exact polar reciprocity in the unit sphere centred on the tip.
        sphere = core.unit_sphere(pose_tip)                                     # [] Plane+ <- Point-
        unscaled_velocity = (
            (scaled_velocity.inverse() + pose_tip * (core.Plane & pose_tip)) / (velocity_scale**2)
            - pose_tip * (core.Plane & pose_tip)
        )                                                                       # [] Point- <- Plane+
        unscaled_force = (
            (scaled_force + core.w * (core.w & core.Point)) / (force_scale**2)
            - core.w * (core.w & core.Point)
        )                                                                       # [] Plane+ <- Point-
        polar_residual = (unscaled_force - sphere(unscaled_velocity(sphere(core.Point))))(basis_points[None, :]) & basis_points[:, None]  # [4, 4] Scalar+
        assert (polar_residual.abs() < 1e-8).all()

    for angles in (reaching_angles, straight_angles):
        _, tip, axes = core.arm_axes(angles)
        unit_velocity = core.velocity_ellipsoid(tip, axes)                      # [] Point- <- Plane+
        unit_force = core.force_ellipsoid(tip, axes)                            # [] Plane+ <- Point-
        sphere = core.unit_sphere(tip)                                          # [] Plane+ <- Point-
        # A unit tip force evaluates on the force quadric to `torque_norm_sq - 1`, vanishing when joint torques sum to 1.
        torques = axes[:, None] & (tip & unit_directions)                       # [4, 2] Scalar+
        torque_norm_sq = torques.squared().sum(axis=0)                          # [2] Scalar+
        loaded_points = tip + unit_directions                                   # [2] Point-
        assert (((unit_force(loaded_points) & loaded_points) - (torque_norm_sq - 1)).abs() < 1e-8).all()
        # Along each direction, the velocity ellipsoid's tangent plane sits at radius `sqrt(torque_norm_sq)` from the tip.
        reach_radii = torque_norm_sq.square_root()                              # [2] Scalar+
        tangent_planes = sphere(tip + unit_directions / reach_radii)            # [2] Plane+
        assert ((tangent_planes & unit_velocity(tangent_planes)).abs() < 1e-8).all()


if __name__ == "__main__":
    main()
