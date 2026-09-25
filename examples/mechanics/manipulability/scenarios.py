"""Scenes for the arm's velocity and force ellipsoids.

One function per figure. Poses the arm, calls the mathematics in `core`, and returns the geometry
for `render`.
"""

from __future__ import annotations

import numpy as np

from examples.mechanics.manipulability import core

# Joint angles: base yaw, shoulder, elbow and wrist pitch.
REACHING = (0.35, 0.5, 1.2, 0.4)
NEARLY_STRAIGHT = (0.35, 0.2, 0.15, 0.1)
# A fixed view of the whole workspace: (elevation, azimuth) in degrees, the centre and half-width.
VIEW, CENTRE, EXTENT = (18, -60), np.array([-0.9, 0.25, 1.3]), 2.2


def ellipsoids(joint_angles: tuple, velocity_scale: float, force_scale: float):
    """The arm in one pose: body maps, tip, and its velocity and force ellipsoids as point quadrics.

    Both are built from the joint axes, independently; the display scales grow the velocity
    ellipsoid and shrink the force ellipsoid, whose radii are reciprocal to it.
    """
    bodies, tip, axes = core.arm_axes(joint_angles)
    # The velocity ellipsoid as a point quadric.
    velocity = core.velocity_ellipsoid(tip, axes * velocity_scale).inverse()   # [] Plane <- Point
    force = core.force_ellipsoid(tip, axes * force_scale)

    # --- checks
    unit_velocity, unit_force = core.velocity_ellipsoid(tip, axes), core.force_ellipsoid(tip, axes)
    # Duality: the force ellipsoid is the velocity ellipsoid's polar in the unit sphere about the tip.
    sphere = core.unit_sphere(tip)
    np.testing.assert_allclose(unit_force.kernel, sphere(unit_velocity(sphere(core.Point))).kernel, rtol=1e-8, atol=1e-8)
    # Both against the joint torques a unit tip force produces: each joint's axis paired with the
    # force's line through the tip, as in the robot arm example.
    unit = np.array([[0.3, -0.2, 0.5], [1.0, 0.4, -0.7]])
    along = core.mv("x y z", unit / np.linalg.norm(unit, axis=-1, keepdims=True)).dual()   # [n_forces] Direction
    torques = (axes[:, None] & (tip & along)).to_array()                                # [n_joints, n_forces]
    # A force lies on the force ellipsoid exactly when its joint torques have unit squared sum.
    held = tip + along
    np.testing.assert_allclose((held & unit_force(held)).to_array(), 1 - (torques**2).sum(axis=0), atol=1e-8)
    # The velocity ellipsoid reaches, along a direction, as far as a unit force that way loads the
    # joints: the plane normal to it at that distance from the tip is tangent.
    reach = core.mv.scalar(np.sqrt((torques**2).sum(axis=0))[:, None])
    normal = along.dual()
    tangent = normal - core.w * (normal & (tip + along * reach))
    np.testing.assert_allclose((tangent & unit_velocity(tangent)).to_array(), 0.0, atol=1e-8)

    return bodies, tip, velocity, force


def sweep(frames: int):
    """The arm around a closed loop through joint space, turning and reshaping its ellipsoids.

    Each joint swings about a centre, (centre, amplitude, frequency, phase) in radians; the elbow
    stays bent by at least half a radian, so the arm never straightens and neither ellipsoid
    degenerates.
    """
    loop = ((0.35, 0.9, 1, 0.0), (0.35, 0.35, 1, 1.5), (1.25, 0.75, 1, 0.3), (0.4, 0.8, 2, 0.0))
    for phase in np.linspace(0.0, 2 * np.pi, frames, endpoint=False):
        angles = tuple(centre + amplitude * np.sin(frequency * phase + offset)
                       for centre, amplitude, frequency, offset in loop)
        yield ellipsoids(angles, 0.25, 2.5)


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.mechanics.manipulability import render

    poses = [ellipsoids(angles, 0.25, 2.5) for angles in (REACHING, NEARLY_STRAIGHT)]
    save_figure(render.draw_ellipsoids(poses, VIEW, CENTRE, EXTENT, 400), "manipulability")
    frames = [render.frame(pose, VIEW, CENTRE, EXTENT, 300) for pose in sweep(96)]
    save_animation(frames, "manipulability_sweep", 60)
