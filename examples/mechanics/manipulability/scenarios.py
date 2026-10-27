"""Scenes for the arm's velocity and force ellipsoids: two poses side by side, and the arm swept around
a closed loop through joint space."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.manipulability import core

# Joint angles: base yaw, shoulder, elbow and wrist pitch.
REACHING = (0.35, 0.5, 1.2, 0.4)
NEARLY_STRAIGHT = (0.35, 0.2, 0.15, 0.1)
# How much the velocity ellipsoid is grown and the force ellipsoid shrunk for display; their radii are
# reciprocal.
VELOCITY_SCALE = 0.25
FORCE_SCALE = 2.5
# The loop through joint space: each joint swings about a centre, with an amplitude, a frequency and a
# phase, in radians; the elbow stays bent by at least half a radian, so neither ellipsoid degenerates.
LOOP_CENTRES = np.array([0.35, 0.35, 1.25, 0.4])
LOOP_AMPLITUDES = np.array([0.9, 0.35, 0.75, 0.8])
LOOP_FREQUENCIES = np.array([1, 1, 1, 2])
LOOP_PHASES = np.array([0.0, 1.5, 0.3, 0.0])
# A fixed view of the whole workspace: (elevation, azimuth) in degrees, the centre and half-width.
VIEW, CENTRE, EXTENT = (18, -60), np.array([-0.9, 0.25, 1.3]), 2.2
# Pixels across each panel of the animation, the frames of the loop, and the duration of a frame in milliseconds.
FRAME_PIXELS = 300
N_FRAMES = 96
FRAME_DURATION = 60
# Unit tip forces checked against the joint torques they produce.
CHECKED_FORCES = np.array([[0.3, -0.2, 0.5], [1.0, 0.4, -0.7]])


# --- math -----------------------------------------------------------------------------
def ellipsoids(joint_angles: tuple, velocity_scale: float, force_scale: float):
    """The arm in one pose: body maps, tip, and its velocity and force ellipsoids as point quadrics.

    Both are built from the joint twists, independently; the display scales grow the velocity
    ellipsoid and shrink the force ellipsoid, whose radii are reciprocal to it.
    """
    bodies, tip, axes = core.arm_axes(joint_angles)
    # The velocity ellipsoid as a point quadric.
    velocity = core.velocity_ellipsoid(tip, axes * velocity_scale).inverse()   # [] Plane+ <- Point-
    force = core.force_ellipsoid(tip, axes * force_scale)                      # [] Plane+ <- Point-
    return bodies, tip, velocity, force


def sweep(n_frames: int) -> Iterator[tuple]:
    """The arm around the closed loop through joint space, turning and reshaping its ellipsoids."""
    for phase in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        angles = LOOP_CENTRES + LOOP_AMPLITUDES * np.sin(LOOP_FREQUENCIES * phase + LOOP_PHASES)
        yield ellipsoids(tuple(angles), VELOCITY_SCALE, FORCE_SCALE)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.manipulability import render

    poses = [ellipsoids(angles, VELOCITY_SCALE, FORCE_SCALE) for angles in (REACHING, NEARLY_STRAIGHT)]
    save_figure(render.draw_ellipsoids(poses, VIEW, CENTRE, EXTENT), "manipulability")
    frames = (render.frame(pose, VIEW, CENTRE, EXTENT, FRAME_PIXELS) for pose in sweep(N_FRAMES))
    save_animation(frames, "manipulability_sweep", FRAME_DURATION)

    # checks
    # In each pose the force ellipsoid is the velocity ellipsoid's polar in the unit sphere about the tip.
    # A unit tip force lies on the force ellipsoid exactly when its joint torques, each joint's twist paired
    # with the force's line through the tip, have unit squared sum; and the velocity ellipsoid reaches,
    # along a direction, as far as a unit force that way loads the joints.
    for angles in (REACHING, NEARLY_STRAIGHT):
        _, tip, axes = core.arm_axes(angles)
        unit_velocity, unit_force = core.velocity_ellipsoid(tip, axes), core.force_ellipsoid(tip, axes)
        sphere = core.unit_sphere(tip)
        np.testing.assert_allclose(unit_force.kernel, sphere(unit_velocity(sphere(core.Point))).kernel, rtol=1e-8, atol=1e-8)
        along = core.mv.direction(CHECKED_FORCES / np.linalg.norm(CHECKED_FORCES, axis=-1, keepdims=True))   # [forces] Direction-
        torques = (axes[:, None] & (tip & along)).to_array()                   # [joints, forces]
        held = tip + along
        np.testing.assert_allclose((held & unit_force(held)).to_array(), 1 - (torques**2).sum(axis=0), atol=1e-8)
        radius = np.sqrt((torques**2).sum(axis=0))
        normal = along.dual()
        tangent = normal - core.w * (normal & (tip + along * radius))
        np.testing.assert_allclose((tangent & unit_velocity(tangent)).to_array(), 0.0, atol=1e-8)


if __name__ == "__main__":
    main()
