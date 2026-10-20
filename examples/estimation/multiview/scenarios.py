"""Scenes for multi-camera reconstruction and camera alignment in PGA2D.

A convergent rig of planar cameras views six landmarks. `bundle_adjustment` returns the
aligned rig for a figure; the convergence scenarios yield one state per Gauss-Newton step for an
animation. All of them return geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from numga.algebras import PGA2D

from examples import instantiate

core = instantiate("examples.estimation.multiview.core", PGA2D)
Camera, Information, Motor, Point, Quadric = core.Camera, core.Information, core.Motor, core.Point, core.Quadric
mv = core.mv

# Landmarks in front of the cameras, at depths y from 0.85 to 2.55:
LANDMARKS = np.array([
    [ 0.15, 0.85],
    [-0.43, 1.15],
    [ 0.50, 1.50],
    [ 0.03, 1.85],
    [ 0.65, 2.20],
    [-0.60, 2.55],
])
BASELINE = 0.75
GAZE = np.radians(18.0)


def rig(offsets: np.ndarray, gazes: np.ndarray) -> Motor:
    """Camera poses at offsets along the x axis, each panned by its gaze angle."""
    return (mv.xw * offsets / 2).exp() * (mv.xy * gazes / 2).exp()


def observe(true_poses: Motor):
    """The landmarks, the pinhole cameras, and the sight cones of their pixel measurements."""
    true_points = mv.point(LANDMARKS)
    # Pinhole at the origin, sensor line y == 1:
    camera = (mv.point(np.zeros(2)) & Point) ^ (mv.y - mv.w)
    cameras = camera.broadcast_to(true_poses.shape)

    # Sensor pixel measurements, in [n_points, n_cams] layout, pulled back into sight cones:
    images = cameras(true_poses << true_points[:, None])
    pixels = images / (mv.w & images)
    return true_points, cameras, core.sight_cones(cameras, core.sensor_disc(pixels))


def cone_cost(poses: Motor, points: Point, cones: Quadric) -> core.Scalar:
    """The objective: every point's cone value, summed over the cameras that see it."""
    local_points = poses << points[:, None]
    return (cones(local_points) & local_points).sum()


def bundle_adjustment():
    """Two convergent cameras, the second panned 5% too far, aligned by eight Gauss-Newton steps."""
    # Cam 0 on the left panned right, Cam 1 on the right panned left:
    true_poses = rig(np.array([-BASELINE, BASELINE]), np.array([GAZE, -GAZE]))
    _, _, cones = observe(true_poses)
    guess = rig(np.array([-BASELINE, BASELINE]), np.array([GAZE, -GAZE * 1.05]))

    # Camera alignment purely via perspective cone quadrics, Cam 0 fixed as reference:
    poses, points, splats = core.align(guess, cones, 8, 0.8, np.array([0.0, 1.0]))
    world_cones = poses >> cones(poses << Point)

    # --- checks
    # The cone cost vanishes at the aligned rig.
    assert abs(cone_cost(poses, points, cones).to_array()) < 1e-12

    return poses, world_cones, splats, points


def convergence(
    true_poses: Motor, guess: Motor, damping: float, moving: np.ndarray, n_steps: int,
):
    """The rig after each alternating Gauss-Newton step, from the initial poses."""
    _, _, cones = observe(true_poses)
    poses = guess
    for _ in range(n_steps + 1):
        points, splats = core.triangulate(poses, cones)
        yield poses, poses >> cones(poses << Point), splats, points
        poses, _, _ = core.align(poses, cones, 1, damping, moving)


def three_camera_truth() -> Motor:
    """Left and right cameras panned inwards, and a third at the origin looking straight ahead."""
    return rig(np.array([-BASELINE, BASELINE, 0.0]), np.array([GAZE, -GAZE, 0.0]))


def one_camera(n_steps: int):
    """One moving camera, Cam 1 with a 25% pan mismatch; Cams 0 and 2 anchored."""
    # Damping 0.38 yields a steady visual trajectory reaching ~95% progress at step 9-10.
    guess = rig(np.array([-BASELINE, BASELINE, 0.0]), np.array([GAZE, -GAZE * 1.25, 0.0]))
    return convergence(three_camera_truth(), guess, 0.38, np.array([0.0, 1.0, 0.0]), n_steps)


def two_cameras(n_steps: int):
    """Two moving cameras, Cams 1 and 2 both perturbed; Cam 0 anchored."""
    # Damping 0.35 yields a steady visual trajectory reaching ~95% progress at step 10.
    guess = rig(np.array([-BASELINE, BASELINE, -0.1]), np.array([GAZE, -GAZE * 1.25, 0.05]))
    return convergence(three_camera_truth(), guess, 0.35, np.array([0.0, 1.0, 1.0]), n_steps)


def three_cameras(n_steps: int):
    """All three cameras perturbed and updating freely, without anchors."""
    # Damping 0.1 keeps every step a tenth shorter than the one before: a gradual trajectory.
    guess = rig(np.array([-BASELINE * 0.85, BASELINE, -0.25]), np.array([GAZE * 1.15, -GAZE * 1.65, 0.15]))
    return convergence(three_camera_truth(), guess, 0.1, np.array([1.0, 1.0, 1.0]), n_steps)


def schur(n_steps: int):
    """One moving camera as in `one_camera`, solved by the joint Gauss-Newton step with the Schur complement.

    Each state carries the marginal information on each camera's pose from the step taken there.
    """
    true_poses = three_camera_truth()
    _, cameras, cones = observe(true_poses)
    poses = rig(np.array([-BASELINE, BASELINE, 0.0]), np.array([GAZE, -GAZE * 1.25, 0.0]))
    moving = np.array([0.0, 1.0, 0.0])
    for _ in range(n_steps + 1):
        points, splats = core.triangulate(poses, cones)
        next_poses, _, _, information = core.align_jointly(cameras, poses, cones, 1, 0.7, moving)
        yield poses, poses >> cones(poses << Point), splats, points, information
        poses = next_poses


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.estimation.multiview import render

    save_figure(render.draw_reconstruction(*bundle_adjustment()), "multiview_bundle_adjustment")
    save_animation(render.animate_convergence(one_camera(10)), "multiview_convergence_1cam", 333)
    save_animation(render.animate_convergence(two_cameras(10)), "multiview_convergence_2cams", 333)
    save_animation(render.animate_convergence(three_cameras(30)), "multiview_convergence_3cams", 120)
    save_animation(render.animate_convergence_with_covariance(schur(10)), "multiview_convergence_schur", 333)
