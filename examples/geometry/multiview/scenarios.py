"""Scenes for multi-camera reconstruction and camera alignment in PGA2D.

A convergent rig of planar cameras views six landmarks. `bundle_adjustment` returns the
aligned rig for a figure; the convergence scenarios yield one state per Gauss-Newton step for an
animation. All of them return geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from numga.algebras import PGA2D

from examples import instantiate

core = instantiate("examples.geometry.multiview.core", PGA2D)
Camera, Information, Motor, Point, Quadric = core.Camera, core.Information, core.Motor, core.Point, core.Quadric
mv, point = core.mv, core.point

# Landmarks in front of the cameras, at depths y in [0.85, 2.55]:
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


def observe(true_motors: Motor):
    """The landmarks, the pinhole cameras, and the sight cones of their pixel measurements."""
    true_points = point(LANDMARKS)
    # Pinhole at the origin, sensor line y = 1:
    camera = (point(np.zeros(2)) & Point) ^ (mv.y - mv.w)
    cameras = camera.broadcast_to(true_motors.shape)

    # Sensor pixel measurements, in [n_points, n_cams] layout, pulled back into sight cones:
    projs = cameras(true_motors << true_points[:, None])
    pixels = projs / (mv.w & projs)
    return true_points, cameras, core.make_cones(cameras, core.sensor_disk(pixels))


def cone_cost(motors: Motor, points: Point, local_cones: Quadric) -> core.Scalar:
    """The objective: every point's cone value, summed over the cameras that see it."""
    local_points = motors << points[:, None]
    return (local_cones(local_points) & local_points).sum()


def bundle_adjustment():
    """Two convergent cameras, the second panned 5% too far, aligned by eight Gauss-Newton steps."""
    # Cam 0 on the left panned right, Cam 1 on the right panned left:
    true_motors = rig(np.array([-BASELINE, BASELINE]), np.array([GAZE, -GAZE]))
    _, _, local_cones = observe(true_motors)
    initial_motors = rig(np.array([-BASELINE, BASELINE]), np.array([GAZE, -GAZE * 1.05]))

    # Camera alignment purely via perspective cone quadrics, Cam 0 fixed as reference:
    motors, points, fused = core.bundle_adjust(initial_motors, local_cones, 8, 0.8, np.array([0.0, 1.0]))
    world_cones = motors >> local_cones(motors << Point)

    # --- checks
    # The cone cost falls by orders of magnitude from the perturbed start.
    initial_points, _ = core.triangulate_cones(initial_motors, local_cones)
    initial_cost = cone_cost(initial_motors, initial_points, local_cones).to_array()
    assert cone_cost(motors, points, local_cones).to_array() < initial_cost * 1e-3

    return motors, world_cones, fused, points


def convergence(
    true_motors: Motor, initial_motors: Motor, damping: float, free: np.ndarray, iterations: int,
):
    """The rig after each alternating Gauss-Newton step, from the initial poses."""
    _, _, local_cones = observe(true_motors)
    motors = initial_motors
    for _ in range(iterations + 1):
        points, fused = core.triangulate_cones(motors, local_cones)
        yield motors, motors >> local_cones(motors << Point), fused, points
        motors, _, _ = core.bundle_adjust(motors, local_cones, 1, damping, free)


def three_camera_truth() -> Motor:
    """Left and right cameras panned inwards, and a third at the origin looking straight ahead."""
    return rig(np.array([-BASELINE, BASELINE, 0.0]), np.array([GAZE, -GAZE, 0.0]))


def one_camera(iterations: int):
    """One moving camera, Cam 1 with a 25% pan mismatch; Cams 0 and 2 anchored."""
    # Damping 0.38 yields a steady visual trajectory reaching ~95% progress at step 9-10.
    initial_motors = rig(np.array([-BASELINE, BASELINE, 0.0]), np.array([GAZE, -GAZE * 1.25, 0.0]))
    return convergence(three_camera_truth(), initial_motors, 0.38, np.array([0.0, 1.0, 0.0]), iterations)


def two_cameras(iterations: int):
    """Two moving cameras, Cams 1 and 2 both perturbed; Cam 0 anchored."""
    # Damping 0.35 yields a steady visual trajectory reaching ~95% progress at step 10.
    initial_motors = rig(np.array([-BASELINE, BASELINE, -0.1]), np.array([GAZE, -GAZE * 1.25, 0.05]))
    return convergence(three_camera_truth(), initial_motors, 0.35, np.array([0.0, 1.0, 1.0]), iterations)


def three_cameras(iterations: int):
    """All three cameras perturbed and updating freely, without anchors."""
    # Damping 0.32 yields a steady visual trajectory reaching ~95% progress at step 10.
    initial_motors = rig(np.array([-BASELINE * 0.95, BASELINE, -0.1]), np.array([GAZE * 1.05, -GAZE * 1.25, 0.05]))
    return convergence(three_camera_truth(), initial_motors, 0.32, np.array([1.0, 1.0, 1.0]), iterations)


def schur(iterations: int):
    """One moving camera as in `one_camera`, solved by the joint Gauss-Newton step with the Schur complement.

    Each state carries the marginal information on each camera's pose from the step taken there.
    """
    true_motors = three_camera_truth()
    _, cameras, local_cones = observe(true_motors)
    motors = rig(np.array([-BASELINE, BASELINE, 0.0]), np.array([GAZE, -GAZE * 1.25, 0.0]))
    free = np.array([0.0, 1.0, 0.0])
    for _ in range(iterations + 1):
        points, fused = core.triangulate_cones(motors, local_cones)
        next_motors, _, _, information = core.bundle_adjust_schur(cameras, motors, local_cones, 1, 0.7, free)
        yield motors, motors >> local_cones(motors << Point), fused, points, information
        motors = next_motors


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.geometry.multiview import render

    save_figure(render.draw_reconstruction(*bundle_adjustment()), "multiview_bundle_adjustment")
    save_animation(render.animate_convergence(one_camera(10)), "multiview_convergence_1cam", 333)
    save_animation(render.animate_convergence(two_cameras(10)), "multiview_convergence_2cams", 333)
    save_animation(render.animate_convergence(three_cameras(10)), "multiview_convergence_3cams", 333)
    save_animation(render.animate_convergence_with_covariance(schur(10)), "multiview_convergence_schur", 333)
