"""Multi-view reconstruction and camera alignment recover landmarks and camera poses with sight cones in 2D and 3D."""

from __future__ import annotations

import signal
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack
from numga.algebras import PGA2D, PGA3D

from examples import instantiate
from examples.estimation.multiview import render, scenarios
from examples.estimation.multiview.render import coordinates

core = instantiate("examples.estimation.multiview.core", PGA2D)
Point, mv = core.Point, core.mv


@pytest.fixture(autouse=True)
def ten_second_budget():
    """Every test in this module finishes within ten seconds."""
    def expired(signum, frame):
        raise TimeoutError("test exceeded the ten second budget")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def rms_distance(points: Point, truth: Point):
    """Root-mean-square Euclidean distance between unit-weight points from the antinorm of their difference."""
    return (points - truth).antinorm().squared().mean(axis=0).square_root()


def pose_discrepancy(first_motors: core.Motor, second_motors: core.Motor):
    """Combined pinhole translation distance and half-rotation angle between two motor arrays."""
    pinhole = mv.point([0, 0])
    shifts = ((first_motors >> pinhole) - (second_motors >> pinhole)).antinorm()
    turns = (first_motors * second_motors.reverse()).log().norm()
    return (shifts + turns).max()


def test_triangulation_from_cones():
    """Perspective cone pullback and fusion reconstruct landmarks given true poses."""
    rng = np.random.default_rng(123)
    xy = rng.uniform([-0.6, 1.0], [0.6, 2.5], size=(6, 2))
    true_points = mv.point(xy)
    motors = scenarios.rig_poses(np.array([-0.75, 0.75]), np.radians([18.0, -18.0]))

    camera = (mv.point([0, 0]) & Point) ^ (mv.y - mv.w)
    cameras = camera.broadcast_to((2,))
    images = cameras(motors << true_points[:, None])
    pixels = images / (mv.w & images)

    sensor_ball = mv.x * (mv.x & Point)
    balls = core.shifted_sensor_balls(pixels, mv.point([0, 1]), sensor_ball)
    local_cones = cameras.adjugate()(balls(cameras))
    points, fused_splats = core.triangulation(motors, local_cones)
    np.testing.assert_allclose(coordinates(points), xy, atol=1e-5)

    # Fused precision stays positive along every spatial direction, bounding each splat into an ellipse.
    angles = np.linspace(0.0, np.pi, 12, endpoint=False)
    turns = (mv.xy * (-angles / 2)).exp()
    displacements = turns >> mv.direction([1, 0])
    precision = fused_splats[:, None](displacements[None, :]) & displacements[None, :]
    assert (precision > 0).all()


def test_alternating_alignment_and_pose_uncertainties():
    """Alternating Gauss-Newton steps recover a perturbed camera and yield centred pose ellipsoids."""
    true_motors = scenarios.rig_poses(scenarios.camera_offsets, scenarios.camera_turns)
    true_points, _, local_cones = scenarios.scene_observations(true_motors)
    initial_motors = scenarios.rig_poses(np.array([-0.75, 0.75, 0.0]), np.radians([18.0, -18.0 * 1.05, 0.0]))
    initial_points, _ = core.triangulation(initial_motors, local_cones)

    _, points, _ = core.alternating_alignment(initial_motors, local_cones, 12, 0.9, np.array([0.0, 1.0, 0.0]))
    assert rms_distance(points, true_points) < rms_distance(initial_points, true_points)
    assert rms_distance(points, true_points) < 0.02

    pinhole = mv.point([0, 0])
    states = list(islice(
        core.alignment_steps(initial_motors, local_cones, pinhole, np.array([0.0, 1.0, 0.0]), 4.0, 0.08, 1.5), 6,
    ))
    final_poses, _, _, step_points, final_curvature = states[-1]
    assert rms_distance(step_points, true_points) < rms_distance(initial_points, true_points)
    ellipsoids, spreads = core.pose_uncertainties(
        final_poses, final_curvature, pinhole, mv.x, mv.direction([0, 1]), 0.035, 3.0,
    )
    assert (ellipsoids(final_poses >> pinhole).norm() < 1e-10).all()
    assert (spreads > 0).all()


def test_joint_alignment_converges_and_yields_information():
    """Coupled camera-and-landmark steps converge and return a symmetric pose information form."""
    true_motors = scenarios.rig_poses(scenarios.camera_offsets, scenarios.camera_turns)
    true_points, cameras, local_cones = scenarios.scene_observations(true_motors)
    initial_motors = scenarios.rig_poses(np.array([-0.75, 0.75, 0.0]), np.radians([18.0, -18.0 * 1.05, 0.0]))
    initial_points, _ = core.triangulation(initial_motors, local_cones)

    _, points, _, information = core.joint_alignment(
        cameras, initial_motors, local_cones, 10, 0.7, np.array([0.0, 1.0, 0.0]),
    )
    assert rms_distance(points, true_points) < rms_distance(initial_points, true_points)
    assert rms_distance(points, true_points) < 0.02

    assert information.gatype.site_shape == (3, 3)
    basis = mv.bivector(np.eye(3))
    units = (basis[None, :, None] * np.eye(3)[:, None, :]).field()              # [3, 3] Twist+[3]
    gram = information(units[:, :, None, None], units[None, None]).to_array()   # [3, 3, 3, 3]
    np.testing.assert_allclose(gram, gram.transpose(2, 3, 0, 1), atol=1e-9)
    np.testing.assert_allclose(gram[[0, 2]], 0.0)
    np.testing.assert_allclose(gram[:, :, [0, 2]], 0.0)
    assert np.linalg.eigvalsh(gram[1, :, 1]).min() > 0


def test_joint_alignment_moves_two_free_cameras_together():
    """Coupled Schur-complement steps align two free cameras within a handful of iterations."""
    true_motors = scenarios.rig_poses(scenarios.camera_offsets, scenarios.camera_turns)
    _, cameras, local_cones = scenarios.scene_observations(true_motors)
    motors = scenarios.rig_poses(
        scenarios.camera_offsets * [1.0, 1.1, 1.0] + [0.0, 0.0, 0.15],
        scenarios.camera_turns * [1.0, 1.25, 1.0] + [0.0, 0.0, 0.12],
    )
    free_cameras = np.array([0.0, 1.0, 1.0])

    def cone_cost(current_motors):
        points, _ = core.triangulation(current_motors, local_cones)
        world_cones = current_motors >> local_cones(current_motors << Point)
        return (world_cones(points[:, None]) & points[:, None]).sum()

    initial_cost = cone_cost(motors)
    for _ in range(5):
        previous, motors = motors, core.joint_alignment(cameras, motors, local_cones, 1, 1.0, free_cameras)[0]
    assert cone_cost(motors) < 1e-5 * initial_cost
    assert pose_discrepancy(motors, previous) < 1e-5


def test_ray_splat_alignment_matches_joint_alignment():
    """Ray-to-splat alignment recovers a perturbed camera without instantiating landmarks."""
    true_motors = scenarios.rig_poses(scenarios.camera_offsets, scenarios.camera_turns)
    true_points, cameras, local_cones = scenarios.scene_observations(true_motors)
    initial_motors = scenarios.rig_poses(np.array([-0.75, 0.75, 0.0]), np.radians([18.0, -18.0 * 1.05, 0.0]))
    free_cameras = np.array([0.0, 1.0, 0.0])
    images = cameras(true_motors << true_points[:, None])
    pixels = images / (mv.w & images)

    splat_motors, _ = core.ray_splat_alignment(
        initial_motors, local_cones, mv.point([0, 0]), pixels, 4, 1.0, free_cameras,
    )
    joint_motors, _, _, _ = core.joint_alignment(cameras, initial_motors, local_cones, 10, 1.0, free_cameras)
    assert pose_discrepancy(splat_motors, true_motors) < 1e-9
    assert pose_discrepancy(splat_motors, joint_motors) < 1e-9


# --- 3D instantiation ---------------------------------------------------------------------
core3 = instantiate("examples.estimation.multiview.core", PGA3D)


def coordinates_3d(points) -> np.ndarray:
    """Cartesian `(x, y, z)` coordinates of 3D points from pairings with the coordinate planes."""
    mv3 = points.context.multivector
    return (stack((mv3.x, mv3.y, mv3.z)) & (points / (mv3.w & points))[..., None]).to_array()


def rig_3d(instance) -> tuple:
    """Three convergent cameras, eight landmarks, and local sight cones in 3D projective geometric algebra."""
    mv3 = instance.mv
    theta = np.radians(18.0)
    true_motors = stack([
        (-mv3.xw * 0.75 / 2).exp() * (-mv3.zx * theta / 2).exp(),
        (mv3.xw * 0.75 / 2).exp() * (mv3.zx * theta / 2).exp(),
        (mv3.yw * 0.35 / 2).exp() * (-mv3.yz * np.radians(12.0) / 2).exp(),
    ])
    camera = (mv3.point([0, 0, 0]) & instance.Point) ^ (mv3.z - mv3.w)
    cameras = camera.broadcast_to((3,))
    xyz = np.array([
        [0.15, -0.20, 1.20],
        [-0.43, 0.15, 1.45],
        [0.50, -0.10, 1.70],
        [0.03, 0.25, 1.95],
        [0.65, -0.15, 2.30],
        [-0.60, 0.10, 2.65],
        [0.20, 0.30, 2.10],
        [-0.25, -0.25, 1.60],
    ])
    images = cameras(true_motors << mv3.point(xyz)[:, None])
    pixels = images / (mv3.w & images)
    sensor_ball = mv3.x * (mv3.x & instance.Point) + mv3.y * (mv3.y & instance.Point)
    balls = instance.shifted_sensor_balls(pixels, mv3.point([0, 0, 1]), sensor_ball)
    return true_motors, xyz, cameras.adjugate()(balls(cameras))


@pytest.mark.parametrize("rotation_deg, translation, n_iterations", [
    ((30.0, -20.0, 25.0), (-0.40, 0.30, 0.45), 30),
    ((50.0, 35.0, -40.0), (0.60, -0.50, 0.80), 40),
])
def test_3d_alignment_recovers_large_perturbation(rotation_deg, translation, n_iterations):
    """In 3D, alternating alignment recovers a camera perturbed by tens of degrees and most of a metre."""
    mv3 = core3.mv
    true_motors, xyz, local_cones = rig_3d(core3)
    rx, ry, rz = np.radians(rotation_deg)
    tx, ty, tz = translation
    perturbation = (
        ((mv3.xw * tx + mv3.yw * ty + mv3.zw * tz) / 2).exp()
        * ((mv3.yz * rx + mv3.zx * ry + mv3.xy * rz) / 2).exp()
    )
    motors = stack([true_motors[0], perturbation * true_motors[1], true_motors[2]])

    def cone_cost(current_motors, points):
        local_points = current_motors << points[:, None]
        return (local_cones(local_points) & local_points).sum()

    initial_points, _ = core3.triangulation(motors, local_cones)
    est_motors, est_points, _ = core3.alternating_alignment(
        motors, local_cones, n_iterations, 0.7, np.array([0.0, 1.0, 0.0]),
    )

    origin = mv3.point([0, 0, 0])
    np.testing.assert_allclose(coordinates_3d(est_motors[1] >> origin), [0.75, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(coordinates_3d(est_points), xyz, atol=0.02)
    assert est_motors[1].scalar_product(true_motors[1].reverse()).abs() > np.cos(np.radians(0.25))
    assert cone_cost(est_motors, est_points) < cone_cost(motors, initial_points) * 1e-3


def test_3d_alignment_converges_from_small_perturbation():
    """In 3D, a 5% pan perturbation on the second camera aligns with the other two cameras anchored."""
    mv3 = core3.mv
    theta = np.radians(18.0)
    true_motors, xyz, local_cones = rig_3d(core3)
    motors = stack([
        true_motors[0],
        (mv3.xw * 0.75 / 2).exp() * (mv3.zx * theta * 1.05 / 2).exp(),
        true_motors[2],
    ])
    truth = mv3.point(xyz)
    initial_points, _ = core3.triangulation(motors, local_cones)
    _, points, _ = core3.alternating_alignment(motors, local_cones, 10, 0.9, np.array([0.0, 1.0, 0.0]))
    initial_err = (initial_points - truth).antinorm().squared().mean(axis=0).square_root()
    final_err = (points - truth).antinorm().squared().mean(axis=0).square_root()
    assert final_err < initial_err
    assert final_err < 0.02


# --- scenarios and figures through render -------------------------------------------------
def test_static_figures_and_paced_trajectory_render():
    """All static figure builders and the paced trajectory helper render without writing files."""
    poses, world_cones, splats, centres = scenarios.aligned_pair()
    pinhole = mv.point([0, 0])
    _, _, local_cones = scenarios.scene_observations(scenarios.rig_poses(scenarios.camera_offsets, scenarios.camera_turns))
    full_poses, full_cones, full_splats, full_centres, curvature = next(
        core.alignment_steps(
            scenarios.rig_poses(scenarios.camera_offsets, scenarios.camera_turns),
            local_cones, pinhole, np.array([0.0, 1.0, 0.0]), 4.0, 0.08, 1.5,
        )
    )
    ellipsoids, spreads = core.pose_uncertainties(
        full_poses, curvature, pinhole, mv.x, mv.direction([0, 1]), 0.035, 3.0,
    )
    for fig in (
        render.rig_figure(poses),
        render.cones_figure(poses, world_cones),
        render.reconstruction_figure(poses, world_cones, splats, centres),
        render.covariance_figure(full_poses, full_cones, full_splats, full_centres, ellipsoids, spreads),
    ):
        assert fig.axes
        plt.close(fig)

    paced = list(scenarios.paced_trajectory(list(islice(scenarios.three_cameras(), 3)), 2, 1))
    assert len(paced) == 3


@pytest.mark.parametrize("name", ["two_cameras", "joint_one_camera"])
def test_convergence_animations_render(name):
    """Alternating and joint scenario generators render RGB frames through `convergence_animation`."""
    frames = list(render.convergence_animation(islice(getattr(scenarios, name)(), 2)))
    assert len(frames) == 2 and frames[0].ndim == 3 and frames[0].shape == frames[1].shape
