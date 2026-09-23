"""Scenes and entry points for epipolar geometry and two-view 3D reconstruction.

One function per figure. Builds the synthetic scene, injects sensor noise, calls the
pure GA mathematics in `core`, and hands the resulting geometry to `render`.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.geometry.epipolar import core, render

ga = core.ga
ctx = core.ctx
mv = core.mv


# --- plumbing: coordinate conversion helpers -------------------------------
def point(xyz: np.ndarray) -> core.Point:
    """Lift an array of Cartesian (x, y, z) coordinates to PGA3D Points."""
    return mv.yzw * xyz[..., 0] + mv.zxw * xyz[..., 1] + mv.xyw * xyz[..., 2] + mv.zyx


def coordinates(points: core.Point) -> np.ndarray:
    """Extract Cartesian (x, y, z) coordinates from PGA3D Points."""
    values = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def rotation_matrix(motor: core.Motor) -> np.ndarray:
    """Extract 3x3 rotation matrix by evaluating the motor's rotor on spatial axes."""
    # A motor is a translator times a rotor, M = T R = R + (w-terms); without the w blades, R remains.
    rotor = motor.select_subspace(ga.subspace("1 yz zx xy")).normalized()
    axes = (mv.x, mv.y, mv.z)
    rotated = [(rotor >> ax).cast(ga.subspace("x y z")).kernel for ax in axes]
    return np.stack(rotated, axis=-1)


def synthesize_scene_landmarks() -> core.Point:
    """Generate 3D world landmark points comprising a wireframe house and ground points."""
    # Base cube corners (8 points):
    cube = np.array([
        [-0.5, -0.4, 3.0],
        [ 0.5, -0.4, 3.0],
        [ 0.5,  0.4, 3.0],
        [-0.5,  0.4, 3.0],
        [-0.5, -0.4, 4.0],
        [ 0.5, -0.4, 4.0],
        [ 0.5,  0.4, 4.0],
        [-0.5,  0.4, 4.0],
    ])
    # Roof ridge and apex points (4 points):
    roof = np.array([
        [ 0.0, -0.4, 4.6],
        [ 0.0,  0.4, 4.6],
        [-0.25, 0.0, 4.3],
        [ 0.25, 0.0, 4.3],
    ])
    # Additional distributed surface / ground markers (12 points):
    grid_x, grid_y = np.meshgrid(np.linspace(-0.8, 0.8, 4), np.linspace(-0.6, 0.6, 3))
    ground = np.stack([grid_x.ravel(), grid_y.ravel(), np.full(12, 2.5)], axis=-1)

    pts_xyz = np.concatenate([cube, roof, ground], axis=0)
    return point(pts_xyz)


def epipolar_figure(plot_path: Path) -> plt.Figure:
    """Simulate two cameras viewing a 3D scene, add sensor noise, and invert the problem."""
    landmarks = synthesize_scene_landmarks()
    points_3d = coordinates(landmarks)
    n_points = len(points_3d)

    # Palette for matching keypoints across camera views and 3D reconstruction:
    colormap = plt.colormaps["turbo"].resampled(n_points)
    point_colors = [colormap(i) for i in range(n_points)]

    # Camera 1 is at the world origin:
    c1_world = mv.zyx

    # Camera 2 ground truth pose: rotated 14 degrees around y, translated along [0.65, 0.08, 0.18]:
    theta = np.radians(14.0)
    true_rot = (mv.zx * (theta * 0.5)).exp()
    true_trans = (mv.xw * 0.65 + mv.yw * 0.08 + mv.zw * 0.18) * 0.5
    true_motor = (true_trans.exp() * true_rot).normalized()
    c2_true = true_motor >> c1_world

    # Forward perspective projection to normalized sensor planes (z = 1):
    p1_true = points_3d[:, :2] / points_3d[:, 2:3]
    landmarks_cam2 = true_motor << landmarks
    pts_cam2 = coordinates(landmarks_cam2)
    p2_true = pts_cam2[:, :2] / pts_cam2[:, 2:3]

    # Realistic sensor measurement noise (sigma = 0.0015, ~1.5 pixels on a 1000px sensor):
    rng = np.random.default_rng(42)
    noise_sigma = 0.0015
    p1_noisy = p1_true + rng.normal(0.0, noise_sigma, size=p1_true.shape)
    p2_noisy = p2_true + rng.normal(0.0, noise_sigma, size=p2_true.shape)

    # Measured sight rays lifted to PGA3D Lines:
    p1_sensor_noisy = point(np.concatenate([p1_noisy, np.ones((n_points, 1))], axis=-1))
    p2_sensor_noisy = point(np.concatenate([p2_noisy, np.ones((n_points, 1))], axis=-1))
    meas_rays_1 = (c1_world & p1_sensor_noisy).normalized()
    meas_rays_2_local = (c1_world & p2_sensor_noisy).normalized()

    # --- INVERSE PROBLEM VIA EXTENSORS --------------------------------------
    # Jointly solve for relative camera pose and 3D world points from noisy sensor measurements:
    initial_motor = true_trans.exp().normalized()
    est_motor, recon_landmarks = core.reconstruct(
        meas_rays_1, meas_rays_2_local, initial_motor, 10
    )
    c2_est = est_motor >> c1_world
    points_reconstructed = coordinates(recon_landmarks)

    # Compute 1D epipolar search lines on Camera 2's sensor plane:
    screen_2 = est_motor >> (mv.z - mv.w)
    epi_planes_cam2 = est_motor << (c2_est & meas_rays_1)
    normal_cam2 = epi_planes_cam2.cast(ga.subspace("x y z")).kernel
    lines_cam2 = normal_cam2 / np.linalg.norm(normal_cam2[:, :2], axis=-1, keepdims=True)

    # Accuracy metrics:
    r2_true_mat = rotation_matrix(true_motor)
    r2_est_mat = rotation_matrix(est_motor)
    rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(r2_est_mat.T @ r2_true_mat) - 1.0) / 2.0, -1.0, 1.0)))
    c2_true_xyz = coordinates(c2_true)
    c2_est_xyz = coordinates(c2_est)
    c2_dir_err_deg = np.degrees(np.arccos(np.clip(
        np.dot(c2_est_xyz / np.linalg.norm(c2_est_xyz), c2_true_xyz / np.linalg.norm(c2_true_xyz)),
        -1.0, 1.0,
    )))
    scale = np.linalg.norm(c2_true_xyz) / np.linalg.norm(c2_est_xyz)
    recon_rmse = np.sqrt(np.mean(np.sum((points_reconstructed * scale - points_3d)**2, axis=-1)))

    panel_cam1 = (
        p1_true,
        p1_noisy,
        None,
        f"1. Camera 1 Image ({n_points} Keypoints)\nForward projection + Gaussian pixel noise",
        point_colors,
    )
    panel_cam2 = (
        p2_true,
        p2_noisy,
        lines_cam2,
        f"2. Camera 2 Image (Epipolar Constraint)\nKeypoints lie along 1D epipolar search lines",
        point_colors,
    )
    panel_3d = (
        points_3d,
        points_reconstructed,
        np.zeros(3),
        meas_rays_1.cast(ga.subspace("yz zx xy")).kernel,
        c2_true_xyz,
        r2_true_mat,
        c2_est_xyz,
        r2_est_mat,
        f"3. 3D World Reconstruction (PGA3D Motor & Line Quadrics)\nRotation err: {rot_err_deg:.2f}°, Translation err: {c2_dir_err_deg:.2f}°, RMSE: {recon_rmse:.3f}m",
        point_colors,
    )

    return render.draw_epipolar_figure(panel_cam1, panel_cam2, panel_3d, plot_path)


def main(plot_path: Path) -> plt.Figure:
    """Render the epipolar geometry and two-view reconstruction figure."""
    return epipolar_figure(plot_path)


if __name__ == "__main__":
    out_file = PLOT_DIR / "epipolar_reconstruction.png"
    main(out_file)
