"""Scenes and entry points for multi-camera bundle adjustment in PGA3D.

One function per figure. Builds the synthetic 4-camera rig viewing 3D landmarks,
hands the geometry to `core`, and hands the resulting geometry to `render`.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.geometry.multiview import core, render

ga = core.ga
ctx = core.ctx
mv = core.mv


# --- plumbing: coordinate conversion helpers -------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Extract Cartesian (x, y, z) coordinates from PGA3D Points."""
    values = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def rotation_matrix(motor: core.Motor) -> np.ndarray:
    """Extract 3x3 rotation matrix by evaluating the motor's rotor on spatial axes."""
    rotor = motor.select_subspace(ga.subspace("1 yz zx xy")).normalized()
    axes = (mv.x, mv.y, mv.z)
    rotated = [(rotor >> ax).cast(ga.subspace("x y z")).kernel for ax in axes]
    return np.stack(rotated, axis=-1)


def extract_covariances(quadrics: core.Quadric) -> np.ndarray:
    """Extract 3x3 Cartesian covariance matrices from fused precision quadrics."""
    q_mat = quadrics.kernel
    precisions = q_mat[:, :3, :3]
    return np.linalg.inv(precisions)


def multiview_figure(plot_path: Path) -> plt.Figure:
    """Build a 4-camera synthetic rig, run bundle adjustment, and render figure."""
    rng = np.random.default_rng(42)

    # 1. World landmarks in front of rig (z in [2.2, 3.8]):
    n_points = 14
    xyz = rng.uniform([-0.85, -0.75, 2.2], [0.85, 0.75, 3.8], size=(n_points, 3))
    true_points = mv.yzw * xyz[:, 0] + mv.zxw * xyz[:, 1] + mv.xyw * xyz[:, 2] + mv.zyx

    # 2. 4 cameras with convergent gaze:
    # Cam 0: origin (reference)
    # Cam 1: right (+x), panned left
    # Cam 2: left (-x), panned right
    # Cam 3: elevated (+y), tilted down
    theta = np.radians(14.0)
    m0 = mv.rotor()
    m1 = ((mv.xw * 0.65) * 0.5).exp() * ((mv.zx * theta) * 0.5).exp()
    m2 = ((-mv.xw * 0.65) * 0.5).exp() * ((-mv.zx * theta) * 0.5).exp()
    m3 = ((mv.yw * 0.50) * 0.5).exp() * ((mv.yz * theta) * 0.5).exp()
    true_motors = type(m0).stack([m0, m1, m2, m3])

    c_local = mv.zyx
    screen = mv.z - mv.w  # focal plane at z = 1
    local_cam = (c_local & core.Point) ^ screen
    cameras = local_cam.broadcast_to((4,))

    # 3. Sensor pixel measurements and sight cones in [n_points, n_cams] layout:
    local_pts = true_motors << true_points[:, None]
    projs = cameras(local_pts)
    pixels = projs / (mv.w & projs)

    # Place isotropic uncertainty disks on the sensor plane via pure GA translation:
    p0 = mv.xyw + mv.zyx  # principal point at (0, 0, 1)
    trans = (pixels / p0).square_root()
    q_sensor = mv.x * (mv.x & core.Point) + mv.y * (mv.y & core.Point)
    sensor_discs = trans >> q_sensor(trans << core.Point)
    local_cones = core.make_cones(cameras, sensor_discs)

    # 4. Initial camera pose estimates (15% angular perturbation):
    init_m1 = ((mv.xw * 0.65) * 0.5).exp() * ((mv.zx * (theta * 1.15)) * 0.5).exp()
    init_m2 = ((-mv.xw * 0.65) * 0.5).exp() * ((-mv.zx * (theta * 0.85)) * 0.5).exp()
    init_m3 = ((mv.yw * 0.50) * 0.5).exp() * ((mv.yz * (theta * 1.12)) * 0.5).exp()
    initial_motors = type(m0).stack([m0, init_m1, init_m2, init_m3])

    # 5. Run bundle adjustment and record convergence profile:
    iterations = 15
    convergence_history: list[float] = []
    xyz_true = coordinates(true_points)

    for it in range(1, iterations + 1):
        m_it, p_it, _ = core.bundle_adjust(initial_motors, local_cones, iterations=it)
        xyz_it = coordinates(p_it)
        scale = float(np.sum(xyz_it * xyz_true) / np.sum(xyz_it**2))
        rmse_it = float(np.sqrt(np.mean(np.sum((xyz_it * scale - xyz_true)**2, axis=-1))))
        convergence_history.append(rmse_it)

    # Final converged state:
    est_motors, est_points, quadrics = core.bundle_adjust(
        initial_motors, local_cones, iterations=iterations,
    )
    xyz_est = coordinates(est_points)
    scale = float(np.sum(xyz_est * xyz_true) / np.sum(xyz_est**2))
    xyz_est_scaled = xyz_est * scale
    final_rmse = np.sqrt(np.mean(np.sum((xyz_est_scaled - xyz_true)**2, axis=-1)))
    print(f"Converged Landmark RMSE: {final_rmse:.4e} m")

    # 6. Extract camera centers and orientations:
    world_cams = true_motors >> c_local
    est_world_cams = est_motors >> c_local
    cams_true_xyz = coordinates(world_cams)
    cams_est_xyz = coordinates(est_world_cams) * scale

    rots_true = [rotation_matrix(m) for m in true_motors]
    rots_est = [rotation_matrix(m) for m in est_motors]

    covariances = extract_covariances(quadrics) * (scale**2)
    cam_colors = ["#0284c7", "#ec4899", "#8b5cf6", "#f59e0b"]

    top_down_data = (cams_true_xyz, cams_est_xyz, xyz_true, xyz_est_scaled, covariances, cam_colors, rots_est)
    world_3d_data = (cams_true_xyz, rots_true, cams_est_xyz, rots_est, xyz_true, xyz_est_scaled, covariances, cam_colors)

    return render.draw_multiview_figure(top_down_data, world_3d_data, convergence_history, plot_path)


def main(plot_path: Path) -> plt.Figure:
    """Render the multi-camera bundle adjustment figure."""
    return multiview_figure(plot_path)


if __name__ == "__main__":
    out_file = PLOT_DIR / "multiview_bundle_adjustment.png"
    main(out_file)
