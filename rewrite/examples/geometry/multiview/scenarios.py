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


def multiview_figure(plot_path: Path, auto_increment: bool = True) -> plt.Figure:
    """Build a 4-camera synthetic rig, run bundle adjustment, and render figure."""
    rng = np.random.default_rng(42)

    # 1. World landmarks structured symmetrically across depths z in [1.45, 3.90] m:
    # Clearly illustrates ray cone widening and depth elongation growth from near to far camera ray intersections.
    xyz = np.array([
        [-0.30, -0.15, 1.45],
        [ 0.30,  0.15, 1.45],
        [ 0.00, -0.10, 2.20],
        [-0.38,  0.20, 3.10],
        [ 0.38, -0.20, 3.10],
        [ 0.00,  0.10, 3.90],
    ])
    true_points = mv.yzw * xyz[:, 0] + mv.zxw * xyz[:, 1] + mv.xyw * xyz[:, 2] + mv.zyx

    # 2. 2 cameras in stereo configuration with convergent gaze:
    # Cam 0: left (-x = -0.55m), panned right (+13°)
    # Cam 1: right (+x = +0.55m), panned left (-13°)
    theta = np.radians(13.0)
    m0 = ((-mv.xw * 0.55) * 0.5).exp() * ((-mv.zx * theta) * 0.5).exp()
    m1 = ((mv.xw * 0.55) * 0.5).exp() * ((mv.zx * theta) * 0.5).exp()
    true_motors = type(m0).stack([m0, m1])

    c_local = mv.zyx
    screen = mv.z - mv.w  # focal plane at z = 1
    local_cam = (c_local & core.Point) ^ screen
    cameras = local_cam.broadcast_to((2,))

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

    # 4. Initial camera pose estimates (Cam 0 fixed as reference, Cam 1 perturbed by 6%):
    init_m1 = ((mv.xw * 0.55) * 0.5).exp() * ((mv.zx * (theta * 1.06)) * 0.5).exp()
    initial_motors = type(m0).stack([m0, init_m1])

    print("=== 2-Camera Stereo Rig & Perspective Cone Bundle Adjustment ===")
    print(f"Cameras  : 2 convergent viewpoints (baseline: 1.10m X; convergent gaze: 13.0°)")
    print(f"Landmarks: {len(xyz)} points spanning depth z in [{xyz[:, 2].min():.2f}m, {xyz[:, 2].max():.2f}m]")

    # Measure initial error before optimization:
    init_pts, _ = core.triangulate_cones(initial_motors, local_cones)
    init_xyz = coordinates(init_pts)
    init_scale = float(np.sum(init_xyz * xyz) / np.sum(init_xyz**2))
    init_rmse = float(np.sqrt(np.mean(np.sum((init_xyz * init_scale - xyz)**2, axis=-1))))
    print(f"Initial Landmark RMSE (unoptimized): {init_rmse:.4e} m ({init_rmse * 1000:.1f} mm)")

    # 5. Run bundle adjustment purely via perspective cone quadrics:
    iterations = 12
    est_motors, est_points, quadrics = core.bundle_adjust(
        initial_motors, local_cones, iterations=iterations,
    )
    xyz_true = coordinates(true_points)
    xyz_est = coordinates(est_points)
    scale = float(np.sum(xyz_est * xyz_true) / np.sum(xyz_est**2))
    xyz_est_scaled = xyz_est * scale
    final_rmse = np.sqrt(np.mean(np.sum((xyz_est_scaled - xyz_true)**2, axis=-1)))
    print(f"Converged Landmark RMSE ({iterations} iters): {final_rmse:.4e} m ({final_rmse * 1000:.2f} mm)")

    # 6. Extract camera centers and orientations:
    world_cams = true_motors >> c_local
    est_world_cams = est_motors >> c_local
    cams_true_xyz = coordinates(world_cams)
    cams_est_xyz = coordinates(est_world_cams) * scale

    rots_true = [rotation_matrix(m) for m in true_motors]
    rots_est = [rotation_matrix(m) for m in est_motors]

    covariances = extract_covariances(quadrics) * (scale**2)
    cam_colors = ["#0284c7", "#ec4899"]

    print("Gaussian Splat Anisotropy Across Depth (X–Z plane):")
    for i, (pt, cov) in enumerate(zip(xyz_est_scaled, covariances)):
        cov_xz = np.array([[cov[0, 0], cov[0, 2]], [cov[2, 0], cov[2, 2]]])
        evals = np.linalg.eigvalsh(cov_xz)
        radii = np.sqrt(np.maximum(evals, 1e-8))
        ratio = radii[1] / radii[0]
        print(f"  Landmark {i} (x={pt[0]:+.2f}m, z={pt[2]:.2f}m): minor={radii[0]:.3f}m, major={radii[1]:.3f}m, ratio={ratio:.2f}")

    # World cones in estimated camera frames:
    world_cones = est_motors >> local_cones(est_motors << core.Point)

    top_down_data = (
        cams_true_xyz, cams_est_xyz, xyz_true, xyz_est_scaled,
        covariances, cam_colors, rots_est,
        world_cones.kernel, quadrics.kernel,
    )
    world_3d_data = (cams_true_xyz, rots_true, cams_est_xyz, rots_est, xyz_true, xyz_est_scaled, covariances, cam_colors)

    if plot_path is not None:
        stem = plot_path.stem
        plot_path_3d = plot_path.with_name(f"{stem}_3d{plot_path.suffix}")
        fig_2d = render.draw_top_down_figure(
            top_down_data, plot_path=plot_path, auto_increment=auto_increment,
        )
        render.draw_3d_figure(
            world_3d_data, plot_path=plot_path_3d, auto_increment=auto_increment,
        )
        return fig_2d
    return render.draw_top_down_figure(top_down_data)


def main(plot_path: Path | None = None, auto_increment: bool = True) -> plt.Figure:
    """Render the multi-camera bundle adjustment figures (separate 2D and 3D plots)."""
    if plot_path is None:
        plot_path = PLOT_DIR / "multiview_bundle_adjustment.png"
    return multiview_figure(plot_path, auto_increment=auto_increment)


if __name__ == "__main__":
    main()
