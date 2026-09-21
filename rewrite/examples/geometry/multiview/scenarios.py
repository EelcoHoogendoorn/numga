"""Scenes and entry points for multi-camera bundle adjustment in PGA2D.

One function per figure. Builds the synthetic 2D camera rig viewing 2D landmarks,
hands the geometry to `core`, and hands the resulting geometry to `render`.
"""

from __future__ import annotations

import sys
# Prevent local types.py from shadowing Python stdlib types if run directly as a script:
if sys.path and sys.path[0].endswith("multiview"):
    sys.path.pop(0)

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.geometry.multiview import core, render
from examples.geometry.multiview.core import sensor_disk
from examples.geometry.multiview.types import (
    Camera,
    Line,
    Motor,
    Point,
    Quadric,
    Twist,
    coordinates,
    mv,
    point,
)


def multiview_figure(plot_path: Path, auto_increment: bool = True) -> plt.Figure:
    """Build a 2-camera stereo rig in PGA2D, run bundle adjustment, and render figure."""
    # 1. 2D landmarks in front of cameras (depth y in [0.85, 2.55]):
    xy = np.array([
        [ 0.15, 0.85],
        [-0.43, 1.15],
        [ 0.50, 1.50],
        [ 0.03, 1.85],
        [ 0.65, 2.20],
        [-0.60, 2.55],
    ])
    true_points = point(xy)

    # 2. 2 convergent cameras in PGA2D:
    # Cam 0: left (-x = -0.75m), panned right (+18°)
    # Cam 1: right (+x = +0.75m), panned left (-18°)
    baseline_x = 0.75
    theta = np.radians(18.0)
    m0 = ((-mv.xw * baseline_x) * 0.5).exp() * ((mv.xy * theta) * 0.5).exp()
    m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((-mv.xy * theta) * 0.5).exp()
    true_motors = type(m0).stack([m0, m1])

    c0 = point([0.0, 0.0])
    screen = mv.y - mv.w
    camera = (c0 & Point) ^ screen
    cameras = camera.broadcast_to((2,))

    # 3. Sensor pixel measurements and sight cones in [n_points, n_cams] layout:
    local_pts = true_motors << true_points[:, None]
    projs = cameras(local_pts)
    pixels = projs / (mv.w & projs)

    # Place 1D uncertainty dyad on the sensor line:
    p0 = point([0.0, 1.0])
    q_sensor = mv.x * (mv.x & Point)
    sensor_discs = sensor_disk(pixels, p0, q_sensor)
    local_cones = core.make_cones(cameras, sensor_discs)

    # 4. Initial camera pose estimates (Cam 0 fixed as reference, Cam 1 perturbed by 5%):
    init_m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((-mv.xy * (theta * 1.05)) * 0.5).exp()
    initial_motors = type(m0).stack([m0, init_m1])

    print("=== 2-Camera Stereo Rig & Perspective Cone Bundle Adjustment (PGA2D) ===")
    print(f"Cameras  : 2 convergent viewpoints (baseline: {2*baseline_x:.2f}m X; convergent gaze: {np.degrees(theta):.1f}°)")
    print(f"Landmarks: {len(xy)} points spanning depth y in [{xy[:, 1].min():.2f}m, {xy[:, 1].max():.2f}m]")

    init_pts, _ = core.triangulate_cones(initial_motors, local_cones)
    init_xy = coordinates(init_pts)
    init_scale = float(np.sum(init_xy * xy) / np.sum(init_xy**2))
    init_rmse = float(np.sqrt(np.mean(np.sum((init_xy * init_scale - xy)**2, axis=-1))))
    print(f"Initial Landmark RMSE (unoptimized): {init_rmse:.4e} m ({init_rmse * 1000:.1f} mm)")

    # 5. Run bundle adjustment purely via perspective cone quadrics:
    iterations = 8
    est_motors, est_points, quadrics = core.bundle_adjust(
        initial_motors, local_cones, iterations=iterations, damping=0.8,
    )
    xy_true = xy
    xy_est = coordinates(est_points)
    scale = float(np.sum(xy_est * xy_true) / np.sum(xy_est**2))
    xy_est_scaled = xy_est * scale
    final_rmse = float(np.sqrt(np.mean(np.sum((xy_est_scaled - xy_true)**2, axis=-1))))
    print(f"Converged Landmark RMSE ({iterations} iters): {final_rmse:.4e} m ({final_rmse * 1000:.2f} mm)")

    # 6. Extract camera positions and directions:
    cams_pos = [coordinates(m >> c0) * scale for m in est_motors]
    cams_dirs = [(m >> mv.y).kernel[:2] for m in est_motors]
    cam_colors = ["#0284c7", "#ec4899"]

    # World cones in estimated camera frames:
    world_cones = est_motors >> local_cones(est_motors << Point)

    return render.draw_top_down_figure(
        cams_pos=cams_pos,
        cams_dirs=cams_dirs,
        world_cones=world_cones,
        fused_quadrics=quadrics,
        landmarks=est_points,
        cam_colors=cam_colors,
        motors=est_motors,
        plot_path=plot_path,
        auto_increment=auto_increment,
    )


import shutil


def convergence_animation(
    gif_path: Path | None = None,
    iterations: int = 10,
    case: str = "1cam",
    auto_increment: bool = True,
) -> Path:
    """Animate bundle adjustment convergence in a GIF (PGA2D).

    Parameters
    ----------
    gif_path : Path, optional
        Destination GIF path.
    iterations : int
        Number of bundle adjustment steps. Defaults to 10.
    case : str
        One of '1cam' (1 moving, anchors=(0, 2)), '2cams' (2 moving, anchors=(0,)),
        or '3cams' (all 3 moving, anchors=()).
    auto_increment : bool
        Whether to auto-increment the output path if it exists.
    """
    if gif_path is None:
        gif_path = PLOT_DIR / f"multiview_convergence_{case}.gif"

    xy = np.array([
        [ 0.15, 0.85],
        [-0.43, 1.15],
        [ 0.50, 1.50],
        [ 0.03, 1.85],
        [ 0.65, 2.20],
        [-0.60, 2.55],
    ])
    true_points = point(xy)

    baseline_x = 0.75
    theta = np.radians(18.0)
    m0 = ((-mv.xw * baseline_x) * 0.5).exp() * ((mv.xy * theta) * 0.5).exp()
    m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((-mv.xy * theta) * 0.5).exp()
    m2 = mv.rotor()
    true_motors = type(m0).stack([m0, m1, m2])

    c0 = point([0.0, 0.0])
    screen = mv.y - mv.w
    camera = (c0 & Point) ^ screen
    cameras = camera.broadcast_to((3,))

    local_pts = true_motors << true_points[:, None]
    projs = cameras(local_pts)
    pixels = projs / (mv.w & projs)

    p0 = point([0.0, 1.0])
    q_sensor = mv.x * (mv.x & Point)
    sensor_discs = sensor_disk(pixels, p0, q_sensor)
    local_cones = core.make_cones(cameras, sensor_discs)

    if case == "1cam":
        # Case 1: 1 moving camera (anchors=(0, 2)), Cam 1 has 25% tilt mismatch.
        # Damping=0.38 yields a steady visual trajectory reaching ~95% progress at step 9-10.
        anchors = (0, 2)
        damping = 0.38
        init_m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((-mv.xy * (theta * 1.25)) * 0.5).exp()
        motors = type(m0).stack([m0, init_m1, m2])
    elif case == "2cams":
        # Case 2: 2 moving cameras (anchors=(0,)), Cam 1 & 2 both perturbed.
        # Damping=0.35 yields a steady visual trajectory reaching ~95% progress at step 10.
        anchors = (0,)
        damping = 0.35
        init_m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((-mv.xy * (theta * 1.25)) * 0.5).exp()
        init_m2 = ((-mv.xw * 0.1) * 0.5).exp() * ((mv.xy * 0.05) * 0.5).exp()
        motors = type(m0).stack([m0, init_m1, init_m2])
    elif case == "3cams":
        # Case 3: All 3 cameras updating freely without anchors (anchors=()).
        # Damping=0.32 yields a steady visual trajectory reaching ~95% progress at step 10.
        anchors = ()
        damping = 0.32
        init_m0 = ((-mv.xw * (baseline_x * 0.95)) * 0.5).exp() * ((mv.xy * (theta * 1.05)) * 0.5).exp()
        init_m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((-mv.xy * (theta * 1.25)) * 0.5).exp()
        init_m2 = ((-mv.xw * 0.1) * 0.5).exp() * ((mv.xy * 0.05) * 0.5).exp()
        motors = type(m0).stack([init_m0, init_m1, init_m2])
    else:
        raise ValueError(f"Unknown case: {case}")

    history = []
    for it in range(iterations + 1):
        pts, qf = core.triangulate_cones(motors, local_cones)
        history.append((motors, pts, qf))
        if it < iterations:
            motors, _, _ = core.bundle_adjust(
                motors, local_cones, iterations=1, damping=damping, anchors=anchors,
            )

    target_path = render.animate_top_down_convergence(
        history=history,
        local_cones=local_cones,
        cam_colors=["#0284c7", "#ec4899", "#8b5cf6"],
        gif_path=gif_path,
        fps=3,
        auto_increment=auto_increment,
    )
    return target_path


def main(
    plot_path: Path | None = None,
    animate: bool = False,
    auto_increment: bool = True,
) -> plt.Figure:
    """Render the multi-camera bundle adjustment figure and convergence GIFs for all cases."""
    if plot_path is None:
        plot_path = PLOT_DIR / "multiview_bundle_adjustment.png"
    fig = multiview_figure(plot_path, auto_increment=auto_increment)
    if animate:
        gif1 = plot_path.with_name("multiview_convergence.gif")
        gif1_alias = plot_path.with_name("multiview_convergence_1cam.gif")
        gif2 = plot_path.with_name("multiview_convergence_2cams.gif")
        gif3 = plot_path.with_name("multiview_convergence_3cams.gif")

        convergence_animation(gif_path=gif1, iterations=10, case="1cam", auto_increment=auto_increment)
        if gif1.exists():
            shutil.copy2(gif1, gif1_alias)
        convergence_animation(gif_path=gif2, iterations=10, case="2cams", auto_increment=auto_increment)
        convergence_animation(gif_path=gif3, iterations=10, case="3cams", auto_increment=auto_increment)
    return fig


if __name__ == "__main__":
    main(animate=True)

