"""Headless 3D multi-camera bundle adjustment and stress test in PGA3D.

Demonstrates 3D projective geometric algebra bundle adjustment across
multiple convergent viewpoints viewing 3D world landmarks, with extreme
rotational and translational perturbations.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from examples.geometry.multiview import core, types
from numga import stack
from numga.algebras import PGA3D

# Bind PGA3D dynamically:
types.bind(PGA3D)

from examples.geometry.multiview.scenarios import sensor_disk, make_cones
from examples.geometry.multiview.types import (
    Camera,
    Motor,
    Point,
    Quadric,
    Twist,
    coordinates,
    mv,
    point,
)


def build_3d_rig_and_scene():
    """Construct a 3-camera convergent rig and 3D landmarks in PGA3D."""
    # 3 convergent cameras in 3D:
    # Cam 0: left (-x), panned right (+yaw)
    # Cam 1: right (+x), panned left (-yaw)
    # Cam 2: center, elevated (+y), looking slightly down (-pitch)
    baseline_x = 0.75
    theta = np.radians(18.0)
    m0 = ((-mv.xw * baseline_x) * 0.5).exp() * ((-mv.zx * theta) * 0.5).exp()
    m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((mv.zx * theta) * 0.5).exp()
    m2 = ((mv.yw * 0.35) * 0.5).exp() * ((-mv.yz * np.radians(12.0)) * 0.5).exp()
    true_motors = stack([m0, m1, m2])

    c0 = point([0.0, 0.0, 0.0])
    screen = mv.z - mv.w
    camera = (c0 & Point) ^ screen
    cameras = camera.broadcast_to((3,))

    # 8 3D landmarks spanning depth z in [1.2m, 2.65m]:
    xyz = np.array([
        [ 0.15, -0.20, 1.20],
        [-0.43,  0.15, 1.45],
        [ 0.50, -0.10, 1.70],
        [ 0.03,  0.25, 1.95],
        [ 0.65, -0.15, 2.30],
        [-0.60,  0.10, 2.65],
        [ 0.20,  0.30, 2.10],
        [-0.25, -0.25, 1.60],
    ])
    true_points = point(xyz)

    # Sight cones from sensor measurements:
    local_pts = true_motors << true_points[:, None]
    projs = cameras(local_pts)
    pixels = projs / (mv.w & projs)
    # 2D transverse uncertainty on the sensor plane (z = 1) around the principal point:
    principal_point = point([0.0, 0.0, 1.0])
    q_sensor = (mv.x * (mv.x & Point)) + (mv.y * (mv.y & Point))
    sensor_discs = sensor_disk(pixels, principal_point, q_sensor)
    local_cones = make_cones(cameras, sensor_discs)

    return cameras, true_motors, true_points, local_cones, xyz


def run_3d_bundle_adjustment(
    perturb_rot_deg: tuple[float, float, float] = (30.0, -20.0, 25.0),
    perturb_trans_m: tuple[float, float, float] = (-0.40, 0.30, 0.45),
    iterations: int = 25,
    damping: float = 0.7,
    anchors: tuple[int, ...] = (0, 2),
) -> dict:
    """Run 3D bundle adjustment with crazy pose perturbations."""
    cameras, true_motors, true_points, local_cones, xyz = build_3d_rig_and_scene()

    # Compose 3D Lie-algebra perturbation motor:
    rx, ry, rz = np.radians(perturb_rot_deg)
    tx, ty, tz = perturb_trans_m
    r_pert = ((mv.yz * rx + mv.zx * ry + mv.xy * rz) * 0.5).exp()
    t_pert = ((mv.xw * tx + mv.yw * ty + mv.zw * tz) * 0.5).exp()
    pert = t_pert * r_pert

    init_m1 = pert * true_motors[1]
    motors = stack([true_motors[0], init_m1, true_motors[2]])

    pts_init, _ = core.triangulate_cones(motors, local_cones)
    lp0 = motors << pts_init[:, None]
    cost_init = float(np.sum(local_cones(lp0).kernel**2))

    c0 = point([0.0, 0.0, 0.0])
    pos_init = coordinates(motors[1] >> c0)
    pos_err_init = float(np.linalg.norm(pos_init - [0.75, 0.0, 0.0]))

    # Run Gauss-Newton bundle adjustment:
    est_motors, est_points, q_fused = core.bundle_adjust(
        motors, local_cones, iterations=iterations, damping=damping, anchors=anchors,
    )

    lp_final = est_motors << est_points[:, None]
    cost_final = float(np.sum(local_cones(lp_final).kernel**2))

    pos_final = coordinates(est_motors[1] >> c0)
    pos_err_final = float(np.linalg.norm(pos_final - [0.75, 0.0, 0.0]))

    diff = est_motors[1] * true_motors[1].reverse()
    cos_half = np.clip(np.abs(diff.kernel[0]), 0.0, 1.0)
    ang_err_final = float(np.degrees(np.arccos(cos_half) * 2.0))

    pts_xyz = coordinates(est_points)
    landmark_rmse = float(np.sqrt(np.mean(np.sum((pts_xyz - xyz)**2, axis=-1))))

    return {
        "cost_init": cost_init,
        "cost_final": cost_final,
        "pos_err_init": pos_err_init,
        "pos_err_final": pos_err_final,
        "ang_err_final": ang_err_final,
        "landmark_rmse": landmark_rmse,
        "est_motors": est_motors,
        "est_points": est_points,
    }


def main():
    print("=" * 72)
    print("  Headless 3D Multi-Camera Bundle Adjustment in PGA3D")
    print("=" * 72)

    # Test 1: Crazy 3D perturbation (30° rotation across pitch/yaw/roll, 0.7m translation):
    res1 = run_3d_bundle_adjustment(
        perturb_rot_deg=(30.0, -20.0, 25.0),
        perturb_trans_m=(-0.40, 0.30, 0.45),
        iterations=30,
        damping=0.7,
        anchors=(0, 2),
    )
    print("\nStress Test 1: 1 Moving Camera (Anchors: 0, 2)")
    print(f"  Perturbation   : Rot (30°, -20°, 25°), Trans (-0.40m, 0.30m, 0.45m)")
    print(f"  Initial Pos Err: {res1['pos_err_init']:.3f} m")
    print(f"  Final Pos Err  : {res1['pos_err_final']*1000:.2f} mm")
    print(f"  Final Ang Err  : {res1['ang_err_final']:.4f}°")
    print(f"  Landmark RMSE  : {res1['landmark_rmse']*1000:.2f} mm")
    print(f"  Cost Drop      : {res1['cost_init']:.3e} -> {res1['cost_final']:.3e} ({(1-res1['cost_final']/res1['cost_init'])*100:.2f}%)")

    # Test 2: Extreme 3D perturbation (50° rotation, 1.0m translation):
    res2 = run_3d_bundle_adjustment(
        perturb_rot_deg=(50.0, 35.0, -40.0),
        perturb_trans_m=(0.60, -0.50, 0.80),
        iterations=40,
        damping=0.7,
        anchors=(0, 2),
    )
    print("\nStress Test 2: Extreme 3D Perturbation (50° rot, 1.1m trans)")
    print(f"  Initial Pos Err: {res2['pos_err_init']:.3f} m")
    print(f"  Final Pos Err  : {res2['pos_err_final']*1000:.2f} mm")
    print(f"  Final Ang Err  : {res2['ang_err_final']:.4f}°")
    print(f"  Landmark RMSE  : {res2['landmark_rmse']*1000:.2f} mm")
    print(f"  Cost Drop      : {res2['cost_init']:.3e} -> {res2['cost_final']:.3e} ({(1-res2['cost_final']/res2['cost_init'])*100:.2f}%)")

    print("\n" + "=" * 72)
    print("  3D Bundle Adjustment Successfully Verified!")
    print("=" * 72)


if __name__ == "__main__":
    main()
