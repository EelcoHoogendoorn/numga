"""Unit tests for N-camera projective bundle adjustment with perspective cone quadrics in PGA2D."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pytest

from numga import stack
from examples import PLOT_DIR
from examples.geometry.multiview import core, render
from examples.geometry.multiview.scenarios import (
    Point,
    coordinates,
    main,
    make_cones,
    multiview_figure,
    mv,
    point,
    sensor_disk,
)


@pytest.fixture(autouse=True, scope="module")
def enforce_regenerate_multiview_plots():
    """Ensure running multiview tests automatically regenerates and validates all module plots."""
    t_start = time.time()
    fig_path = PLOT_DIR / "multiview_bundle_adjustment.png"
    gif_paths = [
        PLOT_DIR / "multiview_convergence.gif",
        PLOT_DIR / "multiview_convergence_1cam.gif",
        PLOT_DIR / "multiview_convergence_2cams.gif",
        PLOT_DIR / "multiview_convergence_3cams.gif",
    ]

    # Regenerate canonical figure and all convergence animations:
    main(plot_path=fig_path, animate=True, auto_increment=False)

    # Verification: Both figure and GIFs must exist, have valid size, and fresh timestamp
    assert fig_path.exists(), f"Figure was not created: {fig_path}"
    assert fig_path.stat().st_size > 1000, f"Figure is empty: {fig_path}"
    assert fig_path.stat().st_mtime >= t_start - 2.0, f"Figure was not regenerated: {fig_path}"

    for gp in gif_paths:
        assert gp.exists(), f"Convergence GIF was not created: {gp}"
        assert gp.stat().st_size > 1000, f"GIF is empty: {gp}"
        assert gp.stat().st_mtime >= t_start - 2.0, f"GIF was not regenerated: {gp}"



def test_triangulate_cones():
    """Verify perspective cone pullback and fusion reconstruct landmarks given true poses."""
    rng = np.random.default_rng(123)

    # 6 points in front of cameras:
    xy = rng.uniform([-0.6, 1.0], [0.6, 2.5], size=(6, 2))
    true_points = point(xy)

    # 2 cameras:
    baseline_x = 0.75
    theta = np.radians(18.0)
    m0 = ((-mv.xw * baseline_x) * 0.5).exp() * ((mv.xy * theta) * 0.5).exp()
    m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((-mv.xy * theta) * 0.5).exp()
    motors = type(m0).stack([m0, m1])

    c0 = point([0.0, 0.0])
    screen = mv.y - mv.w
    camera = (c0 & Point) ^ screen
    cameras = camera.broadcast_to((2,))

    local_pts = motors << true_points[:, None]
    projs = cameras(local_pts)
    pixels = projs / (mv.w & projs)

    # Pullback cones through camera maps:
    principal_point = point([0.0, 1.0])
    q_sensor = mv.x * (mv.x & Point)
    sensor_discs = sensor_disk(pixels, principal_point, q_sensor)
    local_cones = make_cones(cameras, sensor_discs)
    pts_cone, q_cone = core.triangulate_cones(motors, local_cones)
    xy_cone = coordinates(pts_cone)
    np.testing.assert_allclose(xy_cone, xy, atol=1e-5)

    # Covariances extracted from cone quadrics should be positive-definite:
    covs = render.extract_covariances(q_cone)
    for cov in covs:
        evals = np.linalg.eigvalsh(cov)
        assert np.all(evals > 0), "Gaussian splat covariance must be positive-definite"


def test_multiview_bundle_adjust_convergence():
    """Verify Gauss-Newton bundle adjustment converges from perturbed camera poses."""
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
    m2 = ((mv.xw * 0.0) * 0.5).exp()
    true_motors = stack([m0, m1, m2])

    c0 = point([0.0, 0.0])
    screen = mv.y - mv.w
    camera = (c0 & Point) ^ screen
    cameras = camera.broadcast_to((3,))

    local_pts = true_motors << true_points[:, None]
    projs = cameras(local_pts)
    pixels = projs / (mv.w & projs)

    principal_point = point([0.0, 1.0])
    q_sensor = mv.x * (mv.x & Point)
    sensor_discs = sensor_disk(pixels, principal_point, q_sensor)
    local_cones = make_cones(cameras, sensor_discs)

    # Perturb camera 1 orientation by 5%:
    init_m1 = ((mv.xw * baseline_x) * 0.5).exp() * ((-mv.xy * (theta * 1.05)) * 0.5).exp()
    initial_motors = stack([m0, init_m1, m2])

    init_pts, _ = core.triangulate_cones(initial_motors, local_cones)
    init_xy = coordinates(init_pts)
    init_scale = np.sum(init_xy * xy) / np.sum(init_xy**2)
    init_rmse = np.sqrt(np.mean(np.sum((init_xy * init_scale - xy)**2, axis=-1)))

    est_motors, est_pts, _ = core.bundle_adjust(initial_motors, local_cones, iterations=12, anchors=(0, 2))
    est_xy = coordinates(est_pts)
    est_scale = np.sum(est_xy * xy) / np.sum(est_xy**2)
    final_rmse = np.sqrt(np.mean(np.sum((est_xy * est_scale - xy)**2, axis=-1)))

    assert final_rmse < init_rmse, "Bundle adjustment must decrease landmark reconstruction error"
    assert final_rmse < 0.02, f"Final RMSE should be under 2 cm, got {final_rmse}"


def test_scenario_runs_and_saves():
    """The scenario wires math to render and ensures canonical figure and animation are written."""
    fig_path = PLOT_DIR / "multiview_bundle_adjustment.png"
    gif_path = PLOT_DIR / "multiview_convergence.gif"
    assert fig_path.exists() and fig_path.stat().st_size > 1000, f"Missing canonical figure {fig_path}"
    assert gif_path.exists() and gif_path.stat().st_size > 1000, f"Missing canonical GIF {gif_path}"


def test_multiview_3d_bundle_adjust_convergence():
    """Verify Gauss-Newton bundle adjustment works identically in 3D under PGA3D."""
    from examples.geometry.multiview import types
    from numga.algebras import PGA3D, PGA2D

    types.bind(PGA3D)
    try:
        baseline_x = 0.75
        theta = np.radians(18.0)
        m0 = ((-types.mv.xw * baseline_x) * 0.5).exp() * ((-types.mv.zx * theta) * 0.5).exp()
        m1 = ((types.mv.xw * baseline_x) * 0.5).exp() * ((types.mv.zx * theta) * 0.5).exp()
        true_motors = type(m0).stack([m0, m1])

        c0 = types.point([0.0, 0.0, 0.0])
        screen = types.mv.z - types.mv.w
        cameras = ((c0 & types.Point) ^ screen).broadcast_to((2,))

        xyz = np.array([
            [ 0.15, -0.06, 0.85],
            [-0.43,  0.07, 1.15],
            [ 0.50, -0.08, 1.50],
            [ 0.03,  0.06, 1.85],
            [ 0.65, -0.07, 2.20],
            [-0.60,  0.09, 2.55],
        ])
        true_points = types.point(xyz)

        local_pts = true_motors << true_points[:, None]
        projs = cameras(local_pts)
        pixels = projs / (types.mv.w & projs)

        principal_point = types.point([0.0, 0.0, 1.0])
        q_sensor = (types.mv.x * (types.mv.x & types.Point)) + (types.mv.y * (types.mv.y & types.Point))
        sensor_discs = sensor_disk(pixels, principal_point, q_sensor)
        local_cones = make_cones(cameras, sensor_discs)

        init_m1 = ((types.mv.xw * baseline_x) * 0.5).exp() * ((types.mv.zx * (theta * 1.05)) * 0.5).exp()
        motors = stack([m0, init_m1])

        init_pts, _ = core.triangulate_cones(motors, local_cones)
        init_xyz = types.coordinates(init_pts)
        init_scale = np.sum(init_xyz * xyz) / np.sum(init_xyz**2)
        init_rmse = np.sqrt(np.mean(np.sum((init_xyz * init_scale - xyz)**2, axis=-1)))

        est_motors, est_pts, _ = core.bundle_adjust(motors, local_cones, iterations=10)
        est_xyz = types.coordinates(est_pts)
        est_scale = np.sum(est_xyz * xyz) / np.sum(est_xyz**2)
        final_rmse = np.sqrt(np.mean(np.sum((est_xyz * est_scale - xyz)**2, axis=-1)))

        assert final_rmse < init_rmse
        assert final_rmse < 0.02
    finally:
        types.bind(PGA2D)


def test_multiview_3d_crazy_pose_stress():
    """Verify headless 3D multi-camera bundle adjustment converges under extreme perturbations."""
    from examples.geometry.multiview import scenarios_3d, types
    from numga.algebras import PGA2D, PGA3D

    types.bind(PGA3D)
    try:
        res = scenarios_3d.run_3d_bundle_adjustment(
            perturb_rot_deg=(30.0, -20.0, 25.0),
            perturb_trans_m=(-0.40, 0.30, 0.45),
            iterations=30,
            damping=0.7,
            anchors=(0, 2),
        )
        assert res["pos_err_final"] < 0.02, f"3D position error too high: {res['pos_err_final']}"
        assert res["ang_err_final"] < 0.5, f"3D angular error too high: {res['ang_err_final']}"
        assert res["landmark_rmse"] < 0.02, f"3D landmark RMSE too high: {res['landmark_rmse']}"
        assert res["cost_final"] < res["cost_init"] * 1e-3, "Residual did not drop"
    finally:
        types.bind(PGA2D)

