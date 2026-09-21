"""Unit tests for N-camera projective bundle adjustment with perspective cone quadrics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np

from examples.geometry.multiview import core
from examples.geometry.multiview.scenarios import (
    coordinates,
    extract_covariances,
    main,
    multiview_figure,
)


def test_mathematics_does_not_import_plotting():
    """The math layer core.py must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.geometry.multiview.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_triangulate_cones():
    """Verify perspective cone pullback and fusion reconstruct landmarks given true poses."""
    mv = core.mv
    rng = np.random.default_rng(123)

    # 8 points in front of cameras:
    xyz = rng.uniform([-0.7, -0.6, 2.5], [0.7, 0.6, 3.5], size=(8, 3))
    true_points = mv.yzw * xyz[:, 0] + mv.zxw * xyz[:, 1] + mv.xyw * xyz[:, 2] + mv.zyx

    # 3 cameras:
    theta = np.radians(12.0)
    m0 = mv.rotor()
    m1 = ((mv.xw * 0.5) * 0.5).exp() * ((mv.zx * theta) * 0.5).exp()
    m2 = ((-mv.xw * 0.5) * 0.5).exp() * ((-mv.zx * theta) * 0.5).exp()
    motors = type(m0).stack([m0, m1, m2])

    c_local = mv.zyx
    screen = mv.z - mv.w
    cam_single = (c_local & core.Point) ^ screen
    cameras = cam_single.broadcast_to((3,))

    local_pts = motors << true_points[:, None]
    projs = cameras(local_pts)
    pixels = projs / (mv.w & projs)

    # Sensor discs in pure GA:
    p0 = mv.xyw + mv.zyx
    trans = (pixels / p0).square_root()
    q_sensor = mv.x * (mv.x & core.Point) + mv.y * (mv.y & core.Point)
    sensor_discs = trans >> q_sensor(trans << core.Point)

    # Pullback cones through camera maps:
    local_cones = core.make_cones(cameras, sensor_discs)
    pts_cone, q_cone = core.triangulate_cones(motors, local_cones)
    xyz_cone = coordinates(pts_cone)
    np.testing.assert_allclose(xyz_cone, xyz, atol=1e-5)

    # Covariances extracted from cone quadrics should be positive-definite:
    covs = extract_covariances(q_cone)
    for cov in covs:
        evals = np.linalg.eigvalsh(cov)
        assert np.all(evals > 0), "Gaussian splat covariance must be positive-definite"


def test_multiview_bundle_adjust_convergence():
    """Verify Gauss-Newton bundle adjustment converges from perturbed camera poses."""
    mv = core.mv
    rng = np.random.default_rng(42)

    xyz = rng.uniform([-0.8, -0.7, 2.2], [0.8, 0.7, 3.8], size=(12, 3))
    true_points = mv.yzw * xyz[:, 0] + mv.zxw * xyz[:, 1] + mv.xyw * xyz[:, 2] + mv.zyx

    theta = np.radians(14.0)
    m0 = mv.rotor()
    m1 = ((mv.xw * 0.6) * 0.5).exp() * ((mv.zx * theta) * 0.5).exp()
    m2 = ((-mv.xw * 0.6) * 0.5).exp() * ((-mv.zx * theta) * 0.5).exp()
    m3 = ((mv.yw * 0.4) * 0.5).exp() * ((mv.yz * theta) * 0.5).exp()
    true_motors = type(m0).stack([m0, m1, m2, m3])

    c_local = mv.zyx
    screen = mv.z - mv.w
    cam_single = (c_local & core.Point) ^ screen
    cameras = cam_single.broadcast_to((4,))

    local_pts = true_motors << true_points[:, None]
    projs = cameras(local_pts)
    pixels = projs / (mv.w & projs)

    p0 = mv.xyw + mv.zyx
    trans = (pixels / p0).square_root()
    q_sensor = mv.x * (mv.x & core.Point) + mv.y * (mv.y & core.Point)
    sensor_discs = trans >> q_sensor(trans << core.Point)
    local_cones = core.make_cones(cameras, sensor_discs)

    # Perturb camera orientations by 15%:
    init_m1 = ((mv.xw * 0.6) * 0.5).exp() * ((mv.zx * (theta * 1.15)) * 0.5).exp()
    init_m2 = ((-mv.xw * 0.6) * 0.5).exp() * ((-mv.zx * (theta * 0.85)) * 0.5).exp()
    init_m3 = ((mv.yw * 0.4) * 0.5).exp() * ((mv.yz * (theta * 1.12)) * 0.5).exp()
    initial_motors = type(m0).stack([m0, init_m1, init_m2, init_m3])

    # Pre-optimization error:
    init_pts, _ = core.triangulate_cones(initial_motors, local_cones)
    init_xyz = coordinates(init_pts)
    init_scale = np.sum(init_xyz * xyz) / np.sum(init_xyz**2)
    init_rmse = np.sqrt(np.mean(np.sum((init_xyz * init_scale - xyz)**2, axis=-1)))

    # Post-optimization (pure cone bundle adjustment):
    est_motors, est_pts, _ = core.bundle_adjust(initial_motors, local_cones, iterations=15)
    est_xyz = coordinates(est_pts)
    est_scale = np.sum(est_xyz * xyz) / np.sum(est_xyz**2)
    final_rmse = np.sqrt(np.mean(np.sum((est_xyz * est_scale - xyz)**2, axis=-1)))

    assert final_rmse < init_rmse, "Bundle adjustment must decrease landmark reconstruction error"
    assert final_rmse < 0.02, f"Final RMSE should be under 2 cm, got {final_rmse}"


def test_scenario_runs_and_saves(tmp_path: Path):
    """The scenario wires math to render and writes its figure."""
    import matplotlib
    matplotlib.use("Agg")

    out = tmp_path / "multiview.png"
    multiview_figure(plot_path=out)
    assert out.exists()
    assert out.stat().st_size > 1000

    main(plot_path=tmp_path / "again.png")
    assert (tmp_path / "again.png").exists()
