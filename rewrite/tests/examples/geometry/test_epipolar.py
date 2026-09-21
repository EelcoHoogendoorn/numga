"""Unit tests for epipolar geometry and two-view 3D reconstruction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np

from examples.geometry.epipolar import core
from examples.geometry.epipolar.scenarios import (
    coordinates,
    epipolar_figure,
    main,
    point,
    rotation_matrix,
    synthesize_scene_landmarks,
)


def test_mathematics_does_not_import_plotting():
    """The math layer core.py must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.geometry.epipolar.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_reconstruct_from_screen_space_points():
    """Reconstruct relative camera pose and 3D world points from 2D screen points."""
    landmarks = synthesize_scene_landmarks()
    true_xyz = coordinates(landmarks)
    c1_world = core.mv.zyx

    theta = np.radians(14.0)
    true_rot = (core.mv.zx * (theta * 0.5)).exp()
    true_trans = (core.mv.xw * 0.65 + core.mv.yw * 0.08 + core.mv.zw * 0.18) * 0.5
    true_motor = (true_trans.exp() * true_rot).normalized()

    # Forward projections onto local camera screens (z = 1):
    p1_true = true_xyz[:, :2] / true_xyz[:, 2:3]
    landmarks_cam2 = true_motor << landmarks
    cam2_xyz = coordinates(landmarks_cam2)
    p2_true = cam2_xyz[:, :2] / cam2_xyz[:, 2:3]

    screen1 = point(np.concatenate([p1_true, np.ones((len(p1_true), 1))], axis=-1))
    screen2 = point(np.concatenate([p2_true, np.ones((len(p2_true), 1))], axis=-1))
    rays1 = (c1_world & screen1).normalized()
    rays2 = (c1_world & screen2).normalized()

    initial_motor = true_trans.exp().normalized()
    est_motor, recon_world = core.reconstruct(rays1, rays2, initial_motor, 10)

    # Check recovered camera 2 rotation:
    r_est = rotation_matrix(est_motor)
    r_true = rotation_matrix(true_motor)
    rot_err = np.degrees(np.arccos(np.clip((np.trace(r_est.T @ r_true) - 1.0) / 2.0, -1.0, 1.0)))
    assert rot_err < 0.05

    # Check recovered camera 2 translation direction:
    c2_true_xyz = np.array([0.65, 0.08, 0.18])
    c2_est_xyz = coordinates(est_motor >> c1_world)
    c2_dir_err = np.degrees(np.arccos(np.clip(
        np.dot(c2_est_xyz / np.linalg.norm(c2_est_xyz), c2_true_xyz / np.linalg.norm(c2_true_xyz)),
        -1.0, 1.0,
    )))
    assert c2_dir_err < 0.05

    # Check reconstructed 3D points up to baseline scale:
    scale = np.linalg.norm(c2_true_xyz) / np.linalg.norm(c2_est_xyz)
    np.testing.assert_allclose(coordinates(recon_world) * scale, true_xyz, atol=1e-2)


def test_scenario_runs_and_saves(tmp_path: Path):
    """The scenario wires math to render and writes its figure."""
    import matplotlib
    matplotlib.use("Agg")

    out = tmp_path / "epipolar.png"
    epipolar_figure(plot_path=out)
    assert out.exists()
    assert out.stat().st_size > 1000

    main(plot_path=tmp_path / "again.png")
    assert (tmp_path / "again.png").exists()
