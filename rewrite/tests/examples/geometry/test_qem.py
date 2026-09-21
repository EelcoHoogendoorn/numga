"""Unit tests for Garland–Heckbert QEM in PGA3D."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np

from numga import Extensor
from examples.geometry.qem import core
from examples.geometry.qem.core import Plane, Point, Quadric
from examples.geometry.qem.render import euclidean
from examples.geometry.qem.scenarios import main, plane, point, qem_figure


def test_mathematics_does_not_import_plotting():
    """The math layer core.py must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.geometry.qem.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_plane_dyad_evaluates_to_squared_perpendicular_distance():
    """A rank-1 plane dyad P * (P & Point) evaluates to squared distance (P & X)^2."""
    p_on_plane = np.array([1.0, 0.5, -0.5])
    normal = np.array([0.0, 1.0, 0.0])
    p = plane(normal, p_on_plane)

    # Build dyad
    q = p * (p & Point)
    assert q.gatype == Quadric

    # Test points at distance +2, -2, and 0
    q1 = point(np.array([1.0, 2.5, -0.5]))
    q2 = point(np.array([0.0, -1.5, 3.0]))
    q_on = point(p_on_plane)

    np.testing.assert_allclose((q(q1) & q1).to_array(), 4.0, atol=1e-12)
    np.testing.assert_allclose((q(q2) & q2).to_array(), 4.0, atol=1e-12)
    np.testing.assert_allclose((q(q_on) & q_on).to_array(), 0.0, atol=1e-12)


def test_edge_collapse_minimizes_joint_error():
    """Edge collapse combines incident quadrics and finds the joint minimum."""
    coords = np.array([
        [ 0.35,  0.0,   0.25],  # 0: a
        [-0.35,  0.0,   0.25],  # 1: b
        [ 0.0,   0.55, -0.15],  # 2: left base
        [ 0.0,  -0.55, -0.15],  # 3: right base
        [ 0.75,  0.0,  -0.15],  # 4: front tip
        [-0.85,  0.0,  -0.15],  # 5: back tip
    ])
    verts = point(coords)

    faces = np.array([
        [0, 1, 2],  # 0: left flank (shared)
        [1, 0, 3],  # 1: right flank (shared)
        [0, 2, 4],  # 2: corner front-left
        [0, 4, 3],  # 3: corner front-right
        [1, 5, 2],  # 4: ramp back-left
        [1, 3, 5],  # 5: ramp back-right
    ])

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_planes = (v0 & v1 & v2).normalized()

    # Flanking planes pass through both edge endpoints a and b:
    flank_planes = face_planes[[0, 1]]
    pa = verts[0]
    pb = verts[1]
    np.testing.assert_allclose((flank_planes & pa).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((flank_planes & pb).to_array(), 0.0, atol=1e-12)

    planes_a = face_planes[[0, 1, 2, 3]]
    planes_b = face_planes[[0, 1, 4, 5]]

    qa, qb, q_edge, v_edge = core.edge_collapse(planes_a, planes_b)

    # Edge quadric is the extensor sum qa + qb
    test_pt = point(np.array([3.0, 4.0, 5.0]))
    ea = (qa(test_pt) & test_pt).to_array()
    eb = (qb(test_pt) & test_pt).to_array()
    e_edge = (q_edge(test_pt) & test_pt).to_array()
    np.testing.assert_allclose(e_edge, ea + eb, atol=1e-12)

    # Optimal collapse point preserves the ridge line (y = 0) and is biased toward corner a
    v_opt_xyz = euclidean(v_edge)
    np.testing.assert_allclose(v_opt_xyz[1], 0.0, atol=1e-12)
    assert 0.0 < v_opt_xyz[0] < coords[0, 0]

    # Joint error at v_edge is lower than at either endpoint
    e_opt = (q_edge(v_edge) & v_edge).to_array()
    e_at_a = (q_edge(pa) & pa).to_array()
    e_at_b = (q_edge(pb) & pb).to_array()
    assert e_opt < e_at_a
    assert e_opt < e_at_b


def test_scenario_runs_and_saves(tmp_path: Path):
    """The scenario wires math to render and writes its figure."""
    import matplotlib
    matplotlib.use("Agg")

    out = tmp_path / "qem.png"
    qem_figure(plot_path=out)
    assert out.exists()
    assert out.stat().st_size > 1000

    main(plot_path=tmp_path / "again.png")
    assert (tmp_path / "again.png").exists()
