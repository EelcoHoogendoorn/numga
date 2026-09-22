"""Unit tests for the scenegraph, robot kinematics, and compound optics example."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

from numga import stack
from numga.algebras import PGA3D
from numga import NumpyContext

from examples.geometry.scenegraph import core, scenarios
from examples import PLOT_DIR


def test_import_isolation():
    """Verify that core.py does not import matplotlib or render."""
    core_modules = [m for m in sys.modules if m.startswith("examples.geometry.scenegraph.core")]
    assert len(core_modules) > 0

    # Ensure matplotlib was not imported by core:
    with open(Path(core.__file__), "r", encoding="utf-8") as f:
        core_source = f.read()
    assert "matplotlib" not in core_source, "core.py must not import matplotlib"
    assert "render" not in core_source, "core.py must not import render"


def test_anisotropic_scale_extensor():
    """Verify anisotropic scaling extensor scales coordinates precisely."""
    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector

    scale_extensor = core.make_anisotropic_scale(2.0, 0.5, 3.0)
    point = mv.antivector([3.0, 4.0, 5.0, 1.0])
    scaled = scale_extensor(point)

    np.testing.assert_allclose(
        scaled.kernel,
        [6.0, 2.0, 15.0, 1.0],
        atol=1e-12,
        err_msg="Anisotropic scale extensor coordinate mismatch",
    )


def test_robot_arm_kinematics():
    """Verify forward kinematics extensor batch shape and joint pivots."""
    unit_box = core.canonical_unit_box()
    assert unit_box.shape == (8,), "Unit box must have 8 vertices"

    bodies_to_world, pivots = core.make_robot_arm((0.2, -0.3, 0.5, -0.2))
    assert bodies_to_world.shape == (5,), "Must have 5 bodies"
    assert len(pivots) == 4, "Must have 4 joint pivots"

    # Verify that transforming unit_box produces valid 3D points:
    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector
    for i in range(5):
        verts = bodies_to_world[i](unit_box)
        weights = (mv.w & verts).kernel
        np.testing.assert_allclose(weights, 1.0, atol=1e-10)


def test_compound_optics_and_sensor():
    """Verify multi-lens camera projects world points onto the sensor plane."""
    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector

    camera_pose = core.make_camera_pose(position=(0.0, -4.0, 1.5), target=(0.0, 0.0, 1.0))
    camera = core.make_multi_lens_camera(camera_pose, sensor_distance=1.25)

    test_point = mv.antivector([0.1, 0.2, 1.0, 1.0])
    sensor_hit = camera(test_point)
    sensor_hit_norm = sensor_hit / (mv.w & sensor_hit)

    # sensor_hit is in camera local coordinates, lying on plane z = -1.25:
    z_coord = (mv.z & sensor_hit_norm).kernel.item()
    np.testing.assert_allclose(z_coord, -1.25, atol=1e-10, err_msg="Sensor plane distance mismatch")


def test_collapsed_pipeline_equivalence():
    """Verify single extensor collapse produces identical results to step-by-step evaluation."""
    unit_box = core.canonical_unit_box()
    bodies_to_world, _ = core.make_robot_arm()
    camera_pose = core.make_camera_pose()
    camera = core.make_multi_lens_camera(camera_pose)
    viewport = core.make_viewport()

    # Collapsed single extensor:
    local_to_pixel = core.collapse_scenegraph(bodies_to_world, camera, viewport)

    # Batched projection:
    batch_projected = core.project_vertices(local_to_pixel, unit_box)

    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector

    for body_idx in range(5):
        # Step-by-step sequential evaluation:
        body_map = bodies_to_world[body_idx]
        pts_world = body_map(unit_box)
        pts_sensor = camera(pts_world)
        pts_pixel = viewport(pts_sensor)
        pts_pixel_norm = pts_pixel / (mv.w & pts_pixel)

        np.testing.assert_allclose(
            batch_projected[body_idx].kernel,
            pts_pixel_norm.kernel,
            atol=1e-10,
            err_msg=f"Mismatch on body {body_idx}",
        )


def test_scenarios_execution(tmp_path):
    """Verify scenarios CLI executes and generates plot deliverables."""
    plot_path = tmp_path / "test_scenegraph.png"
    scene_3d_path = tmp_path / "test_scene_3d.png"
    camera_image_path = tmp_path / "test_camera_image.png"

    fig = scenarios.main(
        plot_path=plot_path,
        scene_3d_path=scene_3d_path,
        camera_image_path=camera_image_path,
        generate_animation=False,
    )
    assert fig is not None
    assert plot_path.exists(), "Combined plot must be generated"
    assert scene_3d_path.exists(), "3D scene plot must be generated"
    assert camera_image_path.exists(), "Camera image plot must be generated"
