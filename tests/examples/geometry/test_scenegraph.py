"""Unit tests for the scenegraph, robot kinematics, and compound optics example."""

from __future__ import annotations

import subprocess
import sys
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.scenegraph.core import point
from examples.geometry.scenegraph import core, render, scenarios
from examples.geometry.scenegraph.render import euclidean

mv = core.mv


def test_mathematics_does_not_import_plotting():
    """The math layer core.py must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.geometry.scenegraph.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_anisotropic_scale_extensor():
    """The anisotropic scaling extensor scales each coordinate and keeps the weight."""
    scaled = core.anisotropic_scale(2.0, 0.5, 3.0)(point(np.array([3.0, 4.0, 5.0])))
    np.testing.assert_allclose(euclidean(scaled), [6.0, 2.0, 15.0], atol=1e-12)
    np.testing.assert_allclose((mv.w & scaled).to_array(), 1.0, atol=1e-12)


def test_robot_arm_kinematics():
    """Forward kinematics yields five rigidly placed bodies and four joint pivots."""
    unit_box = core.canonical_unit_box()
    bodies_to_world, pivots = core.robot_arm((0.2, -0.3, 0.5, -0.2))
    assert bodies_to_world.shape == (5,)
    assert len(pivots) == 4
    verts = bodies_to_world[:, None](unit_box[None, :])
    np.testing.assert_allclose((mv.w & verts).to_array(), 1.0, atol=1e-10)


def test_compound_optics_lands_on_the_sensor():
    """The multi-lens camera sends world points onto the sensor plane z = -1.25."""
    pose = core.look_at(np.array([0.0, -4.0, 1.5]), np.array([0.0, 0.0, 1.0]))
    front_lens, rear_lens, _ = core.lens_train(1.0, 0.8, -0.3)
    pupil = (mv.xw * 0.02).exp() >> core.origin
    camera = core.lens_camera(pose, front_lens, rear_lens, pupil, mv.z + 1.25 * mv.w)
    sensor_hit = camera(point(np.array([0.1, 0.2, 1.0])))
    np.testing.assert_allclose(euclidean(sensor_hit)[2], -1.25, atol=1e-10)


def test_scenario_renders():
    """The scenario passes its checks; its figures and a short animation render."""
    world_vertices, projected_pixels, camera_pose, rays = scenarios.scenegraph()
    figures = [
        render.draw_scenegraph(world_vertices, projected_pixels, camera_pose, rays),
        render.draw_scene_3d(world_vertices, camera_pose, rays),
        render.draw_camera_image(projected_pixels),
    ]
    assert all(isinstance(figure, plt.Figure) for figure in figures)
    frames = render.animate_scenegraph(islice(scenarios.robot_sweep(36), 4))
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
