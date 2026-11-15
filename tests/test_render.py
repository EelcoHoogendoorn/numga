"""The renderer draws: a ray-traced solid in a scene, a lit mesh, and markers laid over a view's own camera."""

from __future__ import annotations

import numpy as np
import pytest

from examples.util import shapes
from numga.render import drawing, meshes, Scene, simplices, solids

pytest.importorskip("wgpu")


def test_view_to_camera_and_solids_scene_draw():
    mv = solids.mv
    ball = solids.ball(solids.origin[None], mv.scalar([[0.8]]))[:, None]
    cam_view = solids.View(solids.aim(solids.origin, elevation=0.3, azimuth=0.4), np.array([1.2, 1.2]), pixels=64)
    scene = Scene(cam_view, np.ones(3), parts=[(ball, np.array([0.8, 0.3, 0.2]))], meshes=[], inks=[], translucent=[])
    img = scene.draw([], [], [], [])
    assert img.shape == (64, 64, 3)
    assert img[32, 32, 1] < 240
    assert np.all(img[0, 0] == 255)


def test_meshes_and_simplices_without_matplotlib():
    pytest.importorskip("scipy")
    topo, coords = shapes.sphere(1)
    pts = solids.mv.point(0.7 * coords)
    cam_view = solids.View(solids.aim(solids.origin, elevation=0.25, azimuth=0.5), np.array([1.2, 1.2]), pixels=64)
    img = meshes.draw(
        pts, topo.elements[2], np.array([0.2, 0.55, 0.85]), meshes.smooth(pts, topo),
        cam_view, np.ones(3), np.zeros((1,)), meshes.plain, gloss=0.2,
    )
    assert img.shape == (64, 64, 3)
    assert img[32, 32, 0] < 240

    cam2d = cam_view.to_camera()
    dots = simplices.markers(solids.origin[None], np.array([[0.1, 0.2, 0.8]]), np.array([6.0]), cam2d)
    comp = drawing.composited([dots])
    assert comp.shape == (64, 64, 4)
    assert comp[32, 32, 3] > 0.5
