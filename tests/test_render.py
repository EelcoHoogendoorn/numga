"""The renderer draws: a ray-traced solid in a scene, a lit mesh, and markers laid over a view's own camera."""

from __future__ import annotations

import numpy as np
import pytest

from examples.util import shapes
from numga.render import drawing, meshes, Scene, simplices, solids
from numga.topology import Topology

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


def test_tube_narrows_with_its_points_widths():
    # A tube up the picture, 16 pixels wide at its foot and 4 at its head, drawn as one chain.
    mv = solids.mv
    heights = np.linspace(-0.8, 0.8, 5)
    pts = mv.point(heights[:, None] * np.array([0.0, 0.0, 1.0]))
    widths = np.linspace(16.0, 4.0, 5)
    cam_view = solids.View(solids.aim(solids.origin, elevation=0.0, azimuth=0.0), np.array([1.0, 1.0]), pixels=64)
    tube = simplices.tubes(pts, Topology.chains([5]), np.array([0.2, 0.3, 0.8]), widths, cam_view)
    scene = Scene(cam_view, np.ones(3), parts=[], meshes=[(tube, 1.0)], inks=[], translucent=[])
    img = scene.draw([], [], [], [])
    drawn = (img < 200).any(axis=-1).sum(axis=1)                                 # [rows] pixels across
    # Rows a quarter of the way down and three quarters: near the head, about 6 pixels; near the foot, about 14.
    assert drawn[16] < drawn[48] - 3


def test_units_checked_solids_meshes_and_2d_render():
    pytest.importorskip("scipy")
    from numga import NumpyContext, dim, stack
    from numga.algebras import PGA2D, PGA3D
    from numga.render import camera, contours, quadrics

    ctx3 = NumpyContext(PGA3D, units=True)
    mv3 = ctx3.multivector
    Point3 = PGA3D.gatype.antivector()
    r3 = mv3(PGA3D.gatype.scalar(unit=dim.length), 0.55)
    mass = mv3(PGA3D.gatype.scalar(unit=dim.mass), 2.0)

    pts3 = mv3.point([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.8]], unit=dim.length)
    pts_mass = mv3.point([[0.0, 0.0, 0.0], [-0.8, 0.0, 0.0]], unit=dim.mass)
    q_sphere = (
        (mv3.x * (mv3.x & Point3) + mv3.y * (mv3.y & Point3) + mv3.z * (mv3.z & Point3)) / r3**2
        - mv3.w * (mv3.w & Point3)
    )
    q_energy = q_sphere * mass
    # Direct unit-declared point differences translate and align without mixing blade units in dual().
    shifted = solids.translator(pts3[1] - pts3[0]) >> pts3[0]
    np.testing.assert_allclose((mv3.x & shifted).to_array(), 0.8, atol=1e-12)
    solids.aligned(pts3[:2], pts3[2:])

    # Mixed-unit solids, half-spaces, and inks combine in one ray-traced PGA3D scene.
    cam_view = solids.view([pts3, pts_mass], 0.6, elevation=0.3, azimuth=0.4, margin=0.1, pixels=64)
    ink_geom = solids.Ink(0, (mv3.z * (mv3.w & Point3) + mv3.w * (mv3.z & Point3))[None], solids.everywhere[None],
                          np.array([0.1, 0.1, 0.1]), 2.0, 1.0)
    ink_mass = solids.Ink(0, (mv3.x * (mv3.w & Point3) + mv3.w * (mv3.x & Point3))[None] * mass, solids.everywhere[None],
                          np.array([0.9, 0.1, 0.1]), 2.0, 1.0)
    img = solids.draw(
        [(q_sphere[None, None], np.array([0.3, 0.5, 0.8])), (q_energy[None, None], np.array([0.8, 0.4, 0.2]))],
        cam_view, np.ones(3), [ink_geom, ink_mass], [], [],
    )
    assert img.shape == cam_view.shape + (3,)
    assert (img < 240).any()

    # Smooth meshes and circle generation keep declared units through PGA3D rendering.
    topo3 = Topology(4, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
    mesh_img = meshes.draw(
        pts3, topo3.elements[2], np.array([0.2, 0.6, 0.4]), meshes.smooth(pts3, topo3),
        cam_view, np.ones(3), np.zeros((1,)), meshes.plain, gloss=0.2,
    )
    assert mesh_img.shape == cam_view.shape + (3,)
    assert (mesh_img < 240).any()
    ring = meshes.circles(pts3[1:3], stack((mv3.x ^ mv3.y, (mv3.x - mv3.w * r3) ^ mv3.y)), 8)
    assert ring.units == pts3.units and ring.shape == (2, 8)

    # 2D contours, strokes, markers, flow, and implicit quadrics map scene units into `dim.pixel` through `Camera`.
    from numga import Algebra, GATypeError
    from numga.render import flow

    ctx2 = NumpyContext(PGA2D, units=True)
    mv2 = ctx2.multivector
    Point2 = PGA2D.gatype.antivector()
    r2 = mv2.scalar(0.5, unit=dim.length)
    pts2 = mv2.point([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], unit=dim.length)
    topo2 = Topology(3, np.array([[0, 1, 2]]))
    px_per_m = mv2.scalar(32.0, unit=dim.pixel / dim.length)
    cam2 = camera.Camera(
        px_per_m * mv2.x, px_per_m * mv2.y, 0 * px_per_m * mv2.x, mv2.w, 0.0,
        np.array([0.5, 0.5]) * mv2.scalar(1.0, unit=dim.length) * px_per_m,
        np.array([1.0, 1.0]) * mv2.scalar(1.0, unit=dim.length) * px_per_m,
    )
    assert cam2.shape == (64, 64)
    origin, step_x, step_y, _ = camera.preimage(cam2, Point2)
    assert (mv2.w & origin).unit is dim.none
    assert (mv2.x & step_x).unit == dim.length / dim.pixel
    assert (mv2.y & step_y).unit == dim.length / dim.pixel
    assert (cam2.across & (origin + step_x * mv2.scalar(10.0, unit=dim.pixel))).unit == dim.pixel

    q2 = (mv2.x * (mv2.x & Point2) + mv2.y * (mv2.y & Point2)) / r2**2 - mv2.w * (mv2.w & Point2)
    comp = drawing.composited([
        contours.picture(pts2, topo2, np.array([0.0, 1.0, 2.0]), np.ones((3, 3)) * 0.5, cam2),
        simplices.strokes(pts2, topo2, np.array([0.1, 0.2, 0.3]), mv2.scalar(2.0, unit=dim.pixel), cam2),
        simplices.markers(pts2, np.array([0.8, 0.1, 0.1]), mv2.scalar(4.0, unit=dim.pixel), cam2),
        quadrics.regions(q2[None, None], np.array([0.3, 0.5, 0.7]), cam2),
        quadrics.curves(q2[None], mv2.scalar(2.0, unit=dim.pixel), np.array([0.0, 0.0, 0.0]), cam2),
        quadrics.flats((mv2.x - mv2.w * r2)[None], 2.0, np.array([0.0, 0.0, 0.0]), cam2),
    ])
    assert comp.shape == (64, 64, 4)
    assert comp[..., 3].max() > 0.5

    # Physical flow velocities (`length/time`) pair with `pixel/length` camera planes into `pixel/time`.
    vel2 = mv2.direction([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], unit=dim.length / dim.time)
    ref_speed = px_per_m * mv2.scalar(1.0, unit=dim.length / dim.time)
    flow_rgba = flow.layer(pts2, topo2, vel2, cam2, ref_speed, np.array([0.1, 0.2, 0.3]))
    assert flow_rgba.shape == (64, 64, 4)

    # Wrong-unit geometry, unscaled camera planes, wrong marker units, and wrong flow speeds are refused.
    pts_time = mv2.point([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], unit=dim.time)
    bad_cam = camera.Camera(
        mv2.x, mv2.y, 0 * mv2.x, mv2.w, 0.0,
        mv2.scalar([[16.0], [16.0]], unit=dim.pixel), mv2.scalar([[32.0], [32.0]], unit=dim.pixel),
    )
    q_bad = (mv2.x * (mv2.x & Point2) + mv2.y * (mv2.y & Point2)) / mv2.scalar(0.5, unit=dim.time)**2 - mv2.w * (mv2.w & Point2)
    for bad in (
        lambda: simplices.strokes(pts_time, topo2, np.zeros(3), 2.0, cam2),
        lambda: simplices.strokes(pts2, topo2, np.zeros(3), 2.0, bad_cam),
        lambda: simplices.markers(pts2, np.zeros(3), mv2.scalar(4.0, unit=dim.length), cam2),
        lambda: contours.picture(pts_time, topo2, np.zeros(3), np.zeros((3, 3)), cam2),
        lambda: quadrics.curves(q_bad[None], 2.0, np.zeros(3), cam2),
        lambda: flow.layer(pts2, topo2, vel2, cam2, mv2.scalar(1.0, unit=dim.length / dim.time), np.zeros(3)),
        lambda: flow.layer(pts2, topo2, vel2, cam2, 1.0, np.zeros(3)),
    ):
        with pytest.raises(GATypeError):
            bad()

    # Anisotropic phase chart: distinct physical units on x (temperature) and y (entropy) both map into `dim.pixel`.
    chart_ga = Algebra("s0t0")
    cmv = NumpyContext(chart_ga, units=True).multivector
    scale_t = cmv.scalar(10.0, unit=dim.pixel / dim.temperature)
    scale_s = cmv.scalar(20.0, unit=dim.pixel / dim.entropy)
    chart_cam = camera.Camera(
        scale_t * cmv.t, scale_s * cmv.s, 0 * scale_t * cmv.t, cmv.zeros(chart_ga.gatype.vector()), 1.0,
        mv2.scalar([[16.0], [16.0]], unit=dim.pixel), mv2.scalar([[16.0], [16.0]], unit=dim.pixel),
    )
    states = (
        np.array([1.0, 2.0, 1.5]) * cmv.scalar(1.0, unit=dim.temperature) * cmv.t.dual()
        + np.array([0.5, 0.5, 1.2]) * cmv.scalar(1.0, unit=dim.entropy) * cmv.s.dual()
    )
    chart_pic = drawing.composited([simplices.strokes(states, topo2, np.array([0.2, 0.4, 0.6]), 2.0, chart_cam)])
    assert chart_pic.shape == (32, 32, 4)
    assert chart_pic[..., 3].max() > 0.5
    _, chart_dx, chart_dy, _ = camera.preimage(chart_cam, chart_ga.gatype.antivector())
    assert (cmv.t & chart_dx).unit == dim.temperature / dim.pixel
    assert (cmv.s & chart_dy).unit == dim.entropy / dim.pixel

    # `panel` and `offset_panel` scale checked planes into `dim.pixel` from `xlim`/`ylim` bounds (same-unit and anisotropic).
    import matplotlib.pyplot as plt
    from examples.util.panels import offset_panel, panel

    fig, ax = plt.subplots(figsize=(2.0, 1.0), dpi=64)
    fig.subplots_adjust(0, 0, 1, 1)
    p_cam = panel(
        ax, mv2.x, mv2.y, mv2.w,
        xlim=np.array([-0.5, 1.5]) * mv2.scalar(1.0, unit=dim.length),
        ylim=np.array([-0.5, 1.5]) * mv2.scalar(1.0, unit=dim.length),
    )
    assert (p_cam.across & pts2).unit == dim.pixel
    assert (p_cam.up & pts2).unit == dim.pixel
    with pytest.raises(GATypeError):
        simplices.strokes(pts_time, topo2, np.zeros(3), 2.0, p_cam)

    a_cam = offset_panel(
        ax, cmv.t, cmv.s,
        xlim=np.array([0.0, 3.0]) * cmv.scalar(1.0, unit=dim.temperature),
        ylim=np.array([0.0, 2.0]) * cmv.scalar(1.0, unit=dim.entropy),
    )
    assert (a_cam.across & states).unit == dim.pixel
    assert (a_cam.up & states).unit == dim.pixel
    plt.close(fig)
