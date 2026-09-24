"""Unit tests for projective cameras in PGA3D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.projection import core, render, scenarios
from examples.geometry.projection.core import Camera, Correspondence, Line, Plane, Point, cube, direction, ga, point
from examples.geometry.projection.render import screen_coordinates
from examples.geometry.projection.scenarios import mv, origin


def dehomogenize(p: Point) -> np.ndarray:
    k = p.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return k[..., :3] / k[..., 3:]


SCREEN = mv.z - mv.w


def test_pinhole_image_is_perspective_division():
    """A pinhole at the origin with screen z = 1 images (x, y, z) to (x/z, y/z, 1)."""
    camera = origin.regressive(Point).wedge(SCREEN)
    assert camera.gatype == Camera
    world = point(np.array([[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0], [0.2, -0.4, 0.5]]))
    np.testing.assert_allclose(dehomogenize(camera(world)), [[0.25, 0.5, 1.0], [-1.5, 0.25, 1.0], [0.4, -0.8, 1.0]], atol=1e-14)


def test_camera_is_a_rank_three_projection():
    """The centre has no image, screen points are fixed, and the map is projectively idempotent."""
    centre = point(np.array([0.3, -0.2, -1.0]))
    camera = centre.regressive(Point).wedge(SCREEN)
    assert np.linalg.matrix_rank(camera.kernel) == 3
    np.testing.assert_allclose(camera(centre).kernel, 0.0, atol=1e-14)

    on_screen = point(np.array([[0.7, 0.1, 1.0], [-2.0, 3.0, 1.0]]))
    np.testing.assert_allclose(dehomogenize(camera(on_screen)), dehomogenize(on_screen), atol=1e-14)

    twice = camera(camera)
    support = np.abs(camera.kernel) > 1e-9
    scale = np.mean(twice.kernel[support] / camera.kernel[support])
    np.testing.assert_allclose(twice.kernel, camera.kernel * scale, atol=1e-14)


def test_ternary_projector_binds_to_camera():
    """Binding centre and screen of the open ternary projector yields the camera map."""
    projector = Point.regressive(Point).wedge(Plane)
    assert projector.gatype == ga.gatype((Point, Point, Point, Plane))
    assert projector.kernel.shape == (4, 4, 4, 4)
    centre = point(np.array([0.3, -0.2, -1.0]))
    world = point(np.array([1.0, 2.0, 4.0]))
    camera = centre.regressive(Point).wedge(SCREEN)

    bound = projector.bind({0: centre, 2: SCREEN})
    assert bound.gatype == Camera
    np.testing.assert_allclose(bound.kernel, camera.kernel, atol=1e-14)
    np.testing.assert_allclose(projector(centre, world, SCREEN).kernel, camera(world).kernel, atol=1e-14)


def test_ideal_centre_is_orthographic():
    """A centre at infinity projects along a fixed direction: shadows of a distant sun."""
    ground = mv.z
    straight_down = direction(np.array([0.0, 0.0, -1.0])).regressive(Point).wedge(ground)
    world = point(np.array([[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0]]))
    np.testing.assert_allclose(dehomogenize(straight_down(world)), [[1.0, 2.0, 0.0], [-3.0, 0.5, 0.0]], atol=1e-14)

    slanted = direction(np.array([1.0, 0.0, -1.0])).regressive(Point).wedge(ground)
    np.testing.assert_allclose(dehomogenize(slanted(point(np.array([0.0, 0.0, 2.0])))), [2.0, 0.0, 0.0], atol=1e-14)


def test_batched_centres_give_batched_cameras():
    """A batch of centres binds to a batch of 4x4 camera kernels."""
    centres = point(np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    cameras = centres.regressive(Point).wedge(SCREEN)
    assert cameras.shape == (2,)
    assert cameras.kernel.shape == (2, 4, 4)
    world = point(np.array([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(dehomogenize(cameras(world)), [[-0.125, 0.5, 1.0], [0.625, 0.5, 1.0]], atol=1e-14)


def test_bivector_exponentials_translate_and_rotate():
    """The translator adds its displacement; the rotator is right-handed about its axis."""
    moved = (mv.xw * 0.5 - mv.yw + mv.zw * 0.25).exp() >> point(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(dehomogenize(moved), [2.0, 0.0, 3.5], atol=1e-14)

    y_axis = origin.regressive(direction(np.array([0.0, 1.0, 0.0]))).normalized()
    turned = (y_axis * (np.pi / 4.0)).exp() >> point(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(dehomogenize(turned), [0.0, 0.0, -1.0], atol=1e-8)


def test_moving_the_rig_equals_moving_centre_and_screen():
    """Transforming the camera map by a motor equals building it from transformed parts."""
    y_axis = origin.regressive(direction(np.array([0.0, 1.0, 0.0]))).normalized()
    motor = (mv.xw * 0.2 - mv.yw * 0.15 + mv.zw).exp() * (y_axis * 0.35).exp()
    camera = origin.regressive(Point).wedge(SCREEN)
    moved = motor >> camera(motor << Point)
    rebuilt = (motor >> origin).regressive(Point).wedge(motor >> SCREEN)
    np.testing.assert_allclose(moved.kernel, rebuilt.kernel, atol=1e-14)

    world = point(np.array([1.0, 2.0, 4.0]))
    local = dehomogenize(camera(motor << world))[:2]
    np.testing.assert_allclose(screen_coordinates(motor, moved(world)), local, atol=1e-14)


def test_line_camera_commutes_with_join():
    """The image of the line through two points is the line through their images."""
    centre = point(np.array([0.3, -0.2, -1.0]))
    camera = centre.regressive(Point).wedge(SCREEN)
    line_camera = centre.regressive(Line).wedge(SCREEN)
    a = point(np.array([1.0, 0.0, 2.0]))
    b = point(np.array([0.0, 1.0, 3.0]))
    imaged_join = line_camera(a.regressive(b)).cast(Line.output_subspace).kernel
    joined_images = camera(a).regressive(camera(b)).cast(Line.output_subspace).kernel
    support = np.abs(joined_images) > 1e-9
    scale = imaged_join[support][0] / joined_images[support][0]
    np.testing.assert_allclose(imaged_join, joined_images * scale, atol=1e-14)


def test_fundamental_form_and_epipolar_geometry():
    """Corresponding image points annihilate the fundamental form; epipoles span its kernel."""
    centre_1 = point(np.array([-0.6, 0.0, 0.0]))
    centre_2 = point(np.array([0.6, 0.1, -0.2]))
    camera_1 = centre_1.regressive(Point).wedge(SCREEN)
    camera_2 = centre_2.regressive(Point).wedge(SCREEN)
    form = centre_1.regressive(Point).wedge(centre_2.regressive(Point))
    assert form.gatype == Correspondence
    assert np.linalg.matrix_rank(form.kernel.squeeze()) == 2

    world = point(np.array([[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0], [0.2, -0.4, 3.0]]))
    image_1 = camera_1(world)
    image_2 = camera_2(world)
    np.testing.assert_allclose(form(image_1, image_2).kernel, 0.0, atol=1e-13)
    assert np.abs(form(image_1[0], image_2[2]).kernel).max() > 1e-3

    epipole_2 = camera_2(centre_1)
    np.testing.assert_allclose(form.bind({1: epipole_2}).kernel, 0.0, atol=1e-14)

    line_camera_2 = centre_2.regressive(Line).wedge(SCREEN)
    lines_2 = line_camera_2(centre_1.regressive(image_1))
    np.testing.assert_allclose(lines_2.regressive(image_2).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(lines_2.regressive(epipole_2).kernel, 0.0, atol=1e-13)


def test_shadow_trail_agrees_with_the_body_shadow():
    """Reopening the light slot reproduces the corner's shadow under the bound light."""
    body = cube(1.0)
    ground = mv.z
    light = point(np.array([1.0, -1.0, 4.0]))
    sun = direction(np.array([-1.0, 0.6, -2.5]))
    point_shadow, _, shadow_trail = core.shadows(body, ground, light, sun, body[7], light)
    np.testing.assert_allclose(
        shadow_trail.kernel, point_shadow[7].kernel, atol=1e-14
    )


def test_stereo_correspondence_and_epipolar_lines_vanish():
    """Corresponding images annihilate the form, and epipolar lines meet their points."""
    subject = (mv.zw * 2.5).exp() >> cube(1.6)
    screen = mv.z - mv.w
    rig_1 = (mv.xw * -0.3).exp() * (mv.xz * +0.06).exp()
    rig_2 = (mv.xw * +0.3).exp() * (mv.xz * -0.06).exp()
    image_1, image_2, epipole_2, epipolar_lines_2, correspondence = core.stereo(subject, origin, screen, rig_1, rig_2)

    np.testing.assert_allclose(correspondence(image_1, image_2).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(correspondence.bind({1: epipole_2}).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(epipolar_lines_2.regressive(image_2).kernel, 0.0, atol=1e-12)


def test_scenario_renders():
    """The scenario's geometry renders as a figure."""
    figure = render.draw_projection(*scenarios.projection())
    assert isinstance(figure, plt.Figure)
