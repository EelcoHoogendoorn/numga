"""Unit tests for projective cameras in PGA3D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.projection import core, render, scenarios
from examples.geometry.projection.core import Correspondence, Line, Plane, Point, cube, ga, mv
from examples.geometry.projection.render import euclidean, screen_coordinates

origin = mv.w.dual()
Camera = ga.gatype((Point, Point))
SCREEN = mv.z - mv.w


def test_pinhole_image_is_perspective_division():
    """A pinhole at the origin with screen z = 1 images (x, y, z) to (x/z, y/z, 1)."""
    camera = (origin & Point) ^ SCREEN
    assert camera.gatype <= Camera
    world = mv.point([[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0], [0.2, -0.4, 0.5]])
    np.testing.assert_allclose(euclidean(camera(world)), [[0.25, 0.5, 1.0], [-1.5, 0.25, 1.0], [0.4, -0.8, 1.0]], atol=1e-14)


def test_camera_is_a_rank_three_projection():
    """The centre has no image, screen points are fixed, and the map is projectively idempotent."""
    centre = mv.point([0.3, -0.2, -1.0])
    camera = (centre & Point) ^ SCREEN
    assert np.linalg.matrix_rank(camera.kernel) == 3
    np.testing.assert_allclose(camera(centre).kernel, 0.0, atol=1e-14)

    on_screen = mv.point([[0.7, 0.1, 1.0], [-2.0, 3.0, 1.0]])
    np.testing.assert_allclose(euclidean(camera(on_screen)), euclidean(on_screen), atol=1e-14)

    twice = camera(camera)
    support = np.abs(camera.kernel) > 1e-9
    scale = np.mean(twice.kernel[support] / camera.kernel[support])
    np.testing.assert_allclose(twice.kernel, camera.kernel * scale, atol=1e-14)


def test_ternary_projector_binds_to_camera():
    """Binding centre and screen of the projector, a map on centre, point and screen, yields the camera map."""
    projector = (Point & Point) ^ Plane
    assert projector.gatype <= ga.gatype((Point, Point, Point, Plane))
    assert projector.kernel.shape == (4, 4, 4, 4)
    centre = mv.point([0.3, -0.2, -1.0])
    world = mv.point([1.0, 2.0, 4.0])
    camera = (centre & Point) ^ SCREEN

    bound = projector.bind({0: centre, 2: SCREEN})
    assert bound.gatype <= Camera
    np.testing.assert_allclose(bound.kernel, camera.kernel, atol=1e-14)
    np.testing.assert_allclose(projector(centre, world, SCREEN).kernel, camera(world).kernel, atol=1e-14)


def test_ideal_centre_is_orthographic():
    """A centre at infinity projects along a fixed direction: shadows of a distant sun."""
    ground = mv.z
    straight_down = (mv.direction([0.0, 0.0, -1.0]) & Point) ^ ground
    world = mv.point([[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0]])
    np.testing.assert_allclose(euclidean(straight_down(world)), [[1.0, 2.0, 0.0], [-3.0, 0.5, 0.0]], atol=1e-14)

    slanted = (mv.direction([1.0, 0.0, -1.0]) & Point) ^ ground
    np.testing.assert_allclose(euclidean(slanted(mv.point([0.0, 0.0, 2.0]))), [2.0, 0.0, 0.0], atol=1e-14)


def test_batched_centres_give_batched_cameras():
    """A batch of centres binds to a batch of 4x4 camera kernels."""
    centres = mv.point([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cameras = (centres & Point) ^ SCREEN
    assert cameras.shape == (2,)
    assert cameras.kernel.shape == (2, 4, 4)
    world = mv.point([1.0, 2.0, 4.0])
    np.testing.assert_allclose(euclidean(cameras(world)), [[-0.125, 0.5, 1.0], [0.625, 0.5, 1.0]], atol=1e-14)


def test_bivector_exponentials_translate_and_rotate():
    """The translator adds its displacement; the rotator is right-handed about its axis."""
    moved = (mv.xw * 0.5 - mv.yw + mv.zw * 0.25).exp() >> mv.point([1.0, 2.0, 3.0])
    np.testing.assert_allclose(euclidean(moved), [2.0, 0.0, 3.5], atol=1e-14)

    y_axis = mv.zx
    turned = (y_axis * (-np.pi / 4.0)).exp() >> mv.point([1.0, 0.0, 0.0])
    np.testing.assert_allclose(euclidean(turned), [0.0, 0.0, -1.0], atol=1e-8)


def test_moving_the_rig_equals_moving_centre_and_screen():
    """Transforming the camera map by a motor equals building it from transformed parts."""
    y_axis = mv.zx
    motor = (mv.xw * 0.2 - mv.yw * 0.15 + mv.zw).exp() * (y_axis * 0.35).exp()
    camera = (origin & Point) ^ SCREEN
    moved = motor >> camera(motor << Point)
    rebuilt = ((motor >> origin) & Point) ^ (motor >> SCREEN)
    np.testing.assert_allclose(moved.kernel, rebuilt.kernel, atol=1e-14)

    world = mv.point([1.0, 2.0, 4.0])
    local = euclidean(camera(motor << world))[:2]
    np.testing.assert_allclose(screen_coordinates(motor, moved(world)), local, atol=1e-14)


def test_line_camera_commutes_with_join():
    """The image of the line through two points is the line through their images."""
    centre = mv.point([0.3, -0.2, -1.0])
    camera = (centre & Point) ^ SCREEN
    line_camera = (centre & Line) ^ SCREEN
    a = mv.point([1.0, 0.0, 2.0])
    b = mv.point([0.0, 1.0, 3.0])
    imaged_join = line_camera(a & b)
    joined_images = camera(a) & camera(b)
    np.testing.assert_allclose((imaged_join / imaged_join.norm() - joined_images / joined_images.norm()).kernel, 0.0, atol=1e-14)


def test_fundamental_form_and_epipolar_geometry():
    """Corresponding image points annihilate the fundamental form; epipoles span its kernel."""
    centre_1 = mv.point([-0.6, 0.0, 0.0])
    centre_2 = mv.point([0.6, 0.1, -0.2])
    camera_1 = (centre_1 & Point) ^ SCREEN
    camera_2 = (centre_2 & Point) ^ SCREEN
    form = (centre_1 & Point) ^ (centre_2 & Point)
    assert form.gatype <= Correspondence
    assert np.linalg.matrix_rank(form.kernel[0]) == 2

    world = mv.point([[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0], [0.2, -0.4, 3.0]])
    image_1 = camera_1(world)
    image_2 = camera_2(world)
    np.testing.assert_allclose(form(image_1, image_2).kernel, 0.0, atol=1e-13)
    assert np.abs(form(image_1[0], image_2[2]).kernel).max() > 1e-3

    epipole_2 = camera_2(centre_1)
    np.testing.assert_allclose(form.bind({1: epipole_2}).kernel, 0.0, atol=1e-14)

    line_camera_2 = (centre_2 & Line) ^ SCREEN
    lines_2 = line_camera_2(centre_1 & image_1)
    np.testing.assert_allclose((lines_2 & image_2).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((lines_2 & epipole_2).kernel, 0.0, atol=1e-13)


def test_shadow_trail_agrees_with_the_body_shadow():
    """The shadow as a map on the light reproduces the corner's shadow under that light."""
    body = cube(1.0)
    ground = mv.z
    light = mv.point([1.0, -1.0, 4.0])
    sun = mv.direction([-1.0, 0.6, -2.5])
    point_shadow, _, shadow_trail = core.shadows(body, ground, light, sun, body[7], light)
    np.testing.assert_allclose(shadow_trail.kernel, point_shadow[7].kernel, atol=1e-14)


def test_stereo_correspondence_and_epipolar_lines_vanish():
    """Corresponding images annihilate the form, and epipolar lines meet their points."""
    *_, (image_1, image_2, epipole_2, epipolar_lines_2, correspondence), _, _ = scenarios.projection()

    np.testing.assert_allclose(correspondence(image_1, image_2).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(correspondence.bind({1: epipole_2}).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((epipolar_lines_2 & image_2).kernel, 0.0, atol=1e-12)


def test_scenario_renders():
    """The scenario's geometry renders as a figure."""
    figure = render.draw_projection(*scenarios.projection())
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
