"""Coverage and mathematical property tests for projective cameras, shadows, and stereo views in PGA3D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.geometry.projection import core, render, scenarios
from examples.geometry.projection.core import Correspondence, Line, Plane, Point, cube, ga, mv

origin = mv.w.dual()                                                            # [] Point-
Camera = ga.gatype((Point, Point))                                              # Point- <- Point-
screen = mv.z - mv.w                                                            # [] Plane+
basis_points = mv.basis().dual()                                                # [4] Point-


def euclidean(points: Point) -> np.ndarray:
    """Pairing `points` against the coordinate planes and dividing by weight reads Euclidean coordinates."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def test_pinhole_image_is_perspective_division():
    """A pinhole at the origin with screen `z = 1` images `(x, y, z)` to `(x/z, y/z, 1)`."""
    camera = (origin & Point) ^ screen                                          # [] Point- <- Point-
    assert camera.gatype <= Camera
    world = mv.point([[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0], [0.2, -0.4, 0.5]])     # [3] Point-
    np.testing.assert_allclose(euclidean(camera(world)), [[0.25, 0.5, 1.0], [-1.5, 0.25, 1.0], [0.4, -0.8, 1.0]], atol=1e-12)


def test_camera_is_a_rank_three_projection():
    """The optical centre has zero image, screen points stay fixed, and the camera map is projectively idempotent."""
    centre = mv.point([0.3, -0.2, -1.0])                                        # [] Point-
    camera = (centre & Point) ^ screen                                          # [] Point- <- Point-
    assert (camera(centre).antinorm() < 1e-12).all()

    on_screen = mv.point([[0.7, 0.1, 1.0], [-2.0, 3.0, 1.0]])                   # [2] Point-
    np.testing.assert_allclose(euclidean(camera(on_screen)), euclidean(on_screen), atol=1e-12)

    twice_images = camera(camera(basis_points))                                 # [4] Point-
    once_images = camera(basis_points) * -(screen & centre)                     # [4] Point-
    assert ((twice_images - once_images).dual().norm() < 1e-12).all()


def test_ternary_projector_binds_to_camera():
    """Binding the centre and screen arguments of `(Point & Point) ^ Plane` yields the pinhole camera map."""
    projector = (Point & Point) ^ Plane                                         # [] Point- <- (Point-, Point-, Plane+)
    assert projector.gatype <= ga.gatype((Point, Point, Point, Plane))
    centre = mv.point([0.3, -0.2, -1.0])                                        # [] Point-
    world = mv.point([1.0, 2.0, 4.0])                                           # [] Point-
    camera = (centre & Point) ^ screen                                          # [] Point- <- Point-

    bound = projector.bind({0: centre, 2: screen})                              # [] Point- <- Point-
    assert bound.gatype <= Camera
    assert ((bound(basis_points) - camera(basis_points)).antinorm() < 1e-12).all()
    assert ((projector(centre, world, screen) - camera(world)).antinorm() < 1e-12).all()


def test_ideal_centre_is_orthographic():
    """An ideal centre at infinity casts parallel orthographic shadows onto the ground plane."""
    ground = mv.z                                                               # [] Plane+
    straight_down = (mv.direction([0, 0, -1]) & Point) ^ ground                 # [] Point- <- Point-
    world = mv.point([[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0]])                       # [2] Point-
    np.testing.assert_allclose(euclidean(straight_down(world)), [[1.0, 2.0, 0.0], [-3.0, 0.5, 0.0]], atol=1e-12)

    slanted = (mv.direction([1, 0, -1]) & Point) ^ ground                       # [] Point- <- Point-
    np.testing.assert_allclose(euclidean(slanted(mv.point([0.0, 0.0, 2.0]))), [2.0, 0.0, 0.0], atol=1e-12)


def test_batched_centres_give_batched_cameras():
    """A batch of optical centres produces a matching batch of pinhole camera maps."""
    centres = mv.point([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])                     # [2] Point-
    cameras = (centres & Point) ^ screen                                        # [2] Point- <- Point-
    assert cameras.shape == (2,)
    world = mv.point([1.0, 2.0, 4.0])                                           # [] Point-
    np.testing.assert_allclose(euclidean(cameras(world)), [[-0.125, 0.5, 1.0], [0.625, 0.5, 1.0]], atol=1e-12)


def test_moving_the_rig_equals_moving_centre_and_screen():
    """Conjugating the camera map by a pose motor matches building the camera from the posed centre and screen."""
    y_axis = mv.zx                                                              # [] Bivector+
    motor = (mv.xw * 0.2 - mv.yw * 0.15 + mv.zw).exp() * (y_axis * 0.35).exp()  # [] Motor+
    camera = (origin & Point) ^ screen                                          # [] Point- <- Point-
    moved = motor >> camera(motor << Point)                                     # [] Point- <- Point-
    rebuilt = ((motor >> origin) & Point) ^ (motor >> screen)                   # [] Point- <- Point-
    assert ((moved(basis_points) - rebuilt(basis_points)).antinorm() < 1e-12).all()


def test_line_camera_commutes_with_join():
    """Imaging the join of two points with the line camera matches joining their point images."""
    centre = mv.point([0.3, -0.2, -1.0])                                        # [] Point-
    camera = (centre & Point) ^ screen                                          # [] Point- <- Point-
    line_camera = (centre & Line) ^ screen                                      # [] Line- <- Line-
    first_point = mv.point([1.0, 0.0, 2.0])                                     # [] Point-
    second_point = mv.point([0.0, 1.0, 3.0])                                    # [] Point-
    imaged_join = line_camera(first_point & second_point)                       # [] Line-
    joined_images = camera(first_point) & camera(second_point)                  # [] Line-
    assert ((imaged_join + joined_images / (screen & centre)).scalar_norm_squared() < 1e-24).all()


def test_shadow_trail_and_stereo_scene():
    """Shadow casting, stereo epipolar incidence, and the three-panel figure all evaluate cleanly."""
    body = cube(1.0)                                                            # [8] Point-
    light = mv.point([1.0, -1.0, 4.0])                                          # [] Point-
    sun = mv.direction([-1.0, 0.6, -2.5])                                       # [] Point-
    point_shadow, _, shadow_trail = core.shadows(body, mv.z, light, sun, body[7], light)
    assert ((shadow_trail - point_shadow[7]).antinorm() < 1e-12).all()

    scene = scenarios.projection_scene()
    *_, (image_1, image_2, epipole_2, epipolar_lines_2, correspondence), _, _ = scene
    assert correspondence.gatype <= Correspondence
    assert (correspondence(image_1, image_2).dual().abs() < 1e-12).all()
    assert (correspondence(basis_points, epipole_2).dual().abs() < 1e-12).all()
    assert ((epipolar_lines_2 & image_2).norm() < 1e-12).all()
    assert ((epipolar_lines_2 & epipole_2).norm() < 1e-12).all()

    plt.close(render.projection_figure(*scene))
