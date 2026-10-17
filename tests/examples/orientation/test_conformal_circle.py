"""A point join and a surface intersection retain their own reflection behaviour."""

import matplotlib.pyplot as plt
import numpy as np

from numga import Algebra, NumpyContext, stack

from examples.orientation.conformal_circle import core, render


def test_circle_constructions_commute_with_reflection():
    ga = Algebra("x+y+z+w+e-")
    mv = NumpyContext(ga).multivector
    vertices = stack([(mv.x + mv.e).dual(), (-mv.y + mv.e).dual(), (-mv.x + mv.e).dual()])
    angles = np.linspace(0, 2 * np.pi, 49)
    longitudes = np.linspace(0, 2 * np.pi, 13)
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, 7)
    curve = core.circle_samples(vertices[0], mv.xy, angles)
    surface = core.sphere_samples(vertices[0], mv.xy, mv.xz, longitudes, latitudes)
    points, curve, surface, planes, spheres = core.reflected_scene(
        vertices, curve, surface, mv.z, mv.w, mv.z)
    through_points, from_surfaces = core.constructions(points[0], planes[0], spheres[0])
    mirror = planes[0]
    rebuilt_path, rebuilt_cut = core.constructions(points[1], planes[1], spheres[1])

    # Equal blade coefficients carry distinct constructions of the same circle.
    np.testing.assert_allclose(through_points.kernel, from_surfaces.kernel, atol=1e-12)
    np.testing.assert_allclose((mirror >> through_points).kernel, through_points.kernel, atol=1e-12)
    np.testing.assert_allclose((mirror >> from_surfaces).kernel, -from_surfaces.kernel, atol=1e-12)

    # Building from reflected ingredients gives the corresponding reflected result.
    np.testing.assert_allclose(rebuilt_path.kernel, (mirror >> through_points).kernel, atol=1e-12)
    np.testing.assert_allclose(rebuilt_cut.kernel, (mirror >> from_surfaces).kernel, atol=1e-12)
    np.testing.assert_allclose((planes[0] & curve[0]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((spheres[0] & curve[0]).kernel, 0, atol=1e-12)

    figure = render.draw_circle(points, curve, surface, planes, spheres)
    figure.canvas.draw()
    plt.close(figure)


def test_circulation_and_glide_preserve_circle_with_different_motion():
    ga = Algebra("x+y+z+w+e-")
    mv = NumpyContext(ga).multivector
    points = stack([(mv.x + mv.e).dual(), (-mv.y + mv.e).dual(), (-mv.x + mv.e).dual()])
    circle, _ = core.constructions(points, mv.z, mv.w)
    infinity = mv.e + mv.w
    amounts = np.array([0., .4, .8])
    axis = (circle.dual() | infinity).normalized()
    rotation = (-axis * (amounts / 2)).exp()
    glide = mv.x * ((mv.x | circle.dual()) * amounts).exp()
    fixed = stack([(mv.y + mv.e).dual(), (-mv.y + mv.e).dual()])
    rotated = rotation[:, None] >> points[None, :]
    glided = glide[:, None] >> points[None, :]

    # Both motions preserve incidence; the glide starts with a reflection and fixes two points.
    np.testing.assert_allclose((circle & rotated).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((circle & glided).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(rotation.symmetric_reverse_product().kernel, 1, atol=1e-12)
    np.testing.assert_allclose(glide.symmetric_reverse_product().kernel, 1, atol=1e-12)
    np.testing.assert_allclose(glided[0].kernel, (mv.x >> points).kernel, atol=1e-12)
    fixed_positions = render.coordinates(fixed)
    glided_fixed = render.coordinates(glide[:, None] >> fixed[None, :])
    np.testing.assert_allclose(glided_fixed - fixed_positions, 0, atol=1e-12)

    # Euclidean distances distinguish rigid circulation from the conformal glide.
    separation = np.linalg.norm(np.diff(render.coordinates(points), axis=-2), axis=-1)
    rotation_separation = np.linalg.norm(np.diff(render.coordinates(rotated), axis=-2), axis=-1)
    glide_separation = np.linalg.norm(np.diff(render.coordinates(glided), axis=-2), axis=-1)
    np.testing.assert_allclose(rotation_separation - separation, 0, atol=1e-12)
    assert np.all(np.abs(glide_separation[-1] - separation) > .5)
