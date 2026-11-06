"""Checks of the Dupin cyclide tracer on the 3-sphere: the ray circle, its parabola, and the cylinder's quartic."""


import numpy as np

from numga import stack

from examples.quadrics.cyclides import render, scenarios
from examples.quadrics.cyclides.core import (
    Point, antipode, cylinder, mv, origin, ray_bend, ray_linear, ray_quadratic, ray_rotation,
)

direction = mv.y.dual() * 0.6 + mv.z.dual() * 0.8
angles = np.array([0.7, 2.5])


def test_ray_is_a_great_circle():
    """The eye turned exactly by t, and the circle through the ray maps, lie on the sphere of radius t about the eye."""
    exact = (ray_rotation(direction) * angles).exp() >> origin
    circle = origin + ray_linear(direction) * np.sin(angles) + ray_quadratic(direction, direction) * (1 - np.cos(angles))
    radius_spheres = mv.w + mv.e * np.cos(angles)
    np.testing.assert_allclose((radius_spheres & exact).to_array(), 0.0, atol=1e-7)
    np.testing.assert_allclose((radius_spheres & circle).to_array(), 0.0, atol=1e-12)


def test_the_bend_of_the_ray_parabola_is_the_antipode():
    """In the half angle u the circle is a parabola, and its bend is the antipode of the eye."""
    u = np.tan(angles / 2)
    parabola = origin + ray_linear(direction) * (2 * u) + ray_bend(direction, direction) * u**2
    circle = origin + ray_linear(direction) * np.sin(angles) + ray_quadratic(direction, direction) * (1 - np.cos(angles))
    probes = stack((mv.x, mv.y, mv.z, mv.w, mv.e))
    np.testing.assert_allclose((probes & (ray_bend(direction, direction) - antipode)).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((probes[:, None] & (parabola / (1 + u**2) - circle)).to_array(), 0.0, atol=1e-12)


def test_the_cylinder_polynomial_is_palindromic_and_a_dupin_cyclide_breaks_it():
    """Built from great spheres and e only, the cylinder's quartic in u has equal outer and opposite odd
    coefficients, so its hits come in antipodal pairs; an off-axis dilation breaks that symmetry."""
    aim = (mv.zx * (-0.7 / 2)).exp() >> mv.z.dual()
    dilation = ((mv.e.dual() & aim).dual() * (1.5 / 2)).exp()
    lopsided = dilation >> cylinder(0.25)(dilation << Point)
    sample = scenarios.pixels[::20000]
    gaps = []
    for surface in (cylinder(0.3), lopsided):
        form = Point & surface
        outer = form(ray_bend, ray_bend)(sample, sample, sample, sample) - form(origin, origin)
        odd = (4 * form(ray_linear, ray_bend))(sample, sample, sample) + (4 * form(origin, ray_linear))(sample)
        gaps.append(np.abs(outer.to_array()).max() + np.abs(odd.to_array()).max())
    assert gaps[0] < 1e-12 and gaps[1] > 1e-3


def test_one_frame_of_the_vortex_and_of_a_flat_scene_draws():
    traced = list(scenarios.vortex(1))
    render.draw_facing(*traced[0], scenarios.SHAPE)
    next(render.facing_frames(traced, scenarios.SHAPE))
    next(render.scene_frames(scenarios.flat_scene("cyclide", 1), scenarios.SHAPE))
