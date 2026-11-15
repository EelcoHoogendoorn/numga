"""Checks of the Dupin cyclide tracer on the 3-sphere: great-circle rays, parabolic rays, and cylinder quartics."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.quadrics.cyclides import render, scenarios
from examples.quadrics.cyclides.core import (
    Point, antipode, cylinder_quadrics, mv, origin, ray_bend, ray_linear, ray_quadratic, ray_rotation,
)

direction = mv.y.dual() * 0.6 + mv.z.dual() * 0.8                              # [] Direction-
angles = np.array([0.7, 2.5])


def test_ray_is_a_great_circle():
    """Turning the eye by `t` via the exact rotor and via the trigonometric ray maps lands on the
    sphere of angular radius `t` about the eye."""
    rotors = (ray_rotation(direction) * angles).exp()                          # [2] Motor+
    exact = rotors >> origin                                                   # [2] Point-
    circle = (origin + ray_linear(direction) * np.sin(angles)
              + ray_quadratic(direction, direction) * (1 - np.cos(angles)))    # [2] Point-
    radius_spheres = mv.w + mv.e * np.cos(angles)                              # [2] Sphere+
    assert ((radius_spheres & exact).abs() < 1e-7).all()
    assert ((radius_spheres & circle).abs() < 1e-10).all()


def test_the_bend_of_the_ray_parabola_is_the_antipode():
    """Clearing denominators in the half-angle tangent `u` turns the great circle into a parabola
    whose quadratic bend equals the antipode of the eye."""
    half_tangents = np.tan(angles / 2)
    parabola = (origin + ray_linear(direction) * (2 * half_tangents)
                + ray_bend(direction, direction) * half_tangents ** 2)         # [2] Point-
    circle = (origin + ray_linear(direction) * np.sin(angles)
              + ray_quadratic(direction, direction) * (1 - np.cos(angles)))    # [2] Point-
    probes = stack((mv.x, mv.y, mv.z, mv.w, mv.e))                             # [5] Sphere+
    assert ((probes & (ray_bend(direction, direction) - antipode)).abs() < 1e-10).all()
    assert ((probes[:, None] & (parabola / (1 + half_tangents ** 2) - circle)).abs() < 1e-10).all()


def test_the_cylinder_polynomial_is_palindromic_and_a_dupin_cyclide_breaks_it():
    """Built from great spheres and `e`, a placed spherical cylinder keeps a palindromic quartic in
    `u`, whereas a placed Dupin cyclide breaks both outer and odd coefficient symmetries."""
    samples = scenarios.pixels[::20000]                                        # [...] Direction-
    cylinder_turn = (mv.xw * 0.2 + mv.yz * 0.6).exp()                          # [] Motor+
    base_cylinder = cylinder_quadrics(0.3)                                     # [] Sphere+ <- Point-
    placed_cylinder = cylinder_turn >> base_cylinder(cylinder_turn << Point)   # [] Sphere+ <- Point-

    cylinder_form = Point & placed_cylinder                                    # [] Scalar+ <- (Point-, Point-)
    cylinder_outer = (cylinder_form(ray_bend, ray_bend)(samples, samples, samples, samples)
                      - cylinder_form(origin, origin))                         # [...] Scalar+
    cylinder_odd = ((4 * cylinder_form(ray_linear, ray_bend))(samples, samples, samples)
                    + (4 * cylinder_form(origin, ray_linear))(samples))        # [...] Scalar+

    dupin_form = Point & scenarios.placed_dupin_cyclides()[0]                  # [] Scalar+ <- (Point-, Point-)
    dupin_outer = (dupin_form(ray_bend, ray_bend)(samples, samples, samples, samples)
                   - dupin_form(origin, origin))                               # [...] Scalar+
    dupin_odd = ((4 * dupin_form(ray_linear, ray_bend))(samples, samples, samples)
                 + (4 * dupin_form(origin, ray_linear))(samples))              # [...] Scalar+

    assert (cylinder_outer.abs() < 1e-10).all()
    assert (cylinder_odd.abs() < 1e-10).all()
    assert (dupin_outer.abs() > 1e-3).all()
    assert (dupin_odd.abs() > 1e-3).all()


def test_one_frame_of_the_vortex_and_of_a_flat_scene_draws():
    """Single frames of the spherical vortex and flat-chart scenes render cleanly through `render`."""
    states = list(scenarios.tilted_vortex_states(1))
    fig = render.facing_figure(*states[0], scenarios.panel_shape)
    plt.close(fig)
    frame = next(render.facing_animation(scenarios.panel_shape, states))
    scene_frame = next(render.scene_animation(scenarios.panel_shape, render.body_palette,
                                              scenarios.flat_scene_states("cyclide", 1)))
    gallery = render.gallery_figure([frame, scene_frame], 2)
    plt.close(gallery)
