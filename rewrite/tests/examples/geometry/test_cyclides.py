"""Checks of the Dupin cyclide tracer on the 3-sphere: the ray circle, its parabola, and the traced scenes."""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.geometry.cyclides import render, scenarios
from examples.geometry.cyclides.core import (
    Direction, Point, antipode, dilation, cylinder, mv, origin, ray_bend, ray_linear, ray_quadratic, ray_rotation,
)

direction = mv(Direction, [0.0, 0.6, 0.8])
angles = np.array([0.7, 2.5])


def test_mathematics_does_not_import_plotting():
    probe = (
        "import examples.geometry.cyclides.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    repo_rewrite = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": f"{repo_rewrite / 'src'}:{repo_rewrite}"}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_ray_is_a_great_circle():
    """The eye turned exactly by t, and the circle through the ray maps, lie on the sphere of radius t about the eye."""
    exact = (ray_rotation(direction) * angles).exp() >> origin
    circle = origin + ray_linear(direction) * np.sin(angles) + ray_quadratic(direction, direction) * (1 - np.cos(angles))
    reach = mv.w + mv.e * np.cos(angles)
    np.testing.assert_allclose((reach & exact).to_array(), 0.0, atol=1e-9)
    np.testing.assert_allclose((reach & circle).to_array(), 0.0, atol=1e-12)


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
    aim = mv.z * np.cos(0.7) + mv.x * np.sin(0.7)
    lopsided = dilation(aim, 1.5) >> cylinder(0.25)(dilation(aim, 1.5) << Point)
    camera_map = mv.rotor() >> Point
    sample = scenarios.PIXELS[::20000]
    gaps = []
    for surface in (cylinder(0.3), lopsided):
        form = camera_map & surface(camera_map)
        outer = form(ray_bend, ray_bend)(sample, sample, sample, sample) - form(origin, origin)
        odd = (4 * form(ray_linear, ray_bend))(sample, sample, sample) + (4 * form(origin, ray_linear))(sample)
        gaps.append(np.abs(outer.to_array()).max() + np.abs(odd.to_array()).max())
    assert gaps[0] < 1e-12 and gaps[1] > 1e-3


def test_scenes_are_hit_and_draw():
    for scene, panels in ((scenarios.tori, 3), (scenarios.dupin, 3), (scenarios.spindles, 5)):
        facing, angle = scene()
        assert np.isfinite(angle).reshape(panels, -1).any(axis=-1).all()
        assert isinstance(render.draw_facing(facing, angle, scenarios.SHAPE), plt.Figure)


def test_the_vortex_stays_in_view():
    frames = list(render.facing_frames(scenarios.vortex(3), scenarios.SHAPE))
    assert len(frames) == 3 and all(frame.shape == (*scenarios.SHAPE, 3) for frame in frames)
    assert all(np.isfinite(angle).any() for _, angle in scenarios.vortex(3))


def test_the_flat_scenes_are_hit():
    for name in scenarios.FLAT_SCENES:
        for _, angle in scenarios.flat_scene(name, 2):
            assert np.isfinite(angle).any(axis=-1).all(), name
