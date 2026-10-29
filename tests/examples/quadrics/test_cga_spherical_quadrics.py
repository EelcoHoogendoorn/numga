"""Spherical quadrics in the conformal algebra Cl(3, 1), and the conformal flow that carries them."""

from __future__ import annotations


import numpy as np

from examples.quadrics.cga_spherical_quadrics import render, scenarios
from examples.quadrics.cga_spherical_quadrics.core import (
    mv,
    Quadric,
    make_spherical_donut,
    make_conical_donut,
    make_pinched_horn,
    point,
)
from examples.quadrics.elliptic_physics.render import axes, hemisphere


def test_null_cone_pixels():
    """Hemisphere pixels are null, their weight on the unit sphere minus one, the square of w."""
    points, inside, rim = hemisphere(60)
    assert inside.sum() > 0
    pixels = point((axes & points[..., None]).to_array())
    np.testing.assert_allclose((pixels | pixels).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((mv.w & pixels).to_array(), -1.0, atol=1e-12)


def test_conical_donut_topology():
    """The conical donut is a map from points to planes, and its apex touches the north pole."""
    conical = make_conical_donut(0.55, 0.22)
    assert conical.gatype == Quadric
    apex = point(np.array([0, 0, 1]))
    np.testing.assert_allclose((conical(apex) & apex).to_array(), 0.0, atol=1e-12)


def test_spherical_donut_topology():
    """The donut's form is negative inside the ring and positive in the hole and outside."""
    r_core = np.radians(40.0)
    r_tube = np.radians(10.0)
    donut = make_spherical_donut(r_core, r_tube)
    # The pole in the hole, a point on the ring's core circle, and a point on the equator outside.
    probes = point(np.array([[0, 0, 1], [np.sin(r_core), 0, np.cos(r_core)], [1, 0, 0]]))
    np.testing.assert_array_equal(donut(probes) & probes > 0, [True, False, True])


def test_pinched_horn_cusp():
    """The pinched horn cyclide's cusp sits on the pole, where its form vanishes."""
    horn = make_pinched_horn(np.radians(60.0))
    cusp = point(np.array([0, 0, 1]))
    np.testing.assert_allclose((horn(cusp) & cusp).to_array(), 0.0, atol=1e-12)


def test_circle_intersection_vortex():
    """The 2-blade where two circles meet generates a conformal rotor, which keeps points of S² null."""
    generator = scenarios.generator
    np.testing.assert_allclose((generator.squared() + 1).kernel, 0.0, atol=1e-12)

    rotor = (generator * 0.4).exp()
    carried = rotor >> point(np.array([0.5, 0.5, np.sqrt(0.5)]))
    np.testing.assert_allclose(carried.squared().to_array(), 0.0, atol=1e-12)


def test_render_all_shapes():
    """A frame of each shape is a non-empty image."""
    table = scenarios.shapes()
    for name in table:
        world, colors = scenarios.vortex([name], 1)
        frame = list(render.vortex_frames(world, colors, 60))[0]
        assert frame.shape == (60, 60, 3)
        assert frame.dtype == np.uint8
        assert np.any(frame != frame[0, 0])


def test_vortex_animates():
    world, colors = scenarios.vortex(["trio"], 6)
    frames = list(render.vortex_frames(world, colors, 80))
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
