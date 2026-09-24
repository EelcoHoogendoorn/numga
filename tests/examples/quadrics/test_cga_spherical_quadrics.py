"""Unit tests for Spherical Quadric Vortex in 4D CGA Cl(3, 1)."""

from __future__ import annotations


import numpy as np

from examples.quadrics.cga_spherical_quadrics import render, scenarios
from examples.quadrics.cga_spherical_quadrics.core import (
    mv,
    Quadric,
    Bivector,
    make_spherical_donut,
    make_conical_donut,
    make_pinched_horn,
    make_circle_intersection_vortex,
    point,
)
from examples.quadrics.elliptic_physics.render import hemisphere


def test_null_cone_pixels():
    """Hemisphere pixels are null vectors of weight one on the unit sphere."""
    coordinates, inside, rim = hemisphere(60)
    assert inside.sum() > 0
    pixels = point(coordinates)
    np.testing.assert_allclose((pixels | pixels).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((pixels | mv.w).to_array(), -1.0, atol=1e-12)


def test_conical_donut_topology():
    """Verify conical donut has valid Quadric GAType and apex touches north pole."""
    conical = make_conical_donut(0.55, 0.22)
    assert conical.gatype == Quadric
    p_apex = mv.vector([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose((conical(p_apex) & p_apex).kernel[0], 0.0, atol=1e-12)


def test_spherical_donut_topology():
    """Verify donut potential is negative inside the ring and positive in hole and exterior."""
    r_core = np.radians(40.0)
    r_tube = np.radians(10.0)
    donut = make_spherical_donut(r_core, r_tube)

    p_hole = mv.vector([0.0, 0.0, 1.0, 1.0])
    assert (donut(p_hole) & p_hole).kernel[0] > 0.0

    p_ring = mv.vector([np.sin(r_core), 0.0, np.cos(r_core), 1.0])
    assert (donut(p_ring) & p_ring).kernel[0] < 0.0

    p_out = mv.vector([1.0, 0.0, 0.0, 1.0])
    assert (donut(p_out) & p_out).kernel[0] > 0.0


def test_pinched_horn_cusp():
    """Verify pinched horn cyclide touches the pole with zero potential."""
    horn = make_pinched_horn(np.radians(60.0))
    p_cusp = mv.vector([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose((horn(p_cusp) & p_cusp).kernel[0], 0.0, atol=1e-12)


def test_circle_intersection_vortex():
    """Verify intersection 2-blade of two circles generates a valid conformal rotor."""
    generator = make_circle_intersection_vortex(
        np.array([np.sin(0.35), 0.0, np.cos(0.35)]),
        np.radians(48.0),
        np.array([0.0, np.sin(0.40), np.cos(0.40)]),
        np.radians(52.0),
    )
    assert generator.gatype.subspaces == Bivector.subspaces
    np.testing.assert_allclose(generator.squared().kernel[0], -1.0, atol=1e-12)

    rotor = (generator * 0.4).exp()
    p_test = mv.vector([0.5, 0.5, np.sqrt(0.5), 1.0])
    p_flow = rotor >> p_test
    np.testing.assert_allclose(p_flow.squared().kernel[0], 0.0, atol=1e-12)


def test_render_all_shapes():
    """A frame of each of the 13 shapes is a non-empty image."""
    table = scenarios.shapes()
    assert len(table) == 13
    for name in table:
        world, colors = scenarios.vortex([name], 1)
        frame = render.vortex_frames(world, colors, 60, 1)[0]
        assert frame.shape == (60, 60, 3)
        assert frame.dtype == np.uint8
        assert np.any(frame != frame[0, 0])


def test_vortex_animates():
    world, colors = scenarios.vortex(["trio"], 6)
    frames = render.vortex_frames(world, colors, 80, 1)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
