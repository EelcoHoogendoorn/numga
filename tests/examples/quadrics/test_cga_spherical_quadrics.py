"""Unit tests for Spherical Quadric Vortex in 4D CGA Cl(3, 1)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from examples.quadrics.cga_spherical_quadrics import render, scenarios
from examples.quadrics.cga_spherical_quadrics.core import (
    ga,
    mv,
    Vector,
    Plane,
    Quadric,
    Bivector,
    make_spherical_donut,
    make_conical_donut,
    make_bernoulli_lemniscate,
    make_pinched_horn,
    make_eccentric_cyclide,
    make_spherical_cassini,
    make_spherical_crescent,
    make_spherical_spindle,
    make_spherical_clover,
    make_spherical_hourglass,
    make_spherical_parabola,
    make_circle_intersection_vortex,
    point,
)
from examples.quadrics.elliptic_physics.render import hemisphere


def test_algebra_and_gatypes():
    """Verify 4D algebra Cl(3, 1) and extensor type Quadric: Vector -> Plane."""
    assert ga.signature == (1, 1, 1, -1)
    assert Quadric == ga.gatype((Plane, Vector))

    donut = make_spherical_donut(np.radians(30.0), np.radians(10.0))
    assert donut.gatype == Quadric


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


def test_shape_constructors():
    """Verify all constructor functions produce valid Quadrics."""
    alpha = np.radians(25.0)
    assert make_spherical_cassini(alpha, alpha, 0.012).gatype == Quadric
    assert make_bernoulli_lemniscate(0.38).gatype == Quadric
    assert make_spherical_crescent(np.radians(45.0), np.radians(24.0), np.radians(16.0)).gatype == Quadric
    assert make_spherical_spindle(1.8, 0.8, 0.35).gatype == Quadric
    assert make_spherical_clover(np.radians(32.0), np.radians(28.0), 0.35).gatype == Quadric
    assert make_spherical_hourglass(np.radians(24.0), 0.010).gatype == Quadric
    assert make_spherical_parabola(1.0, 0.5, 0.20).gatype == Quadric
    assert make_eccentric_cyclide(np.radians(34.0), np.radians(13.0), 0.65).gatype == Quadric


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


def test_mathematics_does_not_import_plotting():
    """The math layer must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.quadrics.cga_spherical_quadrics.core, sys; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_vortex_animates():
    world, colors = scenarios.vortex(["trio"], 6)
    frames = render.vortex_frames(world, colors, 80, 1)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
