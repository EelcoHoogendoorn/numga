"""Coverage tests for spherical quadrics in the conformal algebra of the sphere."""

from __future__ import annotations

import numpy as np

from examples.quadrics.cga_spherical_quadrics import render, scenarios
from examples.quadrics.cga_spherical_quadrics.core import (
    Quadric,
    conical_donut,
    mv,
    pinched_horn,
    sphere_points,
    spherical_donut,
)


def test_null_cone_points():
    """Unit directions lifted to conformal sphere points are null and pair with `w` to minus one."""
    probes = sphere_points(np.array([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.36, 0.48, 0.8]]))
    assert ((probes | probes).abs() < 1e-12).all()
    assert (((mv.w & probes) + 1).abs() < 1e-12).all()


def test_conical_donut_topology():
    """The conical donut is a symmetric polarity map whose conical apex touches the north pole."""
    conical = conical_donut(0.55, 0.22)
    assert conical.gatype == Quadric
    apex = sphere_points(np.array([0.0, 0.0, 1.0]))
    sample = sphere_points(np.array([0.6, 0.0, 0.8]))
    assert (((conical.adjugate()(apex) & sample) - (conical(sample) & apex)).abs() < 1e-12).all()
    assert ((conical(apex) & apex).abs() < 1e-12).all()


def test_spherical_donut_topology():
    """The spherical donut's quadratic form is negative inside the ring and positive in the hole and outside."""
    r_core = np.radians(40)
    r_tube = np.radians(10)
    donut = spherical_donut(r_core, r_tube)
    probes = sphere_points(np.array([[0.0, 0.0, 1.0], [np.sin(r_core), 0.0, np.cos(r_core)], [1.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(donut(probes) & probes > 0, [True, False, True])


def test_pinched_horn_cusp():
    """The pinched horn cyclide has a self-touching cusp at the north pole where its quadratic form vanishes."""
    horn = pinched_horn(np.radians(60))
    cusp = sphere_points(np.array([0.0, 0.0, 1.0]))
    assert ((horn(cusp) & cusp).abs() < 1e-12).all()


def test_circle_intersection_vortex():
    """Two intersecting circles meet in an elliptic 2-blade whose conformal rotor preserves the null sphere."""
    generator = scenarios.generator
    assert ((generator.squared() + 1).scalar_norm_squared() < 1e-20).all()

    rotor = (generator * 0.4).exp()
    carried = rotor >> sphere_points(np.array([0.5, 0.5, np.sqrt(0.5)]))
    assert ((carried | carried).abs() < 1e-12).all()


def test_one_frame_of_the_vortex_draws():
    """One streamed state of the three-shape vortex renders to an RGB frame."""
    states, colors = scenarios.vortex_states(["trio"], 1)
    frame = next(render.vortex_animation(colors, 32, states))
    assert frame.shape == (32, 32, 3)
