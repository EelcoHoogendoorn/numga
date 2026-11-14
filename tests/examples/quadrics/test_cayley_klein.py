"""Coverage tests for Cayley-Klein geometry in PGA2D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.cayley_klein import render, scenarios
from examples.quadrics.cayley_klein.core import (
    Plane,
    Point,
    Polarity,
    concentric_spheres,
    mv,
    normalized_pairing,
    perpendicular_reflection,
    triangle_metrics,
)
from numga import stack
from numga.topology import Topology

coordinate_planes = stack([mv.x, mv.y, mv.w])                              # [3] Plane+
hyperbolic_absolute = (coordinate_planes * (coordinate_planes & Point) * np.array([1, 1, -1])).sum(axis=0)
elliptic_absolute = (coordinate_planes * (coordinate_planes & Point) * np.array([1, 1, 1])).sum(axis=0)


def test_absolutes_have_polarity_and_pole_types() -> None:
    assert hyperbolic_absolute.gatype == Polarity
    assert elliptic_absolute.gatype == Polarity
    pole_map = hyperbolic_absolute.solve(Plane)                            # [] Point- <- Plane+
    probe = mv.point([0.3, -0.2])                                          # [] Point-
    assert ((pole_map(hyperbolic_absolute(probe)) - probe).antinorm() < 1e-12).all()


def test_radial_distances_match_hyperbolic_arctanh_and_elliptic_arctan() -> None:
    origin = mv.point([0, 0])                                              # [] Point-
    radii = np.array([0.2, 0.6])
    radial_points = mv.point(np.stack([radii, np.zeros_like(radii)], axis=-1))  # [2] Point-

    hyperbolic_pairing = -(hyperbolic_absolute(Point) & Point)
    hyperbolic_distances = normalized_pairing(hyperbolic_pairing, origin, radial_points).clip(1.0, np.inf).arccosh()
    assert ((hyperbolic_distances - np.arctanh(radii)).abs() < 1e-12).all()

    elliptic_pairing = elliptic_absolute(Point) & Point
    elliptic_distances = normalized_pairing(elliptic_pairing, origin, radial_points).clip(-1.0, 1.0).arccos()
    assert ((elliptic_distances - np.arctan(radii)).abs() < 1e-12).all()


def test_perpendicular_and_harmonic_reflection_in_both_geometries() -> None:
    first_vertex, second_vertex = mv.point([0.65, 0]), mv.point([0.2, 0.55])
    probe = mv.point([-0.15, 0.15])
    side = (first_vertex & second_vertex) * (first_vertex & second_vertex & probe)

    for sign, absolute in ((-1, hyperbolic_absolute), (1, elliptic_absolute)):
        pole, normal, foot, reflected_point = perpendicular_reflection(absolute, side, probe)
        distance_pairing = sign * (absolute(Point) & Point)
        assert ((side & (absolute(probe) ^ absolute(pole))).abs() < 1e-12).all()
        assert ((side & foot).abs() < 1e-12).all()
        assert ((normal & reflected_point).antinorm() < 1e-12).all()
        distance_difference = (
            normalized_pairing(distance_pairing, probe, foot)
            - normalized_pairing(distance_pairing, reflected_point, foot)
        )
        assert (distance_difference.abs() < 1e-12).all()


def test_triangle_angle_sum_is_below_pi_in_hyperbolic_and_above_pi_in_elliptic() -> None:
    topology = Topology(3, np.array([[0, 1, 2]]))
    vertices = mv.point([[0, 0], [0.65, 0], [0.2, 0.55]]).field()          # Point-[V]
    _, hyperbolic_angles, _, hyperbolic_area = triangle_metrics(topology, hyperbolic_absolute, vertices)
    _, elliptic_angles, _, _ = triangle_metrics(topology, elliptic_absolute, vertices)
    assert (hyperbolic_area > 0).all()
    assert (hyperbolic_angles.sum(axis=0) < np.pi).all()
    assert (elliptic_angles.sum(axis=0) > np.pi).all()


def test_sphere_quadric_vanishes_on_points_at_its_hyperbolic_radius() -> None:
    centre = mv.point([0, 0])                                              # [] Point-
    radius = 0.9
    sphere = concentric_spheres(hyperbolic_absolute, centre, np.array([radius]))[0]
    boundary_point = mv.point([np.tanh(radius), 0])                        # [] Point-
    assert ((sphere(boundary_point) & boundary_point).abs() < 1e-12).all()


def test_hyperbolic_figure_renders() -> None:
    (
        absolute, vertices, sides, angles, _, probe, foot, normal, reflected_point, pole, sphere_quadrics, centres,
    ) = scenarios.hyperbolic_scene()
    fig = render.hyperbolic_figure(
        absolute, vertices, sides, angles, probe, foot, normal, reflected_point, pole, sphere_quadrics, centres,
    )
    plt.close(fig)
