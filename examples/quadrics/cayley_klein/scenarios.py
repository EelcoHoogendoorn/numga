"""Beltrami-Klein hyperbolic triangle, perpendicular reflection, and concentric spheres."""

from __future__ import annotations

import numpy as np

from examples.quadrics.cayley_klein.core import (
    Line,
    Plane,
    Point,
    Polarity,
    Scalar,
    concentric_spheres,
    mv,
    normalized_pairing,
    perpendicular_reflection,
    triangle_metrics,
)
from numga import stack
from numga.topology import Topology

triangle_simplices = np.array([[0, 1, 2]])                                 # [1, 3]
vertex_coordinates = [[0, 0], [0.65, 0], [0.2, 0.55]]
probe_coordinates = [-0.15, 0.15]
centre_coordinates = [[0, 0], [0.45, 0.25]]
sphere_radii = np.array([0.25, 0.55, 0.9, 1.3])
# Signs of x, y, and w in the unit-circle absolute x^2 + y^2 - w^2 == 0.
absolute_signs = np.array([1, 1, -1])


# --- math -----------------------------------------------------------------------------
def hyperbolic_scene() -> tuple[
    Polarity, Point, Plane, Scalar, Scalar, Point, Point, Line, Point, Point, Polarity, Point,
]:
    """Beltrami-Klein absolute, triangle geometry, perpendicular reflection across the hypotenuse, and
    two concentric sphere families."""
    coordinate_planes = stack([mv.x, mv.y, mv.w])                          # [3] Plane+
    absolute = (coordinate_planes * (coordinate_planes & Point) * absolute_signs).sum(axis=0)  # [] Plane+ <- Point-

    topology = Topology(len(vertex_coordinates), triangle_simplices)
    vertices = mv.point(vertex_coordinates).field()                        # Point-[V]
    sides, angles, _, area = triangle_metrics(topology, absolute, vertices)

    probe = mv.point(probe_coordinates)                                    # [] Point-
    pole, normal, foot, reflected_point = perpendicular_reflection(absolute, sides[0], probe)

    n_centres = len(centre_coordinates)
    n_radii = len(sphere_radii)
    centres = mv.point(centre_coordinates)                                 # [n_centres] Point-
    sphere_quadrics = concentric_spheres(absolute, centres, sphere_radii)  # [n_centres, n_radii] Plane+ <- Point-
    return (
        absolute, vertices, sides, angles, area, probe, foot, normal, reflected_point, pole, sphere_quadrics, centres,
    )


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.quadrics.cayley_klein import render
    from examples.util.animation import save_figure

    (
        absolute, vertices, sides, angles, area, probe, foot, normal, reflected_point, pole, sphere_quadrics, centres,
    ) = hyperbolic_scene()
    save_figure(
        render.hyperbolic_figure(
            absolute, vertices, sides, angles, probe, foot, normal, reflected_point, pole, sphere_quadrics, centres,
        ),
        "cayley_klein",
    )

    # checks
    # Hyperbolic triangles have positive angle defect, the pole of the perpendicular lies on the side,
    # and harmonic reflection across the side preserves hyperbolic distance to the foot.
    distance_pairing = -(absolute(Point) & Point)                          # [] Scalar+ <- (Point-, Point-)
    assert (area > 0).all()
    assert ((sides[0] & (absolute(probe) ^ absolute(pole))).abs() < 1e-12).all()
    distance_difference = (
        normalized_pairing(distance_pairing, probe, foot)
        - normalized_pairing(distance_pairing, reflected_point, foot)
    )
    assert (distance_difference.abs() < 1e-12).all()


if __name__ == "__main__":
    main()
