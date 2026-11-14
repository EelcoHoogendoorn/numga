"""Scene parameters and figure generation for projective ellipsoids and their motor transport."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quadrics.quadrics.core import DualQuadric, Motor, Plane, Point, ellipsoid, mv, support_plane

# Semi-axes along x, y, and z, the ellipsoid's center, and the outward normal direction of the tangent plane.
semi_axes = [[3, 0, 0], [0, 2, 0], [0, 0, 1]]
center = [0, 0, 0]
facing = [1, 2, 3, 0]
# Translation of the center and preceding xy-plane turn angle for rigid transport.
displacement = np.array([1.0, 2.0, 3.0])
turn_angle = -np.pi / 6


# --- math -----------------------------------------------------------------------------
def rest_ellipsoid() -> DualQuadric:
    """Forms the reference ellipsoid (`[] Point- <- Plane+`) at the origin from `semi_axes` and `center`."""
    return ellipsoid(mv.direction(semi_axes), mv.point(center))                    # [] Point- <- Plane+


def polar_reciprocity() -> tuple[DualQuadric, Plane, Point, Plane]:
    """Finds the tangent plane facing `facing`, its contact point, and the plane solved back from that point."""
    dual = rest_ellipsoid()                                                        # [] Point- <- Plane+
    tangent = support_plane(dual, mv.vector(facing))                               # [] Plane+
    contact = dual(tangent)                                                        # [] Point-
    recovered_tangent = dual.solve(contact)                                        # [] Plane+
    return dual, tangent, contact, recovered_tangent


def transport_motor() -> Motor:
    """Composes the xy-plane rotation with the translation by `displacement` into one motor (`[] Rotor+`)."""
    # Each translation generator is an ideal line formed by wedging a coordinate plane with the plane at infinity.
    translation = (stack((mv.xw, mv.yw, mv.zw)) * displacement).sum(axis=0)        # [] Bivector+
    return (translation / 2).exp() * (mv.xy * (turn_angle / 2)).exp()              # [] Rotor+


def motor_transport() -> tuple[DualQuadric, Plane, Point, DualQuadric, Plane, Point]:
    """Carries the reference ellipsoid, its tangent plane, and its contact point by `transport_motor`."""
    motor = transport_motor()                                                      # [] Rotor+
    local = rest_ellipsoid()                                                       # [] Point- <- Plane+
    tangent = support_plane(local, mv.vector(facing))                              # [] Plane+
    contact = local(tangent)                                                       # [] Point-
    world = motor >> local(motor << Plane)                                         # [] Point- <- Plane+
    world_tangent = motor >> tangent                                               # [] Plane+
    return local, tangent, contact, world, world_tangent, world(world_tangent)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.quadrics.quadrics import render
    from examples.util.animation import save_figure

    dual, tangent, contact, recovered_tangent = polar_reciprocity()
    local, rest_tangent, rest_contact, world, world_tangent, world_contact = motor_transport()
    save_figure(render.draw_polar_reciprocity(dual, tangent, contact), "quadrics_polar_reciprocity")
    save_figure(
        render.draw_motor_transport(local, rest_tangent, rest_contact, world, world_tangent, world_contact),
        "quadrics_motor_transport",
    )

    # checks
    basis_planes = stack((mv.x, mv.y, mv.z, mv.w))                                 # [4] Plane+
    basis_points = basis_planes.dual()                                             # [4] Point-
    motor = transport_motor()                                                      # [] Rotor+
    moved_center = world(mv.w)                                                     # [] Point-
    center_offset = (stack((mv.x, mv.y, mv.z)) & moved_center) / (mv.w & moved_center)   # [3] Scalar+
    assert (tangent & contact).abs() < 1e-12
    assert (recovered_tangent & contact).abs() < 1e-12
    assert (((recovered_tangent - tangent) & basis_points).abs() < 1e-12).all()
    assert (world_tangent & world_contact).abs() < 1e-12
    assert ((basis_planes & (world_contact - (motor >> rest_contact))).abs() < 1e-11).all()
    assert ((center_offset - displacement).abs() < 1e-12).all()


if __name__ == "__main__":
    main()
