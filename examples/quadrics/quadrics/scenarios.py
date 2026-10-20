"""One ellipsoid: a tangent plane and its contact point, each recovered from the other, and the same
ellipsoid carried by a motor."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quadrics.quadrics.core import DualQuadric, Motor, Plane, Point, ellipsoid, mv, support_plane

# The ellipsoid's semi-axes along x, y and z, and its centre.
SEMI_AXES = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
CENTRE = [0.0, 0.0, 0.0]
# The way the tangent plane faces.
FACING = [1.0, 2.0, 3.0]
# Where the motor moves the centre, and how far it turns the ellipsoid in the xy plane before.
DISPLACEMENT = np.array([1.0, 2.0, 3.0])
TURN = -np.pi / 6


# --- math -----------------------------------------------------------------------------
def body() -> DualQuadric:
    """The ellipsoid at rest."""
    return ellipsoid(mv.direction(SEMI_AXES), mv.point(CENTRE))


def polar_reciprocity() -> tuple[DualQuadric, Plane, Point, Plane]:
    """The ellipsoid, its tangent plane facing a direction, the plane's contact point, and the tangent
    plane recovered from the point."""
    dual = body()                                                              # [] Point <- Plane
    tangent = support_plane(dual, mv.vector(FACING + [0.0]), mv.w)             # [] Plane
    contact = dual(tangent)                                                    # [] Point
    return dual, tangent, contact, dual.solve(contact)


def carrier() -> Motor:
    """The motor that turns the ellipsoid in the xy plane, then moves its centre by the displacement."""
    # Each translation is a rotation about a line at infinity, the plane of an axis with w.
    translation = (stack([mv.xw, mv.yw, mv.zw]) * DISPLACEMENT).sum(axis=-1)    # [] Bivector
    return (translation / 2).exp() * (mv.xy * (TURN / 2)).exp()                # [] Motor


def motor_transport() -> tuple[DualQuadric, Plane, Point, DualQuadric, Plane, Point]:
    """The ellipsoid with a tangent plane and its contact point, at rest and carried by a motor: the map
    moved by pulling planes into the body and pushing points back out."""
    motor = carrier()                                                          # [] Motor
    local = body()                                                             # [] Point <- Plane
    tangent = support_plane(local, mv.vector(FACING + [0.0]), mv.w)            # [] Plane
    contact = local(tangent)                                                   # [] Point
    world = motor >> local(motor << Plane)                                     # [] Point <- Plane
    world_tangent = motor >> tangent                                           # [] Plane
    return local, tangent, contact, world, world_tangent, world(world_tangent)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_figure
    from examples.quadrics.quadrics import render

    reciprocity = polar_reciprocity()
    transport = motor_transport()
    save_figure(render.draw_polar_reciprocity(*reciprocity), "quadrics_polar_reciprocity")
    save_figure(render.draw_motor_transport(*transport), "quadrics_motor_transport")

    # checks
    # The tangent plane contains its contact point, and the plane recovered from the point is the tangent
    # plane again. The carried map takes the carried plane to the carried contact point.
    _, tangent, contact, recovered = reciprocity
    np.testing.assert_allclose((tangent & contact).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((recovered & contact).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((recovered ^ tangent).kernel, 0.0, atol=1e-12)
    _, _, contact, _, world_tangent, world_contact = transport
    np.testing.assert_allclose((world_contact & world_tangent).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((world_contact & (carrier() >> contact)).kernel, 0.0, atol=1e-11)


if __name__ == "__main__":
    main()
