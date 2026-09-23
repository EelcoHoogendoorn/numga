"""One ellipsoid: its tangent-contact reciprocity, and the same ellipsoid moved by a motor."""

from __future__ import annotations

import numpy as np

from numga import Extensor
from examples.quadrics.quadrics.core import DualQuadric, Plane, ellipsoid, mv, support_plane


def body() -> DualQuadric:
    """The ellipsoid with semi-axes 3, 2 and 1 along x, y and z, centered at the origin."""
    return ellipsoid(Extensor.stack([mv.yzw * 3, mv.zxw * 2, mv.xyw]), mv.zyx)


def polar_reciprocity():
    """Construct an ellipsoid, then map tangent -> contact -> tangent."""
    dual = body()
    tangent = support_plane(dual, mv.x + mv.y * 2 + mv.z * 3, mv.w)
    contact = dual(tangent)                          # Point <- Plane

    primal = dual.inverse()                          # Plane <- Point
    recovered_tangent = primal(contact)

    # --- checks ---------------------------------------------------------------------------
    # Incidence is reciprocal: tangent & contact = contact & primal(contact) = 0.
    np.testing.assert_allclose((tangent & contact).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((recovered_tangent & contact).to_array(), 0.0, atol=1e-12)
    return dual, recovered_tangent, contact


def motor_transport():
    """Move an ellipsoid and its tangent together by transforming the map."""
    motor = (mv.xw * 0.5 + mv.yw + mv.zw * 1.5).exp() * (mv.xy * (-np.pi / 12)).exp()

    local = body()
    tangent = support_plane(local, mv.x + mv.y * 2 + mv.z * 3, mv.w)
    contact = local(tangent)

    # Pull the input plane into the body frame; push the output point into world.
    world = motor >> local(motor << Plane)
    world_tangent = motor >> tangent
    world_contact = world(world_tangent)             # same point as motor >> contact

    # --- checks ---------------------------------------------------------------------------
    moved = motor >> contact
    joined = world_contact & moved                  # the line through both points vanishes: they coincide
    np.testing.assert_allclose((joined | joined).to_array(), 0.0, atol=1e-20)
    return local, tangent, contact, world, world_tangent, world_contact


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.quadrics.quadrics import render

    save_figure(render.draw_polar_reciprocity(*polar_reciprocity()), "quadrics_polar_reciprocity")
    save_figure(render.draw_motor_transport(*motor_transport()), "quadrics_motor_transport")
