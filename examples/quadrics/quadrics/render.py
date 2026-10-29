"""Drawing and coordinate readout for the ellipsoid example, on the GPU: the ellipsoid traced as the quadric it is,
translucent, the tangent plane a patch about its contact point, and the points balls."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.quadrics.quadrics.core import DualQuadric, Plane, Point, mv, support_plane
from examples.util.wgsl import show, solids

ellipsoid_colour = np.array(to_rgb("#0284c7"))
contact_colour = np.array(to_rgb("#e11d48"))
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the ellipsoids as a
# share of the larger side, and the pixels across each panel.
elevation, azimuth = np.radians(23.0), np.radians(128.0)
margin, n_pixels = 0.25, 300
# The opacities of each surface of the ellipsoid and the tangent patch, front and back alike; the thickness of the
# patch, and the radii of the centre and the contact point, in pixels.
ellipsoid_opacity, patch_opacity = 0.19, 0.22
patch_thickness, centre_radius, contact_radius = 1.0, 3.0, 4.0
# The tangent normals that sample the ellipsoid for framing, along each meridian and around the pole.
n_latitudes, n_longitudes = 33, 65
# The patch's half side, as a share of the ellipsoid's widest extent along x, y and z, and its corners in its plane.
patch_share = 0.22
square_across, square_up = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
axes = stack((mv.x, mv.y, mv.z))                                             # [3] Plane+
origin = mv.w.dual()                                                         # [] Point-


# --- plumbing -------------------------------------------------------------------------
def surface(quadric: DualQuadric) -> Point:
    """The ellipsoid sampled by its contact points, the tangent normal swept over a sphere: z turned towards x by the
    latitude, then about z by the longitude."""
    latitudes = np.linspace(0, np.pi, n_latitudes)[:, None]
    longitudes = np.linspace(0, 2 * np.pi, n_longitudes)[None, :]
    meridian = (mv.zx * (-latitudes / 2)).exp() >> mv.z                      # [latitudes, 1] Plane+
    normals = (mv.xy * (-longitudes / 2)).exp() >> meridian                  # [latitudes, longitudes] Plane+
    return quadric(support_plane(quadric, normals, mv.w))


def tangent_patch(quadric: DualQuadric, tangent: Plane, contact: Point) -> Point:
    """The corners of a square piece of the tangent plane about the contact point, its side in proportion to the
    ellipsoid, along the directions of x and y turned as z is turned onto the plane's normal."""
    contact = contact.normalized()                                           # [] Point-
    # The tangent plane moved through the origin; its dual is the direction square to it.
    square = tangent - mv.w * (tangent & origin)                             # [] Plane+
    _, _, turn = solids.aligned(contact, contact + square.dual())            # [] Motor+
    across, up = turn >> axes[:2].dual()                                     # [] Point-, Point-
    # The ellipsoid's widths along x, y and z lie between its contact points with the tangent planes facing either way.
    far, near = (quadric(support_plane(quadric, facing, mv.w)).normalized() for facing in (axes, -axes))   # [3] Point-
    widths = axes & (far - near)                                             # [3] Scalar+
    half_side = patch_share * widths[widths.argmax()]                        # [] Scalar+
    return contact + across * (half_side * square_across) + up * (half_side * square_up)   # [corners] Point-


def draw_panels(panels: list) -> plt.Figure:
    """Draw each ellipsoid with a tangent patch, its contact point and its centre, each panel framing its own
    ellipsoid at one scale for all."""
    patches = [tangent_patch(quadric, tangent, contact) for quadric, tangent, contact in panels]
    views = [solids.view([surface(quadric), patch], 0.0, elevation, azimuth, margin, n_pixels)
             for (quadric, _, _), patch in zip(panels, patches)]
    half = np.max([view.half for view in views], axis=0)
    figure = plt.figure(figsize=(len(panels) * n_pixels / 100, views[0]._replace(half=half).shape[0] / 100), dpi=100)
    for index, ((quadric, _, contact), patch, view) in enumerate(zip(panels, patches, views)):
        view = view._replace(half=half)
        # The ellipsoid's point quadric, which pairs each point with its polar plane: the inverse of the dual quadric.
        body = quadric.inverse()[None, None]                                  # [1, 1] Plane+ <- Point-
        plate = solids.plates(patch[None], patch_thickness * view.pixel)      # [1, 6] Plane+
        # The centre is the pole of the plane at infinity.
        marks = [(solids.ball(quadric(mv.w), centre_radius * view.pixel)[None, None], ellipsoid_colour),
                 (solids.ball(contact, contact_radius * view.pixel)[None, None], contact_colour)]
        image = solids.draw(marks, view, background,
                            inks=[],
                            translucent=[(body, ellipsoid_colour, ellipsoid_opacity),
                                         (plate, contact_colour, patch_opacity)],
                            meshes=[])
        show(figure.add_axes((index / len(panels), 0.0, 1 / len(panels), 1.0)), image, view)
    return figure


def draw_polar_reciprocity(dual: DualQuadric, tangent: Plane, contact: Point, recovered: Plane) -> plt.Figure:
    """The ellipsoid with its tangent plane, drawn as a patch about the contact point, and its centre."""
    return draw_panels([(dual, tangent, contact)])


def draw_motor_transport(
    local: DualQuadric, tangent: Plane, contact: Point, world: DualQuadric, world_tangent: Plane, world_contact: Point,
) -> plt.Figure:
    """The ellipsoid with its tangent plane at rest, and carried by the motor."""
    return draw_panels([(local, tangent, contact), (world, world_tangent, world_contact)])
