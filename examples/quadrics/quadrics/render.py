"""GPU ray-traced rendering for projective ellipsoids, their tangent patches, and contact points."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.render import solids
from examples.quadrics.quadrics.core import DualQuadric, Plane, Point, mv, support_plane
from examples.util.panels import show

ellipsoid_colour = np.array(to_rgb("#0284c7"))
contact_colour = np.array(to_rgb("#e11d48"))
background = np.ones(3)
# Camera elevation and azimuth in radians (azimuth zero looks along +y), framing margin, and panel width in pixels.
elevation, azimuth = np.radians(23.0), np.radians(128.0)
margin, n_pixels = 0.25, 300
# Surface opacities for front and back hits, tangent-plate thickness, and point marker radii in pixels.
ellipsoid_opacity, patch_opacity = 0.19, 0.22
patch_thickness, centre_radius, contact_radius = 1.0, 3.0, 4.0
# Meridian and longitude counts used to sample contact points when framing the camera.
n_latitudes, n_longitudes = 33, 64
# Tangent-patch half-side as a fraction of the ellipsoid's widest coordinate extent, and unit square corner signs.
patch_share = 0.22
square_across, square_up = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
axes = stack((mv.x, mv.y, mv.z))                                             # [3] Plane+
origin = mv.w.dual()                                                         # Point-


def surface_samples(quadric: DualQuadric) -> Point:
    """Samples contact points (`[n_latitudes, n_longitudes] Point-`) over a sphere of tangent normals for camera framing."""
    latitudes = np.linspace(0, np.pi, n_latitudes)[:, None]
    longitudes = np.linspace(0, 2 * np.pi, n_longitudes, endpoint=False)[None, :]
    meridian = (mv.zx * (-latitudes / 2)).exp() >> mv.z                      # [n_latitudes, 1] Plane+
    normals = (mv.xy * (-longitudes / 2)).exp() >> meridian                  # [n_latitudes, n_longitudes] Plane+
    return quadric(support_plane(quadric, normals))                          # [n_latitudes, n_longitudes] Point-


def tangent_patch(quadric: DualQuadric, tangent: Plane, contact: Point) -> Point:
    """Places the four corners (`[4] Point-`) of a square patch on `tangent` centred at `contact`."""
    contact = contact.normalized()                                           # [] Point-
    # Shifting the tangent plane through the origin isolates its normal plane, whose dual is the normal direction.
    normal_plane = tangent - mv.w * (tangent & origin)                       # [] Plane+
    _, _, alignment = solids.aligned(contact, contact + normal_plane.dual()) # [] Rotor+
    across, up = alignment >> axes[:2].dual()                                # [] Point-, Point-
    far = quadric(support_plane(quadric, axes)).normalized()                 # [3] Point-
    near = quadric(support_plane(quadric, -axes)).normalized()               # [3] Point-
    widths = axes & (far - near)                                             # [3] Scalar+
    half_side = patch_share * widths[widths.argmax()]                        # [] Scalar+
    return contact + across * (half_side * square_across) + up * (half_side * square_up)   # [4] Point-


def draw_panels(panels: list[tuple[DualQuadric, Plane, Point]]) -> plt.Figure:
    """Draws one panel per `(quadric, tangent, contact)` triple at a shared camera scale."""
    patches = [tangent_patch(quadric, tangent, contact) for quadric, tangent, contact in panels]
    views = [solids.view([surface_samples(quadric), patch], 0.0, elevation, azimuth, margin, n_pixels)
             for (quadric, _, _), patch in zip(panels, patches)]
    half = np.max([view.half for view in views], axis=0)
    n_panels = len(panels)
    figure = plt.figure(figsize=(n_panels * n_pixels / 100, views[0]._replace(half=half).shape[0] / 100), dpi=100)
    for index, ((quadric, _, contact), patch, view) in enumerate(zip(panels, patches, views)):
        view = view._replace(half=half)
        # Inverting the dual quadric gives the point quadric that pairs each surface point with its tangent plane.
        point_quadric = quadric.inverse()[None, None]                        # [1, 1] Plane+ <- Point-
        plate = solids.plates(patch[None], patch_thickness * view.pixel)     # [1, 6] Plane+
        marks = [(solids.ball(quadric(mv.w), centre_radius * view.pixel)[None, None], ellipsoid_colour),
                 (solids.ball(contact, contact_radius * view.pixel)[None, None], contact_colour)]
        image = solids.draw(marks, view, background,
                            inks=[],
                            translucent=[(point_quadric, ellipsoid_colour, ellipsoid_opacity),
                                         (plate, contact_colour, patch_opacity)],
                            meshes=[])
        show(figure.add_axes((index / n_panels, 0.0, 1 / n_panels, 1.0)), image, view)
    return figure


def draw_polar_reciprocity(dual: DualQuadric, tangent: Plane, contact: Point) -> plt.Figure:
    """Draws the ellipsoid with its tangent patch, contact point, and centre."""
    return draw_panels([(dual, tangent, contact)])


def draw_motor_transport(
    local: DualQuadric, tangent: Plane, contact: Point, world: DualQuadric, world_tangent: Plane, world_contact: Point,
) -> plt.Figure:
    """Draws the ellipsoid and its tangent patch side by side at rest and after motor transport."""
    return draw_panels([(local, tangent, contact), (world, world_tangent, world_contact)])
