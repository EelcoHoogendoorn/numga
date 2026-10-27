"""Drawing and coordinate readout for the ellipsoid example, on the GPU: the ellipsoid traced as the quadric it is,
translucent, the tangent plane a patch about its contact point, and the points balls."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.quadrics.quadrics.core import DualQuadric, Plane, Point, ga, mv, support_plane
from examples.util.wgsl import show, solids

ELLIPSOID = np.array(to_rgb("#0284c7"))
CONTACT = np.array(to_rgb("#e11d48"))
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the ellipsoids as a
# share of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(23.0), np.radians(128.0)
MARGIN, N_PIXELS = 0.25, 700
# The opacities of each surface of the ellipsoid and the tangent patch, front and back alike; the thickness of the
# patch, and the radii of the centre and the contact point, in pixels.
ELLIPSOID_OPACITY, PATCH_OPACITY = 0.19, 0.22
PATCH_THICKNESS, CENTRE_RADIUS, CONTACT_RADIUS = 1.0, 6.0, 8.0


# --- plumbing -------------------------------------------------------------------------
def surface(quadric: DualQuadric) -> Point:
    """Sample the ellipsoid by sweeping its tangent-plane normal over a sphere."""
    longitude = np.linspace(0, 2 * np.pi, 65)[None, :]
    latitude = np.linspace(0, np.pi, 33)[:, None]
    normals = (mv.x * (np.sin(latitude) * np.cos(longitude))
               + mv.y * (np.sin(latitude) * np.sin(longitude))
               + mv.z * np.cos(latitude))
    return quadric(support_plane(quadric, normals, mv.w))


def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 3]`: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def tangent_patch(quadric: DualQuadric, tangent: Plane, contact: Point) -> np.ndarray:
    """The corners `[4, 3]` of a square piece of the tangent plane about the contact point, its side in proportion
    to the ellipsoid."""
    contact_xyz = euclidean(contact)
    # A basis of the plane's normal complement also handles vertical planes.
    normal = tangent.cast(ga.subspace("x y z w")).kernel[..., :3]
    _, _, frame = np.linalg.svd(normal[None, :])
    extent = np.ptp(euclidean(surface(quadric)).reshape(-1, 3), axis=0).max()
    square = 0.22 * extent * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return contact_xyz + square[:, :1] * frame[1] + square[:, 1:] * frame[2]


def draw_panels(panels: list) -> plt.Figure:
    """Draw each ellipsoid with a tangent patch, its contact point and its centre, each panel framing its own
    ellipsoid at one scale for all."""
    patches = [tangent_patch(quadric, tangent, contact) for quadric, tangent, contact in panels]
    views = [solids.view([solids.mv.point(euclidean(surface(quadric))), solids.mv.point(patch)], 0.0, ELEVATION,
                         AZIMUTH, MARGIN, N_PIXELS) for (quadric, _, _), patch in zip(panels, patches)]
    half = np.max([view.half for view in views], axis=0)
    figure = plt.figure(figsize=(len(panels) * N_PIXELS / 100, views[0]._replace(half=half).shape[0] / 100), dpi=100)
    for index, ((quadric, _, contact), patch, view) in enumerate(zip(panels, patches, views)):
        view = view._replace(half=half)
        # The ellipsoid's point quadric, which pairs each point with its polar plane: the inverse of the dual quadric.
        body = quadric.inverse()[None, None]                                  # [1, 1] Plane+ <- Point-
        plate = solids.plates(solids.mv.point(patch)[None], PATCH_THICKNESS * view.pixel)   # [1, 6] Plane+
        marks = [(solids.ball(solids.mv.point(euclidean(quadric(mv.w))), CENTRE_RADIUS * view.pixel)[None, None],
                  ELLIPSOID),
                 (solids.ball(solids.mv.point(euclidean(contact)), CONTACT_RADIUS * view.pixel)[None, None], CONTACT)]
        image = solids.draw(marks, view, background,
                            inks=[],
                            translucent=[(body, ELLIPSOID, ELLIPSOID_OPACITY), (plate, CONTACT, PATCH_OPACITY)],
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
