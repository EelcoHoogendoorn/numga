"""Drawing and coordinate read-out for Garland–Heckbert QEM, on the GPU: translucent triangles with their edges as
rods, and the region where a quadric's error stays small traced as the translucent ellipsoid it is.

This module consumes the geometry that `core` produces and turns it into a figure.
It constructs viewport geometry only, never scene geometry, and the mathematics
never imports it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.surfaces.qem.core import Point, Quadric, mv
from examples.wgsl import compose, image_figure, meshes, solids

# Blue: shared flanking ridge triangles (faces 0, 1)
# Red/Rose: steep corner triangles at vertex a (faces 2, 3)
# Purple/Violet: gentle ramp triangles at vertex b (faces 4, 5)
FACE_COLORS = np.array([to_rgb(colour) for colour in ("#0ea5e9", "#0284c7", "#f43f5e", "#e11d48", "#8b5cf6", "#7c3aed")])
A_COLOUR, B_COLOUR, OPTIMUM_COLOUR = (np.array(to_rgb(colour)) for colour in ("#f43f5e", "#8b5cf6", "#10b981"))
EDGE_COLOUR, FAINT_EDGE_COLOUR = np.array(to_rgb("#1e293b")), np.array(to_rgb("#94a3b8"))
VERTEX_COLOUR, ENDPOINT_COLOUR, COLLAPSE_COLOUR = (np.array(to_rgb(colour)) for colour in ("#0f172a", "#475569", "#ef4444"))
# How opaque the triangles are, the faint original mesh and its edges behind the collapsed one, and the ellipsoids.
MESH_OPACITY, FAINT_OPACITY, FAINT_EDGE_OPACITY, ELLIPSOID_OPACITY = 0.5, 0.12, 0.5, 0.4
# The drawn region: where the error exceeds its least by less than this distance squared.
EPSILON = 0.12
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; how far the ellipsoids reach past the
# vertices, the margin as a share of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(24.0), np.radians(138.0)
REACH, MARGIN, N_PIXELS = 0.1, 0.1, 600
# Widths of the edges and of the collapsing edge, the dashes along it, and the radii of the vertices and the
# optimum, in pixels.
EDGE_WIDTH, COLLAPSE_WIDTH, N_DASHES = 2.6, 4.0, 5
VERTEX_RADIUS, OPTIMUM_RADIUS = 6.0, 9.0


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Cartesian coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def error_region(quadric: Quadric, centre: Point) -> Quadric:
    """The points whose error exceeds the centre's by less than EPSILON squared, the centre of unit weight: the
    quadric less that level times the squared weight, negative inside, an ellipsoid about the centre where the
    error is least."""
    level = (quadric(centre) & centre) + EPSILON**2
    return quadric - mv.w * (mv.w & Point) * level


def triangles(vertices: Point, faces: np.ndarray, colours: np.ndarray, view: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """Triangles in their own colours, flat, as a layer."""
    return meshes.layer(vertices, faces, np.zeros(3), meshes.flat(vertices, faces), view, face_values=colours,
                        colour=meshes.by_face)


def edges(vertices: Point, faces: np.ndarray, width: float) -> Quadric:
    """Every edge of the faces once, as rods of a width."""
    ends = np.unique(np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1), axis=-1).reshape(-1, 2), axis=0)
    return solids.rods(vertices[ends[:, 0]], vertices[ends[:, 1]], width / 2)


def vertex_panel(vertices: Point, faces: np.ndarray, vertex: Point, quadric: Quadric, colours: np.ndarray,
                 ellipsoid_colour: np.ndarray, view: solids.View) -> np.ndarray:
    """A vertex with its incident triangles and the ellipsoid where its error stays small."""
    marks = solids.layer([(edges(vertices, faces, EDGE_WIDTH * view.pixel), EDGE_COLOUR),
                          (solids.ball(vertex, VERTEX_RADIUS * view.pixel)[None, None], VERTEX_COLOUR)], view)
    ellipsoid = solids.layer([(error_region(quadric, vertex)[None, None], ellipsoid_colour)], view)
    return compose([(*triangles(vertices, faces, colours, view), MESH_OPACITY), (*marks, 1.0),
                    (*ellipsoid, ELLIPSOID_OPACITY)], background)


def collapse_panel(vertices: Point, faces: np.ndarray, collapsed: Point, surviving: np.ndarray, quadric: Quadric,
                   optimum: Point, view: solids.View) -> np.ndarray:
    """The collapsed mesh about the optimal vertex and its ellipsoid, the contracting edge dashed between its
    endpoints, and the original mesh faint behind."""
    pixel = view.pixel
    # The edge between the endpoints cut into equal pieces, one fewer than twice the dashes, every other a dash.
    share = np.linspace(0, 1, 2 * N_DASHES).reshape(N_DASHES, 2)                # [dashes, 2]
    dashes = vertices[0] * (1 - share) + vertices[1] * share                    # [dashes, 2] Point
    faint = solids.layer([(edges(vertices, faces, EDGE_WIDTH * pixel), FAINT_EDGE_COLOUR)], view)
    marks = solids.layer([(edges(collapsed, faces[surviving], EDGE_WIDTH * pixel), EDGE_COLOUR),
                          (solids.rods(dashes[:, 0], dashes[:, 1], COLLAPSE_WIDTH / 2 * pixel), COLLAPSE_COLOUR),
                          (solids.ball(vertices[:2], VERTEX_RADIUS * pixel)[:, None], ENDPOINT_COLOUR),
                          (solids.ball(optimum, OPTIMUM_RADIUS * pixel)[None, None], OPTIMUM_COLOUR)], view)
    ellipsoid = solids.layer([(error_region(quadric, optimum)[None, None], OPTIMUM_COLOUR)], view)
    return compose([(*triangles(vertices, faces, FACE_COLORS, view), FAINT_OPACITY), (*faint, FAINT_EDGE_OPACITY),
                    (*triangles(collapsed, faces[surviving], FACE_COLORS[surviving], view), MESH_OPACITY),
                    (*marks, 1.0), (*ellipsoid, ELLIPSOID_OPACITY)], background)


def draw_qem(
    vertices: Point,
    faces: np.ndarray,
    incident_a: np.ndarray,
    incident_b: np.ndarray,
    quadric_a: Quadric,
    quadric_b: Quadric,
    edge_quadric: Quadric,
    optimum: Point,
    collapsed: Point,
    surviving: np.ndarray,
) -> plt.Figure:
    """Both endpoint neighbourhoods with their error ellipsoids, and the collapsed mesh, seen by one camera."""
    view = solids.view([vertices], REACH, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    return image_figure(np.concatenate([
        vertex_panel(vertices, faces[incident_a], vertices[0], quadric_a, FACE_COLORS[incident_a], A_COLOUR, view),
        vertex_panel(vertices, faces[incident_b], vertices[1], quadric_b, FACE_COLORS[incident_b], B_COLOUR, view),
        collapse_panel(vertices, faces, collapsed, surviving, edge_quadric, optimum, view),
    ], axis=1))
