"""Render QEM edge-collapse neighbourhoods, implicit error ellipsoids, and progressive mesh simplification on the GPU.

This module consumes the geometry that `core` produces and turns it into a figure.
It constructs viewport geometry only, never scene geometry, and the mathematics
never imports it.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor
from examples.surfaces.qem.core import Point, mv
from examples.util.wgsl import image_figure, meshes, solids

# Blue: shared flanking ridge triangles (faces 0, 1)
# Red/Rose: steep corner triangles at vertex a (faces 2, 3)
# Purple/Violet: gentle ramp triangles at vertex b (faces 4, 5)
FACE_COLORS = np.array([to_rgb(colour) for colour in ("#0ea5e9", "#0284c7", "#f43f5e", "#e11d48", "#8b5cf6", "#7c3aed")])
A_COLOUR, B_COLOUR, OPTIMUM_COLOUR = (np.array(to_rgb(colour)) for colour in ("#f43f5e", "#8b5cf6", "#10b981"))
EDGE_COLOUR, FAINT_EDGE_COLOUR = np.array(to_rgb("#1e293b")), np.array(to_rgb("#94a3b8"))
VERTEX_COLOUR, ENDPOINT_COLOUR, COLLAPSE_COLOUR = (np.array(to_rgb(colour)) for colour in ("#0f172a", "#475569", "#ef4444"))
# How much each surface hides, a ray passing through two of each: of the triangles, of the faint initial mesh and
# its edges behind the collapsed one, and of the ellipsoids.
MESH_OPACITY, FAINT_OPACITY, FAINT_EDGE_OPACITY, ELLIPSOID_OPACITY = 0.29, 0.06, 0.29, 0.23
# The drawn region: where the error exceeds its least by less than this distance squared.
EPSILON = 0.12
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; how far the ellipsoids reach past the
# vertices, the margin as a share of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(24.0), np.radians(138.0)
REACH, MARGIN, N_PIXELS = 0.1, 0.1, 300
# Widths of the edges and of the collapsing edge, the dashes along it, the radii of the vertices and the
# optimum, and the triangles' thickness, in pixels.
EDGE_WIDTH, COLLAPSE_WIDTH, N_DASHES = 1.3, 2.0, 5
VERTEX_RADIUS, OPTIMUM_RADIUS = 3.0, 4.5
TRIANGLE_THICKNESS = 0.25
# A fixed three-quarter view of the whole surface, with each surviving triangle lit separately.
MESH_ELEVATION, MESH_AZIMUTH = np.radians(24.0), np.radians(54.0)
MESH_MARGIN, MESH_PIXELS, MESH_GLOSS = 0.05, 300, 0.18
SURFACE_COLOUR = np.array(to_rgb("#8fb3c9"))


# --- plumbing -------------------------------------------------------------------------
def edges(vertices: Point, faces: np.ndarray, width: float) -> Extensor:
    """Build a rod quadric for each unique edge of `faces`."""
    directed_edges = np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1).reshape(-1, 2)
    ends = np.unique(np.sort(directed_edges, axis=-1), axis=0)
    return solids.rods(vertices[ends[:, 0]], vertices[ends[:, 1]], width / 2)


def vertex_panel(vertices: Point, faces: np.ndarray, vertex: Point, quadric: Extensor, colours: np.ndarray,
                 ellipsoid_colour: np.ndarray, view: solids.View) -> np.ndarray:
    """Draw a vertex with its incident triangles and the ellipsoid where its error stays small."""
    pixel = view.pixel
    level = (quadric(vertex) & vertex) + EPSILON**2                             # [] Scalar+
    ellipsoid = quadric - mv.w * (mv.w & Point) * level                         # [] Plane+ <- Point-
    marks = [(edges(vertices, faces, EDGE_WIDTH * pixel), EDGE_COLOUR),
             (solids.ball(vertex, VERTEX_RADIUS * pixel)[None, None], VERTEX_COLOUR)]
    translucent = [(solids.plates(vertices[faces], TRIANGLE_THICKNESS * pixel), colours, MESH_OPACITY),
                   (ellipsoid[None, None], ellipsoid_colour, ELLIPSOID_OPACITY)]
    return solids.draw(marks, view, background, inks=[], translucent=translucent, meshes=[])


def collapse_panel(vertices: Point, faces: np.ndarray, collapsed_vertices: Point, surviving_faces: np.ndarray,
                   quadric: Extensor, optimum: Point, view: solids.View) -> np.ndarray:
    """Draw the collapsed mesh around the optimal vertex and its ellipsoid, with the initial mesh faint behind."""
    pixel = view.pixel
    # The edge between the endpoints cut into equal pieces, one fewer than twice the dashes, every other a dash.
    share = np.linspace(0, 1, 2 * N_DASHES).reshape(N_DASHES, 2)                # [dashes, 2]
    dashes = vertices[0] * (1 - share) + vertices[1] * share                    # [dashes, 2] Point-
    level = (quadric(optimum) & optimum) + EPSILON**2                           # [] Scalar+
    ellipsoid = quadric - mv.w * (mv.w & Point) * level                         # [] Plane+ <- Point-
    marks = [(edges(collapsed_vertices, faces[surviving_faces], EDGE_WIDTH * pixel), EDGE_COLOUR),
             (solids.rods(dashes[:, 0], dashes[:, 1], COLLAPSE_WIDTH / 2 * pixel), COLLAPSE_COLOUR),
             (solids.ball(vertices[:2], VERTEX_RADIUS * pixel)[:, None], ENDPOINT_COLOUR),
             (solids.ball(optimum, OPTIMUM_RADIUS * pixel)[None, None], OPTIMUM_COLOUR)]
    translucent = [(solids.plates(vertices[faces], TRIANGLE_THICKNESS * pixel), FACE_COLORS, FAINT_OPACITY),
                   (edges(vertices, faces, EDGE_WIDTH * pixel), FAINT_EDGE_COLOUR, FAINT_EDGE_OPACITY),
                   (solids.plates(collapsed_vertices[faces[surviving_faces]], TRIANGLE_THICKNESS * pixel),
                    FACE_COLORS[surviving_faces], MESH_OPACITY),
                   (ellipsoid[None, None], OPTIMUM_COLOUR, ELLIPSOID_OPACITY)]
    return solids.draw(marks, view, background, inks=[], translucent=translucent, meshes=[])


def draw_qem(
    vertices: Point,
    faces: np.ndarray,
    incident_a: np.ndarray,
    incident_b: np.ndarray,
    quadric_a: Extensor,
    quadric_b: Extensor,
    edge_quadric: Extensor,
    optimum: Point,
    collapsed_vertices: Point,
    surviving_faces: np.ndarray,
) -> plt.Figure:
    """Render both endpoint neighbourhoods with their error ellipsoids alongside the collapsed mesh."""
    view = solids.view([vertices], REACH, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    return image_figure(np.concatenate([
        vertex_panel(vertices, faces[incident_a], vertices[0], quadric_a, FACE_COLORS[incident_a], A_COLOUR, view),
        vertex_panel(vertices, faces[incident_b], vertices[1], quadric_b, FACE_COLORS[incident_b], B_COLOUR, view),
        collapse_panel(vertices, faces, collapsed_vertices, surviving_faces, edge_quadric, optimum, view),
    ], axis=1))


def animate_mesh(states: Iterator[tuple[Point, np.ndarray]], reference: Point) -> Iterator[np.ndarray]:
    """Render progressive mesh simplification from a camera fixed by the reference surface."""
    view = solids.view([reference], 0.0, MESH_ELEVATION, MESH_AZIMUTH, MESH_MARGIN, MESH_PIXELS).squared()
    for vertices, faces in states:
        yield meshes.draw(vertices, faces, SURFACE_COLOUR, meshes.flat(vertices, faces), view, background,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=MESH_GLOSS)
