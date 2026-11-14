"""Render QEM edge-collapse neighbourhoods, implicit error ellipsoids, and progressive surface simplification on the GPU.

This module consumes the geometry that `core` produces and turns it into a figure. It constructs viewport geometry only, never scene geometry, and the mathematics never imports it.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, stack
from examples.surfaces.qem.core import Point, Topology, mv
from numga.render import meshes, solids
from examples.util.panels import image_figure

# Blue/rose/violet face colours, endpoint and optimum colours, edge colours, and vertex colours.
face_colors = np.array([to_rgb(colour) for colour in ("#0ea5e9", "#0284c7", "#f43f5e", "#e11d48", "#8b5cf6", "#7c3aed")])
a_colour, b_colour, optimum_colour = (np.array(to_rgb(colour)) for colour in ("#f43f5e", "#8b5cf6", "#10b981"))
edge_colour, faint_edge_colour = np.array(to_rgb("#1e293b")), np.array(to_rgb("#94a3b8"))
vertex_colour, endpoint_colour, collapse_colour = (np.array(to_rgb(colour)) for colour in ("#0f172a", "#475569", "#ef4444"))
# Per-layer opacities for triangles, faint reference surface, faint reference edges, and error ellipsoids.
surface_opacity, faint_opacity, faint_edge_opacity, ellipsoid_opacity = 0.29, 0.06, 0.29, 0.23
# Level-set offset above the minimum error where the implicit error ellipsoid is drawn.
epsilon = 0.12
background = np.ones(3)
# Viewport elevation, azimuth, reach beyond vertices, margin fraction, and panel pixel size.
elevation, azimuth = np.radians(24.0), np.radians(138.0)
reach, margin, n_pixels = 0.1, 0.1, 300
# Pixel widths, dash count, marker radii, and triangle plate thickness.
edge_width, collapse_width, n_dashes = 1.3, 2.0, 5
vertex_radius, optimum_radius = 3.0, 4.5
triangle_thickness = 0.25
# Fixed three-quarter camera parameters for flat-shaded surface simplification.
bunny_elevation, bunny_azimuth = np.radians(24.0), np.radians(54.0)
bunny_margin, bunny_pixels, bunny_gloss = 0.05, 300, 0.18
surface_colour = np.array(to_rgb("#8fb3c9"))


# --- plumbing -------------------------------------------------------------------------
def edges(topology: Topology, vertices: Point, width: float) -> Extensor:
    """Build a rod quadric along each edge of `topology`."""
    tails, heads = (topology.incidence[1, 0] * vertices).batch()                # [E] Point-, Point-
    return solids.rods(tails, heads, width / 2)


def plates(topology: Topology, vertices: Point, thickness: float) -> Extensor:
    """Build a thin plate on each triangle of `topology`."""
    return solids.plates(stack(tuple((topology.incidence[2, 0] * vertices).batch()), axis=-1), thickness)


def vertex_panel(star: Topology, vertices: Point, vertex: Point, quadric: Extensor, colours: np.ndarray,
                 ellipsoid_colour: np.ndarray, view: solids.View) -> np.ndarray:
    """Draw a vertex with its incident triangle star and error level-set ellipsoid."""
    pixel = view.pixel
    level = (quadric(vertex) & vertex) + epsilon**2                             # [] Scalar+
    ellipsoid = quadric - mv.w * (mv.w & Point) * level                         # [] Plane+ <- Point-
    marks = [(edges(star, vertices, edge_width * pixel), edge_colour),
             (solids.ball(vertex, vertex_radius * pixel)[None, None], vertex_colour)]
    translucent = [(plates(star, vertices, triangle_thickness * pixel), colours, surface_opacity),
                   (ellipsoid[None, None], ellipsoid_colour, ellipsoid_opacity)]
    return solids.draw(marks, view, background, inks=[], translucent=translucent, meshes=[])


def collapse_panel(ridge: Topology, vertices: Point, collapsed: Topology, collapsed_vertices: Point, surviving_faces: np.ndarray,
                   quadric: Extensor, optimum: Point, view: solids.View) -> np.ndarray:
    """Draw the contracted patch around `optimum` and its error ellipsoid over the faint initial patch."""
    pixel = view.pixel
    endpoints = vertices.batch()[:2]                                            # [2] Point-
    # Subdivide the segment between the two endpoints into alternating dash intervals.
    share = np.linspace(0, 1, 2 * n_dashes).reshape(n_dashes, 2)                # [n_dashes, 2]
    dashes = endpoints[0] * (1 - share) + endpoints[1] * share                  # [n_dashes, 2] Point-
    level = (quadric(optimum) & optimum) + epsilon**2                           # [] Scalar+
    ellipsoid = quadric - mv.w * (mv.w & Point) * level                         # [] Plane+ <- Point-
    marks = [(edges(collapsed, collapsed_vertices, edge_width * pixel), edge_colour),
             (solids.rods(dashes[:, 0], dashes[:, 1], collapse_width / 2 * pixel), collapse_colour),
             (solids.ball(endpoints, vertex_radius * pixel)[:, None], endpoint_colour),
             (solids.ball(optimum, optimum_radius * pixel)[None, None], optimum_colour)]
    translucent = [(plates(ridge, vertices, triangle_thickness * pixel), face_colors, faint_opacity),
                   (edges(ridge, vertices, edge_width * pixel), faint_edge_colour, faint_edge_opacity),
                   (plates(collapsed, collapsed_vertices, triangle_thickness * pixel),
                    face_colors[surviving_faces], surface_opacity),
                   (ellipsoid[None, None], optimum_colour, ellipsoid_opacity)]
    return solids.draw(marks, view, background, inks=[], translucent=translucent, meshes=[])


def draw_qem(
    ridge: Topology,
    vertices: Point,
    incident_a: np.ndarray,
    incident_b: np.ndarray,
    quadric_a: Extensor,
    quadric_b: Extensor,
    edge_quadric: Extensor,
    optimum: Point,
    collapsed_vertices: Point,
    surviving_faces: np.ndarray,
) -> plt.Figure:
    """Render both endpoint neighbourhoods and their error ellipsoids beside the contracted patch."""
    star_a, star_b, collapsed = (ridge.subset(shown, -1) for shown in (incident_a, incident_b, surviving_faces))
    vertex_a, vertex_b = vertices.batch()[:2]                                   # [] Point-, Point-
    view = solids.view([vertices.batch()], reach, elevation, azimuth, margin, n_pixels)
    return image_figure(np.concatenate([
        vertex_panel(star_a, vertices, vertex_a, quadric_a, face_colors[incident_a], a_colour, view),
        vertex_panel(star_b, vertices, vertex_b, quadric_b, face_colors[incident_b], b_colour, view),
        collapse_panel(ridge, vertices, collapsed, collapsed_vertices, surviving_faces, edge_quadric, optimum, view),
    ], axis=1))


def animate_surface(reference: Point, states: Iterator[tuple[Topology, Point]]) -> Iterator[np.ndarray]:
    """Render progressive surface simplification from a camera framed on `reference`."""
    view = solids.view([reference.batch()], 0.0, bunny_elevation, bunny_azimuth, bunny_margin, bunny_pixels).squared()
    for topology, vertices in states:
        places = vertices.batch()                                               # [V] Point-
        triangles = topology.elements[2]                                        # [F, 3]
        yield meshes.draw(places, triangles, surface_colour, meshes.flat(places, triangles), view, background,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=bunny_gloss)
