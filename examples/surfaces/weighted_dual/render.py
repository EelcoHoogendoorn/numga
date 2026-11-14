"""GPU ray tracing renders the polyhedron with its polar dual in space alongside their spherical projections."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.surfaces.weighted_dual import core
from examples.util.panels import image_figure
from numga import stack
from numga.render import simplices, solids
from numga.topology import Topology

sm = solids.mv
primal_colour = np.array(to_rgb("#287fa0"))         # [3]
dual_colour = np.array(to_rgb("#d06c36"))           # [3]
hull_colour = np.array(to_rgb("#b4d4e0"))           # [3]
ray_colour = np.full(3, 0.55)                       # [3]
sphere_colour = np.full(3, 0.97)                    # [3]
background_colour = np.ones(3)                      # [3]
# Each tetrahedral cone joins vertex 0 at the centre to one triangle's three corners (1, 2, 3).
cone_faces = np.array([[1, 2, 3], [0, 1, 2], [0, 2, 3], [0, 3, 1]])            # [4, 3]
distance_cmap = "coolwarm"
distance_range = 0.1
ray_span = 0.15
elevation, azimuth = np.radians(20.0), np.radians(30.0)
half_width, n_pixels = 1.3, 300
edge_width, ray_width, arc_width, vertex_radius = 1.3, 1.0, 1.4, 3.0


def great_arcs(starts: solids.Point, ends: solids.Point) -> tuple[solids.Quadric, solids.Quadric]:
    """Pairs of half-spaces (`[...] Plane+ <- Point-`) ink the shorter great-circle arcs between unit-sphere endpoints."""
    # Form the chord midpoint direction and the plane orthogonal to it through the start point facing the centre.
    midpoints = (starts - solids.origin) + (ends - solids.origin)              # [...] Point-
    chord_planes = midpoints.dual() - sm.w * (midpoints.dual() & starts)       # [...] Plane+
    return solids.half_space((solids.origin & starts & ends) * solids.flip), solids.half_space(chord_planes)


def complex_figure(topology: Topology, points: core.Point, duals: core.Point) -> plt.Figure:
    """Wraps the side-by-side spatial and spherical ray-traced panels into a matplotlib `Figure`."""
    return image_figure(complex_image(topology, points, duals))


def complex_image(topology: Topology, points: core.Point, duals: core.Point) -> np.ndarray:
    """Renders the spatial polyhedron and polar dual beside their great-circle projections (`[n_pixels, 2 * n_pixels, 3]`)."""
    V, E, F = topology.n_elements
    view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, half_width), n_pixels)
    pixel = view.pixel
    sphere = solids.ball(solids.origin, 1.0)[None, None]                       # [1, 1] Plane+ <- Point-
    vertices = solids.origin + solids.context.include_from_algebra(points.batch())  # [V] Point-
    # Assemble one tetrahedral cone from the centre over each triangle so folded and convex polyhedra alike render solid.
    corners = solids.origin + solids.context.include_from_algebra((topology.incidence[2, 0] * points).batch())  # [3, F] Point-
    cones = stack([solids.origin.broadcast_to((F,)), *corners], axis=-1)       # [F, 4] Point-
    hull_cones = solids.polyhedra(cones, cone_faces)                           # [F, 4] Plane+
    # Colour each vertex by its radial offset from the unit sphere, from blue inside to red outside.
    distances = points.norm().batch().to_array()                               # [V]
    rays = points.normalized()                                                 # Point-[V]
    ray_ends = solids.origin + solids.context.include_from_algebra((topology.incidence[1, 0] * rays).batch())  # [2, E] Point-
    colours = plt.get_cmap(distance_cmap)((distances - 1 + distance_range) / (2 * distance_range))[:, :3]  # [V, 3]
    # Draw each dual edge when at least one of the two dual faces it bounds has its primal ray facing the camera.
    camera_plane = view.camera >> sm.z                                         # [] Plane+
    front_edges = ((camera_plane & ray_ends) > 0).any(axis=0)                  # [E]
    tails, heads = solids.origin + solids.context.include_from_algebra((topology.incidence[1, 0] * points).batch())  # [E] Point-, [E] Point-
    dual_ends = topology.dual.incidence[1, 0]                                  # [2] Scalar+[E <- F]
    dual_tails, dual_heads = solids.origin + solids.context.include_from_algebra((dual_ends * duals).batch())  # [E] Point-, [E] Point-
    primal_rods, primal_joints = solids.edges(tails, heads, edge_width / 2 * pixel)
    dual_rods, dual_joints = solids.edges(dual_tails[front_edges], dual_heads[front_edges], edge_width / 2 * pixel)
    ray_offsets = solids.context.include_from_algebra(rays.batch())            # [V] Point-
    ray_spans = stack(
        (solids.origin + ray_offsets * (1 - ray_span), solids.origin + ray_offsets * (1 + ray_span)),
        axis=-1,
    )                                                                          # [V, 2] Point-
    ray_strokes = simplices.tubes(
        ray_spans.reshape(-1), Topology.chains([2] * V), ray_colour, ray_width, view, solids.gloss
    )
    parts = [
        (hull_cones, hull_colour),
        (primal_rods, primal_colour),
        (primal_joints, primal_colour),
        (dual_rods, dual_colour),
        (dual_joints, dual_colour),
        (solids.ball(vertices, vertex_radius * pixel)[:, None], colours),
    ]
    space_panel = solids.draw(parts, view, background_colour, inks=[], translucent=[], meshes=[(ray_strokes, 1.0)])
    # Ink both primal and normalized dual great-circle arcs directly onto the unit sphere in the right panel.
    primal_arcs = great_arcs(*ray_ends)
    dual_ray_ends = solids.origin + solids.context.include_from_algebra((dual_ends * duals.normalized()).batch())  # [2, E] Point-
    dual_arcs = great_arcs(*dual_ray_ends)
    inks = [
        solids.Ink(0, *primal_arcs, primal_colour, arc_width, 1.0),
        solids.Ink(0, *dual_arcs, dual_colour, arc_width, 1.0),
    ]
    sphere_panel = solids.draw([(sphere, sphere_colour)], view, background_colour, inks, translucent=[], meshes=[])
    return np.concatenate([space_panel, sphere_panel], axis=1)                 # [n_pixels, 2 * n_pixels, 3]


def complex_frames(states: Iterable[tuple[Topology, core.Point, core.Point]]) -> Iterator[np.ndarray]:
    """Yields side-by-side rendered frames (`[n_pixels, 2 * n_pixels, 3]`) from a stream of `(topology, points, duals)` states."""
    for geometry in states:
        yield complex_image(*geometry)
