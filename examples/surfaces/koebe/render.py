"""GPU ray-tracing draws a Koebe polyhedron, its polar dual, and their orthogonal circle packings on the unit sphere."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate
from examples.surfaces.koebe import core
from numga.render import solids
from examples.util.panels import show

primal_colour = np.array(to_rgb("#287fa0"))
dual_colour = np.array(to_rgb("#d06c36"))
sphere_colour, touch_colour = np.full(3, 0.97), np.full(3, 0.25)
background = np.ones(3)
# Spherical camera angles in radians (zero azimuth looks along +y), fractional margin, and panel widths in pixels.
elevation, azimuth = np.radians(20.0), np.radians(30.0)
margin, n_pixels, n_animation_pixels = 0.25, 500, 250
# Edge rod widths, inked circle width, and tangency ball radius in pixels of a panel `n_pixels` wide.
edge_width, circle_width, touch_radius = 2.6, 2.4, 3.5


def gpu_points(points: core.Point) -> solids.Point:
    """Projecting `points` onto spatial offsets from `core.origin` at unit weight embeds them into `solids` (`[...] Point-`)."""
    points = points.batch()
    offsets = (core.mv.w ^ (points & core.origin)) / (core.mv.w & points)   # [...] Point-
    return solids.origin + solids.context.include_from_algebra(offsets)


def polyhedron_view(apexes: core.Point, width_px: int) -> solids.View:
    """Orthographic camera bounds framing `apexes` and the unit sphere across `width_px` horizontal pixels."""
    return solids.view([gpu_points(apexes)], 0.0, elevation, azimuth, margin, width_px)


def polyhedron_panels(topology: core.Topology, circles: core.Plane, view: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """Renders the primal and dual tangent polyhedra on the left and their orthogonal circle packings on the right."""
    V, E, F = topology.n_elements
    scale = view.pixels / n_pixels
    pixel = view.pixel * scale
    vertex_apexes = core.cone_apexes(circles)                               # Point-[V]
    face_apexes = core.cone_apexes(core.face_circles(topology, circles))    # Point-[F]
    tails, heads = topology.incidence[1, 0] * vertex_apexes                 # Point-[E], Point-[E]
    dual_tails, dual_heads = topology.dual.incidence[1, 0] * face_apexes    # Point-[E], Point-[E]
    unit_sphere = solids.ball(solids.origin, 1.0)                           # [] Plane+ <- Point-
    sphere = (unit_sphere[None, None], sphere_colour)
    marks = [
        sphere,
        (solids.rods(gpu_points(tails), gpu_points(heads), edge_width / 2 * pixel), primal_colour),
        (solids.rods(gpu_points(dual_tails), gpu_points(dual_heads), edge_width / 2 * pixel), dual_colour),
        (solids.ball(gpu_points(core.tangency_points(topology, circles)), touch_radius * pixel)[:, None], touch_colour),
    ]
    # Polarity in the unit sphere maps each cone apex directly to the plane cutting its circle on the sphere.
    all_apexes = concatenate([vertex_apexes.batch(), face_apexes.batch()])  # [V + F] Point-
    planes = unit_sphere(gpu_points(all_apexes))                            # [V + F] Plane+
    kinds = np.repeat(np.stack([primal_colour, dual_colour]), [V, F], axis=0)  # [V + F, 3]
    loops = solids.Ink(0, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape), kinds,
                       circle_width * scale, 1.0)
    return (
        solids.draw(marks, view, background, inks=[], translucent=[], meshes=[]),
        solids.draw([sphere], view, background, [loops], translucent=[], meshes=[]),
    )


def polyhedron_figure(topology: core.Topology, circles: core.Plane) -> plt.Figure:
    """Two-panel figure pairing the primal and dual Koebe polyhedra with their vertex and face circles on the sphere."""
    view = polyhedron_view(core.cone_apexes(circles), n_pixels)
    figure = plt.figure(figsize=(2 * n_pixels / 100, view.shape[0] / 100), dpi=100)
    for place, image in zip(((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)), polyhedron_panels(topology, circles, view)):
        show(figure.add_axes(place), image, view)
    return figure


def boost_animation(topology: core.Topology, view: solids.View, states: Iterable[core.Plane]) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the polyhedra and circle packings across streamed `states`."""
    for circles in states:
        yield np.concatenate(polyhedron_panels(topology, circles, view), axis=1)
