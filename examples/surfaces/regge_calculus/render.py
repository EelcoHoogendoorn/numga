"""The drawings show Regge edge metric cones, metric ellipses, vertex curvature defects, and intrinsic Laplace–Beltrami flow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import stack
from examples.util import charts
from examples.util.wgsl import contours, flow, solids
from examples.surfaces.regge_calculus import core

primal_colour = "#286580"
accent_colour = "#d77542"
radial_ellipse_colour = "#c84b31"
hoop_ellipse_colour = "#2d6a9f"
mesh_colour = "#b8c4c9"
wall_colour = "#52616a"
flow_colour = "#173c4b"
hat_colours = LinearSegmentedColormap.from_list(
    "regge_metric", ("#f6f4ee", "#8fb8c9", "#286580"))
diverging_colours = LinearSegmentedColormap.from_list(
    "regge_diverging", ("#286580", "#f5f2eb", "#d77542"))
indicatrix_colours = LinearSegmentedColormap.from_list(
    "regge_indicatrix", (hoop_ellipse_colour, wall_colour, radial_ellipse_colour))
n_contours = 7
n_ellipse_grid = 25
# The pixel sizes of a row of four triangles and of a row of three disks.
row_size, disk_row_size = (780, 195), (780, 260)


# --- plumbing -------------------------------------------------------------------------
def plane_places(points: core.Point) -> np.ndarray:
    """x and y of intrinsic positions and directions `[...]` on the last axis, for matplotlib's artists: their
    pairings with the planes x and y."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def frame_panel(ax: plt.Axes, places: np.ndarray, margin_ratio: float) -> None:
    """Square equal-aspect viewport around `places` with no axes or ticks."""
    low, high = places.min(axis=0), places.max(axis=0)
    centre = (low + high) / 2
    half_span = (1 + margin_ratio) * (high - low).max() / 2
    ax.set(xlim=(centre[0] - half_span, centre[0] + half_span),
           ylim=(centre[1] - half_span, centre[1] + half_span), aspect="equal")
    ax.set_axis_off()


def spaced_faces(centres: core.Point, max_radius: float, min_separation: float) -> list[int]:
    """Well-spaced interior face indices ordered from the origin outward: each face's centre `Point-[faces]` within
    `max_radius` of the origin and further than `min_separation` from those of the faces chosen before it."""
    centres = centres.batch()                                                  # [faces] Point-
    reach = centres.norm()                                                   # [faces] Scalar+
    separations = (centres[:, None] - centres[None, :]).norm()                 # [faces, faces] Scalar+
    chosen: list[int] = []
    for face in reach.argsort():
        if reach[face] < max_radius and all(separations[face, other] > min_separation for other in chosen):
            chosen.append(int(face))
    return chosen


def draw_metric_basis(corners: core.Point, points: core.Point,
                      basis: core.Metric, metric: core.Metric) -> plt.Figure:
    """The three hyperbolic Regge edge metric basis forms and their sum into a positive-definite metric ellipse."""
    corner_xy = plane_places(corners)
    # The samples triangulated, for the pictures of the forms across them.
    mesh = core.Mesh.triangulate(points)
    places = solids.origin + solids.context.include_from_algebra(points.batch())   # [samples] Point-
    centroid = corners.mean()                                                  # [] Point-
    offsets = points - centroid

    basis_forms = np.asarray((basis[:, None](offsets[None, :]) & offsets[None, :]).to_array())
    metric_form = np.asarray((metric(offsets) & offsets).to_array())
    reference_form = np.asarray(offsets.norm_squared().to_array())
    basis_bound = np.abs(basis_forms).max()
    # Six levels at even steps either side of zero, the outermost three quarters of the way to the bound.
    nonzero_levels = contours.Lines(0.3 * basis_bound, 0.15 * basis_bound, np.append(to_rgb(wall_colour), 0.55), 0.9)
    # The zero level through the centroid consists of the two null lines parallel to the other two edges.
    null_lines = contours.Level(0.0, np.append(to_rgb(primal_colour), 1.0), 1.5)

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        frame_panel(ax, corner_xy, 0.22)
        values = basis_forms[index]
        colours = diverging_colours(Normalize(-basis_bound, basis_bound)(values))[:, :3]
        contours.overlay(ax, places, mesh, values, colours, lines=(nonzero_levels,),
                         levels=(null_lines,)).set_zorder(1)
        ax.add_patch(Polygon(corner_xy, fill=False, edgecolor=wall_colour, linewidth=1.1, zorder=4))
        start, end = corner_xy[(index + 1) % 3], corner_xy[(index + 2) % 3]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=accent_colour,
                linewidth=2.6, solid_capstyle="round", zorder=5)
        ax.scatter(*corner_xy.T, s=18, color=wall_colour, zorder=6)
        ax.scatter(*corner_xy[index], s=44, color=primal_colour, edgecolor="white", linewidth=0.9, zorder=7)

    ax = axes[3]
    frame_panel(ax, corner_xy, 0.22)
    max_metric = metric_form.max()
    ellipse_levels = np.linspace(0.03 * max_metric, 0.28 * max_metric, 4)
    ellipses = tuple(contours.Level(level, np.append(to_rgb(accent_colour), 1.0), 1.9) for level in ellipse_levels)
    contours.overlay(ax, places, mesh, metric_form, hat_colours(Normalize(0.0, max_metric)(metric_form))[:, :3],
                     levels=ellipses).set_zorder(1)
    # The reference circles, the same levels of a second field over the same samples, a clear picture over the first.
    circles = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.65), 0.9) for level in ellipse_levels)
    contours.overlay(ax, places, mesh, reference_form, np.zeros(3), levels=circles, opacity=0.0).set_zorder(2)
    ax.add_patch(Polygon(corner_xy, fill=False, edgecolor=wall_colour, linewidth=1.2, zorder=4))
    ax.scatter(*corner_xy.T, s=36, color=accent_colour, edgecolor="white", linewidth=0.9, zorder=6)
    ax.scatter(*plane_places(centroid), s=22, color=wall_colour, zorder=6)
    return figure


def draw_curvature_defects(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                           metrics: core.Metric, curvatures: core.Bivector) -> plt.Figure:
    """Vertex curvature shades the fixed reference disks, with equal-length displacement contours at sampled faces."""
    mv = vertices.context.multivector
    places = plane_places(vertices)
    curvature_values = np.asarray((curvatures.batch() & mv.xy.dual()).to_array())
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())   # [vertices] Point-
    walls = plane_places(mesh.reading(mesh.boundary_edges, 1) * (mesh.incidence[1, 0] * vertices)).transpose(1, 0, 2)
    boundary = mesh.exterior(0)
    interior = ~np.isin(np.arange(mesh.n_vertices), boundary)
    bound = np.abs(curvature_values[:, interior]).max()

    ellipse_faces = spaced_faces(centres, 0.84, 0.24)
    radius = 0.082
    span = np.linspace(-1.6 * radius, 1.6 * radius, n_ellipse_grid)
    grid_x, grid_y = np.meshgrid(span, span, indexing="ij")
    grid_offsets = mv.x.dual() * grid_x + mv.y.dual() * grid_y                 # [grid, grid] Point-
    sampled_metrics = metrics.batch()[:, ellipse_faces]
    # Implicit evaluation of each face's Regge metric quadratic form on the local displacement grid.
    quadrics = np.asarray((sampled_metrics[:, :, None, None](grid_offsets) & grid_offsets).to_array())
    reference_quadric = grid_offsets.norm_squared().to_array()                # [grid, grid]
    sampled_centres = centres.batch()[ellipse_faces]                          # [samples] Point-
    # Each local grid placed at its face's centre, where its ellipses are drawn, its points numbered row by row; and
    # all the grids together, one mesh for the reference circles of them all.
    n_samples = len(ellipse_faces)
    grid_points = (solids.origin + solids.context.include_from_algebra(sampled_centres[:, None, None] + grid_offsets)
                   ).reshape(n_samples, -1)                                   # [samples, grid * grid] Point-
    grid = contours.Mesh.grid(n_ellipse_grid, n_ellipse_grid)
    grids = contours.Mesh(n_samples * grid.n_vertices,
                          (grid.faces + grid.n_vertices * np.arange(n_samples)[:, None, None]).reshape(-1, 3))
    reference_circles = contours.Level(radius**2, np.append(to_rgb(wall_colour), 0.65), 0.75)

    # Log-ratio of radial to circumferential metric stretch at each sampled face centroid for indicatrix colouring.
    radial_dirs = sampled_centres.normalized()                                # [samples] Point-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()
    hoop_dirs = quarter_turn >> radial_dirs
    radial_stretch = np.asarray((sampled_metrics(radial_dirs[None, :]) & radial_dirs[None, :]).to_array())
    hoop_stretch = np.asarray((sampled_metrics(hoop_dirs[None, :]) & hoop_dirs[None, :]).to_array())
    anisotropy = np.log(radial_stretch / hoop_stretch)
    anisotropy_bound = np.abs(anisotropy).max()
    indicatrix_norm = Normalize(-anisotropy_bound, anisotropy_bound)

    figure, axes = charts.subplots(*disk_row_size, ncols=3)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for case_index, ax in enumerate(axes):
        frame_panel(ax, places, 0.08)
        field = curvature_values[case_index].copy()
        field[boundary] = 0.0
        contours.overlay(ax, points, mesh, field, diverging_colours(Normalize(-bound, bound)(field))[:, :3]).set_zorder(1)
        ax.triplot(triangulation, color=mesh_colour, linewidth=0.35, alpha=0.38, zorder=2)
        # Each face's ellipse its own clear picture, filled translucent inside and outlined in its colour: the fills
        # under the reference circles, the outlines over them.
        for places_at, quadric, stretch in zip(grid_points, quadrics[case_index], anisotropy[case_index]):
            colour = np.array(indicatrix_colours(indicatrix_norm(stretch)))
            inside = contours.Bands(0.0, radius**2, np.array([np.append(colour[:3], 0.22), np.zeros(4)]))
            contours.overlay(ax, places_at, grid, quadric.reshape(-1), np.zeros(3), bands=(inside,),
                             opacity=0.0).set_zorder(3)
            outline = contours.Level(radius**2, colour, 1.5)
            contours.overlay(ax, places_at, grid, quadric.reshape(-1), np.zeros(3), levels=(outline,),
                             opacity=0.0).set_zorder(5)
        contours.overlay(ax, grid_points.reshape(-1), grids, np.tile(reference_quadric.reshape(-1), len(grid_points)),
                         np.zeros(3), levels=(reference_circles,), opacity=0.0).set_zorder(4)
        ax.add_collection(LineCollection(walls, colors=wall_colour, linewidths=1.3, zorder=6))
    return figure


def draw_intrinsic_flow(mesh: core.Mesh, vertices: core.Point,
                        potential: core.Scalar,
                        current: core.Point) -> plt.Figure:
    """Potential colours and current streaks on one shared magnitude scale across the Regge metrics."""
    places = plane_places(vertices)
    potentials = np.asarray(potential.batch().to_array())
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = plane_places(mesh.reading(mesh.boundary_edges, 1) * (mesh.incidence[1, 0] * vertices)).transpose(1, 0, 2)
    potential_levels = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.45), 1.0)
                             for level in np.linspace(-0.85, 0.85, n_contours + 2)[1:-1])
    # Area-weighted vertex currents give a continuous drawing of the facewise solution.
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)
    areas = ((sides[0] & sides[1]) / 2).norm()                              # Scalar+[faces]
    corners = ~mesh.incidence[2, 0].mean(axis=0)
    vertex_current = (corners * (areas * current)) / (corners * areas)       # [cases] Point-[vertices]
    reference_speed = float(np.percentile(vertex_current.norm().batch().to_array(), 95))
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())   # [vertices] Point-
    directions = solids.context.include_from_algebra(vertex_current.batch())        # [cases, vertices] Point-

    figure, axes = charts.subplots(*disk_row_size, ncols=3)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index, ax in enumerate(axes):
        frame_panel(ax, places, 0.08)
        colours = diverging_colours(Normalize(-1.0, 1.0)(potentials[index]))[:, :3]
        contours.overlay(ax, points, mesh, potentials[index], colours, levels=potential_levels).set_zorder(1)
        ax.triplot(triangulation, color=mesh_colour, linewidth=0.35, alpha=0.45, zorder=2)
        flow.overlay(ax, points, mesh, directions[index], reference_speed, np.array(to_rgb(flow_colour)))
        ax.add_collection(LineCollection(walls, colors=wall_colour, linewidths=1.3, zorder=6))
    return figure
