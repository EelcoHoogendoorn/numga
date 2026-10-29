"""The drawings show Regge edge metric cones, metric ellipses, vertex curvature defects, and intrinsic Laplace–Beltrami flow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import stack
from examples.util import charts
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
n_streamlines = 15
n_ellipse_grid = 25
# The pixel sizes of a row of four triangles and of a row of three disks.
row_size, disk_row_size = (780, 195), (780, 260)


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Drawing coordinates of intrinsic positions and directions."""
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


def spaced_faces(centre_xy: np.ndarray, max_radius: float, min_separation: float) -> list[int]:
    """Well-spaced interior face indices ordered from the origin outward."""
    chosen: list[int] = []
    for face in np.argsort(np.linalg.norm(centre_xy, axis=-1)):
        if np.linalg.norm(centre_xy[face]) < max_radius and all(
            np.linalg.norm(centre_xy[face] - centre_xy[other]) > min_separation for other in chosen
        ):
            chosen.append(int(face))
    return chosen


def draw_metric_basis(corners: core.Point, points: core.Point,
                      basis: core.Metric, metric: core.Metric) -> plt.Figure:
    """The three hyperbolic Regge edge metric basis forms and their sum into a positive-definite metric ellipse."""
    sample_xy = coordinates(points)
    corner_xy = coordinates(corners)
    triangulation = Triangulation(sample_xy[:, 0], sample_xy[:, 1])
    centroid = corners.mean()
    offsets = points - centroid

    basis_forms = np.asarray((basis[:, None](offsets[None, :]) & offsets[None, :]).to_array())
    metric_form = np.asarray((metric(offsets) & offsets).to_array())
    reference_form = np.asarray(offsets.norm_squared().to_array())
    basis_bound = np.abs(basis_forms).max()
    nonzero_levels = np.linspace(-0.75 * basis_bound, 0.75 * basis_bound, 6)

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        frame_panel(ax, corner_xy, 0.22)
        values = basis_forms[index]
        ax.tripcolor(triangulation, values, shading="gouraud", cmap=diverging_colours,
                     norm=Normalize(-basis_bound, basis_bound), zorder=1)
        ax.tricontour(triangulation, values, levels=nonzero_levels, colors=wall_colour,
                      linestyles="solid", linewidths=0.65, alpha=0.55, zorder=2)
        # The zero contour through the centroid consists of the two null lines parallel to the other two edges.
        ax.tricontour(triangulation, values, levels=[0.0], colors=primal_colour,
                      linestyles="dashed", linewidths=1.35, zorder=3)
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
    ax.tripcolor(triangulation, metric_form, shading="gouraud", cmap=hat_colours,
                 norm=Normalize(0.0, max_metric), zorder=1)
    ax.tricontour(triangulation, reference_form, levels=ellipse_levels, colors=wall_colour,
                  linestyles="dashed", linewidths=0.9, alpha=0.65, zorder=2)
    ax.tricontour(triangulation, metric_form, levels=ellipse_levels, colors=accent_colour,
                  linestyles="solid", linewidths=1.35, zorder=3)
    ax.add_patch(Polygon(corner_xy, fill=False, edgecolor=wall_colour, linewidth=1.2, zorder=4))
    ax.scatter(*corner_xy.T, s=36, color=accent_colour, edgecolor="white", linewidth=0.9, zorder=6)
    ax.scatter(*corner_xy.mean(axis=0), s=22, color=wall_colour, zorder=6)
    return figure


def draw_curvature_defects(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                           metrics: core.Metric, curvatures: core.Bivector) -> plt.Figure:
    """Vertex curvature shades the fixed reference disks, with equal-length displacement contours at sampled faces."""
    mv = vertices.context.multivector
    places = coordinates(vertices)
    centre_xy = coordinates(centres)
    curvature_values = np.asarray((curvatures.batch() & mv.xy.dual()).to_array())
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = places[mesh.edges[mesh.boundary_edges]]
    boundary = mesh.exterior(0)
    interior = ~np.isin(np.arange(mesh.n_vertices), boundary)
    bound = np.abs(curvature_values[:, interior]).max()

    ellipse_faces = spaced_faces(centre_xy, 0.84, 0.24)
    radius = 0.082
    span = np.linspace(-1.6 * radius, 1.6 * radius, n_ellipse_grid)
    grid_x, grid_y = np.meshgrid(span, span, indexing="ij")
    grid_offsets = mv.x.dual() * grid_x + mv.y.dual() * grid_y
    sampled_metrics = metrics.batch()[:, ellipse_faces]
    # Implicit evaluation of each face's Regge metric quadratic form on the local displacement grid.
    quadrics = np.asarray((sampled_metrics[:, :, None, None](grid_offsets) & grid_offsets).to_array())
    reference_quadric = grid_x**2 + grid_y**2

    # Log-ratio of radial to circumferential metric stretch at each sampled face centroid for indicatrix colouring.
    sampled_centres = centres.batch()[ellipse_faces]
    radial_dirs = sampled_centres / sampled_centres.norm()
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
        ax.tripcolor(triangulation, field, shading="gouraud", cmap=diverging_colours,
                     norm=Normalize(-bound, bound), zorder=1)
        ax.triplot(triangulation, color=mesh_colour, linewidth=0.35, alpha=0.38, zorder=2)
        for sample_index, face in enumerate(ellipse_faces):
            cx, cy = centre_xy[face]
            colour = indicatrix_colours(indicatrix_norm(anisotropy[case_index, sample_index]))
            ax.contourf(cx + grid_x, cy + grid_y, quadrics[case_index, sample_index],
                        levels=[0.0, radius**2], colors=[colour], alpha=0.22, zorder=3)
            ax.contour(cx + grid_x, cy + grid_y, reference_quadric, levels=[radius**2],
                       colors=wall_colour, linestyles="dashed", linewidths=0.75, alpha=0.65, zorder=4)
            ax.contour(cx + grid_x, cy + grid_y, quadrics[case_index, sample_index], levels=[radius**2],
                       colors=[colour], linestyles="solid", linewidths=1.5, zorder=5)
        ax.add_collection(LineCollection(walls, colors=wall_colour, linewidths=1.3, zorder=6))
    return figure


def draw_intrinsic_flow(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                        potential: core.Scalar, streamfunction: core.Scalar,
                        current: core.Point) -> plt.Figure:
    """Steady potential contours, intrinsic flux streamlines, and current vectors across the three Regge metrics."""
    places = coordinates(vertices)
    centre_xy = coordinates(centres)
    potentials = np.asarray(potential.batch().to_array())
    streams = np.asarray(streamfunction.batch().to_array())
    currents = coordinates(current)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = places[mesh.edges[mesh.boundary_edges]]
    potential_levels = np.linspace(-0.85, 0.85, n_contours + 2)[1:-1]
    arrow_faces = spaced_faces(centre_xy, 0.88, 0.18)

    figure, axes = charts.subplots(*disk_row_size, ncols=3)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index, ax in enumerate(axes):
        frame_panel(ax, places, 0.08)
        ax.tripcolor(triangulation, potentials[index], shading="gouraud", cmap=diverging_colours,
                     norm=Normalize(-1.0, 1.0), zorder=1)
        ax.triplot(triangulation, color=mesh_colour, linewidth=0.35, alpha=0.45, zorder=2)
        ax.tricontour(triangulation, potentials[index], levels=potential_levels,
                      colors=wall_colour, linestyles="dashed", linewidths=0.75, alpha=0.65, zorder=3)
        stream = streams[index]
        margin = 0.07 * (stream.max() - stream.min())
        stream_levels = np.linspace(stream.min() + margin, stream.max() - margin, n_streamlines)
        ax.tricontour(triangulation, stream, levels=stream_levels,
                      colors=flow_colour, linestyles="solid", linewidths=0.85, alpha=0.78, zorder=4)
        panel_speed = np.quantile(np.linalg.norm(currents[index], axis=-1), 0.92)
        ax.quiver(centre_xy[arrow_faces, 0], centre_xy[arrow_faces, 1],
                  currents[index, arrow_faces, 0], currents[index, arrow_faces, 1],
                  color=flow_colour, angles="xy", scale_units="xy", scale=7.2 * panel_speed,
                  width=0.0055, headwidth=3.4, headlength=4.2, pivot="mid", zorder=5)
        ax.add_collection(LineCollection(walls, colors=wall_colour, linewidths=1.3, zorder=6))
    return figure
