"""Swept edge ribbons, magnetic flux density fields, and side-by-side Lie advection animations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import stack
from examples.electromagnetism.lie_advection import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import contours, flow, pinned, solids
from examples.util.wgsl.flow import axes_view

primal_colour = "#286580"
accent_colour = "#d77542"
ink_colour = "#173c4b"
teal_colour = "#148d88"
mesh_colour = "#b8c4c9"
wall_colour = "#52616a"
flux_colours = LinearSegmentedColormap.from_list(
    "lie_flux", ("#184a63", "#5b9ab5", "#f6f4ee", "#df7a47", "#a6341b"))
n_flux_contours = 8
# The iso-flux contours' ink, colour and opacity.
flux_ink = np.append(to_rgb(ink_colour), 0.68)
# The velocity arrows bolder than the chart's lines; the swept ribbons' dotted outlines and the mesh as thin as its
# frame, the mesh faint.
velocity_width = charts.points(1.5 * charts.LINE_PX)
mesh_alpha = 0.16
# The observer's slice of space, which carries the disk's events into the renderer's space.
space_slice = core.mv.t ^ (core.Point & core.time)                              # [] Point- <- Point-


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Drawing coordinates of intrinsic positions and directions."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def frame_panel(ax: plt.Axes, points: core.Point, margin_ratio: float) -> None:
    """Square equal-aspect viewport around the points with no axes or ticks."""
    mv = points.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & points.batch().reshape(-1)
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
    centre = (low + high) / 2
    spans = high - low
    half_span = (1 + margin_ratio) * spans.max() / 2
    limits = stack((centre - half_span, centre + half_span), axis=-1).to_array()
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.set_axis_off()


def plain_axes(ax: plt.Axes) -> None:
    """Minimal left and bottom spines for quantity plots."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.78")
    ax.tick_params(color="0.78")


def rim_segments(mesh: core.Mesh, vertices: core.Point) -> np.ndarray:
    """The rim's edges as segments to draw, `[rim, 2, 2]`: each edge's tail and head."""
    tails, heads = mesh.reading(mesh.boundary_edges, 1) * (mesh.incidence[1, 0] * vertices)   # Point-[rim], Point-[rim]
    return np.stack((coordinates(tails), coordinates(heads)), axis=-2)


def vertex_averages(mesh: core.Mesh, areas: core.Scalar, face_field: core.Scalar) -> np.ndarray:
    """Area-weighted vertex average of a scalar face field for smooth Gouraud shading."""
    corner_mean = mesh.incidence[2, 0].mean(axis=0)
    vertex_field = (~corner_mean * (areas * face_field)) / (~corner_mean * areas)
    return np.asarray(vertex_field.batch().to_array())


@charts.style()
def draw_cartan_ribbon(corners: core.Point, corner_forms: core.Plane,
                       corner_velocities: core.Point, ribbon_step: float,
                       endpoint_differences: core.Scalar, swept_fluxes: core.Scalar) -> plt.Figure:
    """Swept edge parallelogram ribbons on a moving triangle beside the Cartan rate decomposition."""
    displaced = corners + ribbon_step * corner_velocities                       # [corners] Point-
    # The edge facing each corner runs from the corner after it to the one before it; the ribbon it sweeps in the
    # step is the parallelogram between it and its displaced copy.
    firsts, seconds = corners[[1, 2, 0]], corners[[2, 0, 1]]                    # [corners] Point-, [corners] Point-
    ribbons = stack((firsts, seconds, displaced[[2, 0, 1]], displaced[[1, 2, 0]]), axis=-1)   # [corners, 4] Point-
    middles, edges = (firsts + seconds) / 2, seconds - firsts                   # [corners] Point-, [corners] Point-
    corner_xy = coordinates(corners)
    displaced_xy = coordinates(displaced)
    ribbon_xy = coordinates(ribbons)                                            # [corners, 4, 2]
    tails_xy, heads_xy = coordinates(middles - 0.16 * edges), coordinates(middles + 0.16 * edges)
    endpoint_vals = np.asarray(endpoint_differences.to_array())
    swept_vals = np.asarray(swept_fluxes.to_array())
    total_vals = endpoint_vals + swept_vals

    # Three quarters of the column: the triangle in a square beside the bars of its three edges and their sum.
    figure, (geom_ax, bar_ax) = charts.subplots(600, 270, ncols=2, width_ratios=(1, 1.2), layout="constrained")

    frame_panel(geom_ax, stack((corners, displaced)), 0.16)
    flux_scale = max(np.abs(swept_vals).max(), 1e-8)
    for index in range(3):
        colour = flux_colours(0.5 + 0.42 * swept_vals[index] / flux_scale)
        geom_ax.add_patch(Polygon(ribbon_xy[index], facecolor=colour, edgecolor=wall_colour,
                                  linewidth=charts.FRAME_WIDTH, linestyle=":", alpha=0.72, zorder=1))

    geom_ax.add_patch(Polygon(corner_xy, facecolor="#f6f4ee", edgecolor=wall_colour, alpha=0.88, zorder=2))
    geom_ax.add_patch(Polygon(displaced_xy, fill=False, edgecolor=ink_colour, linestyle="--", zorder=4))
    # The form across the triangle as streaks, their length its strength: its values at the corners, interpolated by
    # the corners' hats as the picture's triangle interpolates them.
    triangle = core.Mesh(3, np.array([[0, 1, 2]]))
    places = solids.origin + solids.context.include_from_algebra(space_slice(corners))   # [corners] Point-
    directions = solids.context.include_from_algebra(space_slice(corner_forms.dual()))   # [corners] Point-
    strongest = float(directions.dual().norm().max().to_array())
    flow.overlay(geom_ax, places, triangle, directions, strongest, np.array(to_rgb(primal_colour))).set_zorder(3)

    for index in range(3):
        geom_ax.annotate("", xy=heads_xy[index], xytext=tails_xy[index],
                         arrowprops=dict(arrowstyle="-|>", mutation_scale=10, color=wall_colour),
                         zorder=5)
        geom_ax.annotate("", xy=displaced_xy[index], xytext=corner_xy[index],
                         arrowprops=dict(arrowstyle="-|>", mutation_scale=11, color=accent_colour,
                                         linewidth=velocity_width),
                         zorder=6)
    geom_ax.scatter(*corner_xy.T, color=wall_colour, edgecolor="white", linewidth=charts.FRAME_WIDTH, zorder=7)
    geom_ax.scatter(*displaced_xy.T, color=ink_colour, edgecolor="white", linewidth=charts.FRAME_WIDTH, zorder=7)

    groups = np.arange(4)
    endpoint_all = np.append(endpoint_vals, endpoint_vals.sum())
    swept_all = np.append(swept_vals, swept_vals.sum())
    total_all = np.append(total_vals, total_vals.sum())
    width = 0.24
    bar_ax.axhline(0.0, color="0.78", linewidth=charts.FRAME_WIDTH, zorder=1)
    bar_ax.bar(groups - width, endpoint_all, width=width, color=primal_colour, zorder=2)
    bar_ax.bar(groups, swept_all, width=width, color=accent_colour, zorder=2)
    bar_ax.bar(groups + width, total_all, width=width, color=ink_colour, zorder=2)
    span = max(np.abs(np.concatenate((endpoint_all, swept_all, total_all))).max(), 0.1)
    limit = float(np.round(1.18 * span, 2))
    bar_ax.set(xlim=(-0.55, 3.55), ylim=(-limit, limit),
               xticks=groups, xticklabels=["edge 0", "edge 1", "edge 2", "loop sum"],
               yticks=[-round(span, 2), 0.0, round(span, 2)],
               ylabel="circulation rate")
    plain_axes(bar_ax)
    return figure


def disk_panel(ax: plt.Axes, triangulation: Triangulation, vertices: core.Point, walls: np.ndarray) -> None:
    """One disk panel's frame, faint mesh and rim, over the magnetic flux density its picture shades."""
    frame_panel(ax, vertices, 0.06)
    ax.triplot(triangulation, color=mesh_colour, linewidth=charts.FRAME_WIDTH, alpha=mesh_alpha, zorder=2)
    ax.add_collection(LineCollection(walls, colors=wall_colour, zorder=4))


def flux_levels(flux_limit: float) -> tuple[contours.Level, ...]:
    """The iso-flux contours, at even steps either side of zero up to near the limit of the colour scale."""
    half_levels = np.linspace(0.16 * flux_limit, 0.88 * flux_limit, n_flux_contours // 2)
    return tuple(contours.Level(level, flux_ink, charts.LINE_PX)
                 for level in np.concatenate((-half_levels[::-1], half_levels)))


@charts.style()
def draw_advection_comparison(mesh: core.Mesh, vertices: core.Point,
                              flux_density: core.Scalar, times: np.ndarray,
                              energies: core.Scalar, total_fluxes: core.Scalar) -> plt.Figure:
    """Ideal versus resistive magnetic flux advection alongside magnetic energy and total flux traces."""
    places = coordinates(vertices)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = rim_segments(mesh, vertices)

    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)
    areas = (core.time & (sides[0] & sides[1]) / 2).norm()
    vertex_fluxes = vertex_averages(mesh, areas, flux_density)
    flux_limit = float(np.percentile(np.abs(vertex_fluxes[0]), 96))

    # Nine tenths of the column: the two disks square at the chart's full height, the traces beside them in a
    # layout of their own, so that the room for the traces' labels does not shrink the disks.
    figure = plt.figure(figsize=(720 / charts.DPI, 250 / charts.DPI), layout="constrained")
    disks, traces = figure.subfigures(1, 2, width_ratios=(2, 0.9))
    left_ax, mid_ax = disks.subplots(1, 2)
    trace_ax = traces.subplots()

    for ax in (left_ax, mid_ax):
        disk_panel(ax, triangulation, vertices, walls)

    energy_vals = np.asarray(energies.to_array())
    flux_vals = np.asarray(total_fluxes.to_array())
    trace_ax.plot(times, energy_vals[:, 0] / energy_vals[0, 0], color=ink_colour)
    trace_ax.plot(times, energy_vals[:, 1] / energy_vals[0, 0], color=accent_colour)
    trace_ax.plot(times, flux_vals[:, 1] / flux_vals[0, 1], color=teal_colour, linestyle="--")
    trace_ax.set(xlim=(times[0], times[-1]), ylim=(0.0, 1.08),
                 xticks=[0.0, round(float(times[-1]), 2)], yticks=[0.0, 1.0],
                 xlabel="time", ylabel=r"relative $E_B(t),\, \Phi(t)$")
    plain_axes(trace_ax)

    # The flux densities shaded once the layout is set, each picture at the pixels its panel gets.
    points = solids.origin + solids.context.include_from_algebra(space_slice(vertices.batch()))   # [vertices] Point-
    flux_scale = Normalize(-flux_limit, flux_limit)
    for ax, vertex_flux in zip((left_ax, mid_ax), vertex_fluxes):
        colours = flux_colours(flux_scale(vertex_flux))[:, :3]
        contours.overlay(ax, points, mesh, vertex_flux, colours, levels=flux_levels(flux_limit)).set_zorder(1)
    return figure


def animate_advection(mesh: core.Mesh, vertices: core.Point,
                      frames: Iterable[core.Scalar]
                      ) -> Iterator[np.ndarray]:
    """Side-by-side ideal and resistive magnetic flux winding frames captured at fixed colour scale."""
    frames = iter(frames)
    first = next(frames)
    places = coordinates(vertices)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = rim_segments(mesh, vertices)
    points = solids.origin + solids.context.include_from_algebra(space_slice(vertices.batch()))   # [vertices] Point-

    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)
    areas = (core.time & (sides[0] & sides[1]) / 2).norm()
    first_vertex_flux = vertex_averages(mesh, areas, first)
    flux_limit = float(np.percentile(np.abs(first_vertex_flux[0]), 95))
    levels = flux_levels(flux_limit)
    flux_scale = Normalize(-flux_limit, flux_limit)

    # The disks drawn in the charts' strokes, as in the comparison of their final states; the figure and everything
    # that stays put drawn once, each frame shading the flux density anew.
    with charts.style():
        figure, axes = charts.subplots(612, 306, ncols=2)
        figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98, wspace=0.04)
        for ax in axes:
            disk_panel(ax, triangulation, vertices, walls)
        views = [axes_view(ax) for ax in axes]
        textures = [pinned(ax, np.zeros(view.shape + (4,)), (*ax.get_xlim(), *ax.get_ylim()))
                    for ax, view in zip(axes, views)]
    for texture in textures:
        texture.set_zorder(1)
    for flux_density in chain((first,), frames):
        vertex_fluxes = vertex_averages(mesh, areas, flux_density)
        for texture, view, vertex_flux in zip(textures, views, vertex_fluxes):
            colours = flux_colours(flux_scale(vertex_flux))[:, :3]
            texture.set_data(contours.picture(points, mesh, vertex_flux, colours, view, levels=levels))
        with charts.style():
            image = capture(figure)
        yield image
    plt.close(figure)
