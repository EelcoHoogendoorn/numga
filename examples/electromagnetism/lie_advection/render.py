"""Swept edge ribbons, magnetic flux density fields, and side-by-side Lie advection animations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import concatenate, stack
from examples.electromagnetism.lie_advection import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import contours, flow, simplices
from examples.util.wgsl.camera import Camera, flat_panel

Topology = core.Topology

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
# The velocity arrows bolder than the chart's lines and the edges' arrows twice as bold, to stand out along the edges
# they lie on, each with a chevron at its tip; the swept ribbons' dotted outlines and the mesh as thin as its frame,
# the mesh faint. Widths, dashes, radii and the length of a chevron's strokes in pixels.
velocity_width, velocity_chevron = 1.5 * charts.LINE_PX, 9.0
edge_arrow_width, edge_chevron = 2 * charts.LINE_PX, 10.0
ribbon_dots = (1.0, 2.0)
ribbon_opacity, triangle_opacity = 0.72, 0.88
triangle_colour = np.array(to_rgb("#f6f4ee"))
displaced_dashes = (5.5, 2.4)
corner_radius, corner_ring_radius = 2.5, 3.0
mesh_alpha = 0.16


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """A named colour's red, green and blue."""
    return np.array(to_rgb(colour))


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


def vertex_averages(topology: core.Topology, areas: core.Scalar, face_field: core.Scalar) -> np.ndarray:
    """Area-weighted vertex average of a scalar face field for smooth Gouraud shading."""
    corner_mean = topology.incidence[2, 0].mean(axis=0)
    vertex_field = (~corner_mean * (areas * face_field)) / (~corner_mean * areas)
    return np.asarray(vertex_field.batch().to_array())


def disk_lines(topology: core.Topology, points: core.Point, camera: Camera) -> list[np.ndarray]:
    """One disk panel's faint mesh and its rim, over the magnetic flux density its picture shades."""
    return [simplices.strokes(points, topology, rgb(mesh_colour), charts.FRAME_PX, camera, opacity=mesh_alpha),
            simplices.strokes(points, topology.border, rgb(wall_colour), charts.LINE_PX, camera)]


@charts.style()
def draw_cartan_ribbon(corners: core.Point, corner_forms: core.Plane,
                       corner_velocities: core.Point, ribbon_step: float,
                       endpoint_differences: core.Scalar, swept_fluxes: core.Scalar) -> plt.Figure:
    """Swept edge parallelogram ribbons on a moving triangle beside the Cartan rate decomposition."""
    displaced = corners + ribbon_step * corner_velocities                       # [corners] Point-
    # The ribbon each edge sweeps in the step is the parallelogram between its endpoints and their displaced copies.
    tails, heads = core.triangle.corners[2, 1]
    ribbons = stack((corners[tails], corners[heads], displaced[heads], displaced[tails]), axis=-1)   # [corners, 4] Point-
    middles = (core.triangle.incidence[2, 1] * (core.triangle.incidence[1, 0].mean(axis=0) * corners.field())).batch()[:, 0]  # [corners] Point-
    edges = (core.triangle.boundary[2, 1] * (core.triangle.d[0] * corners.field())).batch()[:, 0]  # [corners] Point-
    # Each edge's arrow about its middle, and each corner's velocity from it to its displaced copy, chains of two.
    edge_arrows = stack((middles - 0.16 * edges, middles + 0.16 * edges), axis=-1).reshape(-1)   # [corners * 2] Point-
    velocity_arrows = stack((corners, displaced), axis=-1).reshape(-1)                           # [corners * 2] Point-
    arrows = Topology.chains([2, 2, 2])
    # Each ribbon two triangles, its outline their boundary; the triangle and its displaced copy closed chains.
    ribbon = Topology(4, np.array([[0, 1, 2], [0, 2, 3]]))
    endpoint_vals = np.asarray(endpoint_differences.to_array())
    swept_vals = np.asarray(swept_fluxes.to_array())
    total_vals = endpoint_vals + swept_vals

    # Three quarters of the column: the triangle in a square beside the bars of its three edges and their sum.
    figure, (geom_ax, bar_ax) = charts.subplots(600, 270, ncols=2, width_ratios=(1, 1.2), layout="constrained")

    frame_panel(geom_ax, stack((corners, displaced)), 0.16)
    camera = flat_panel(geom_ax, corners.algebra)
    flux_scale = max(np.abs(swept_vals).max(), 1e-8)
    ribbon_colours = flux_colours(0.5 + 0.42 * swept_vals / flux_scale)[:, :3]   # [corners, 3]
    # The form across the triangle as streaks, their length its strength: its values at the corners, interpolated by
    # the corners' hats as the picture's triangle interpolates them.
    directions = corner_forms.dual()                                            # [corners] Point-
    strongest = float(directions.dual().norm().max().to_array())
    # Each ribbon a picture of its own, so that where ribbons overlap, both show.
    ribbon_pictures = [picture for index in range(3) for picture in (
        contours.picture(ribbons[index], ribbon, np.zeros(4), ribbon_colours[index], camera, opacity=ribbon_opacity),
        simplices.strokes(ribbons[index], ribbon.border, rgb(wall_colour), charts.FRAME_PX, camera, dashes=ribbon_dots,
                          cap=simplices.Cap.BUTT, opacity=ribbon_opacity))]
    simplices.overlay(geom_ax, [
        *ribbon_pictures,
        contours.picture(corners, core.triangle, np.zeros(3), triangle_colour, camera, opacity=triangle_opacity),
        simplices.strokes(corners, core.triangle.border, rgb(wall_colour), charts.LINE_PX, camera,
                          opacity=triangle_opacity),
        flow.layer(corners, core.triangle, directions, camera, strongest, rgb(primal_colour)),
        simplices.strokes(displaced, core.triangle.border, rgb(ink_colour), charts.LINE_PX, camera,
                          dashes=displaced_dashes, cap=simplices.Cap.BUTT),
        simplices.strokes(edge_arrows, arrows, rgb(wall_colour), edge_arrow_width, camera, chevrons=np.ones(3),
                          chevron=edge_chevron),
        simplices.strokes(velocity_arrows, arrows, rgb(accent_colour), velocity_width, camera, chevrons=np.ones(3),
                          chevron=velocity_chevron),
        simplices.markers(concatenate((corners, displaced)), np.ones(3), corner_ring_radius, camera),
        simplices.markers(concatenate((corners, displaced)),
                          np.repeat(np.stack((rgb(wall_colour), rgb(ink_colour))), 3, axis=0), corner_radius, camera)])

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


def flux_levels(flux_limit: float) -> tuple[contours.Level, ...]:
    """The iso-flux contours, at even steps either side of zero up to near the limit of the colour scale."""
    half_levels = np.linspace(0.16 * flux_limit, 0.88 * flux_limit, n_flux_contours // 2)
    return tuple(contours.Level(level, flux_ink, charts.LINE_PX)
                 for level in np.concatenate((-half_levels[::-1], half_levels)))


@charts.style()
def draw_advection_comparison(topology: core.Topology, vertices: core.Point,
                              flux_density: core.Scalar, times: np.ndarray,
                              energies: core.Scalar, total_fluxes: core.Scalar) -> plt.Figure:
    """Ideal versus resistive magnetic flux advection alongside magnetic energy and total flux traces."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)
    areas = (core.time & (sides[0] & sides[1]) / 2).norm()
    vertex_fluxes = vertex_averages(topology, areas, flux_density)
    flux_limit = float(np.percentile(np.abs(vertex_fluxes[0]), 96))

    # Nine tenths of the column: the two disks square at the chart's full height, the traces beside them in a
    # layout of their own, so that the room for the traces' labels does not shrink the disks.
    figure = plt.figure(figsize=(720 / charts.DPI, 250 / charts.DPI), layout="constrained")
    disks, traces = figure.subfigures(1, 2, width_ratios=(2, 0.9))
    left_ax, mid_ax = disks.subplots(1, 2)
    trace_ax = traces.subplots()

    for ax in (left_ax, mid_ax):
        frame_panel(ax, vertices, 0.06)

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
    points = vertices.batch()                                                # [vertices] Point-
    flux_scale = Normalize(-flux_limit, flux_limit)
    for ax, vertex_flux in zip((left_ax, mid_ax), vertex_fluxes):
        camera = flat_panel(ax, core.ga)
        colours = flux_colours(flux_scale(vertex_flux))[:, :3]
        simplices.overlay(ax, [contours.picture(points, topology, vertex_flux, colours, camera,
                                                levels=flux_levels(flux_limit)),
                               *disk_lines(topology, points, camera)])
    return figure


def animate_advection(topology: core.Topology, vertices: core.Point,
                      frames: Iterable[core.Scalar]
                      ) -> Iterator[np.ndarray]:
    """Side-by-side ideal and resistive magnetic flux winding frames captured at fixed colour scale."""
    frames = iter(frames)
    first = next(frames)
    points = vertices.batch()                                                # [vertices] Point-

    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)
    areas = (core.time & (sides[0] & sides[1]) / 2).norm()
    first_vertex_flux = vertex_averages(topology, areas, first)
    flux_limit = float(np.percentile(np.abs(first_vertex_flux[0]), 95))
    levels = flux_levels(flux_limit)
    flux_scale = Normalize(-flux_limit, flux_limit)

    # The disks drawn in the charts' strokes, as in the comparison of their final states; the figure and everything
    # that stays put drawn once, each frame shading the flux density anew.
    with charts.style():
        figure, axes = charts.subplots(612, 306, ncols=2)
        figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98, wspace=0.04)
        for ax in axes:
            frame_panel(ax, vertices, 0.06)
        views = [flat_panel(ax, core.ga) for ax in axes]
        lines = [disk_lines(topology, points, view) for view in views]
        textures = [simplices.overlay(ax, panel_lines) for ax, panel_lines in zip(axes, lines)]
    for flux_density in chain((first,), frames):
        vertex_fluxes = vertex_averages(topology, areas, flux_density)
        for texture, view, panel_lines, vertex_flux in zip(textures, views, lines, vertex_fluxes):
            colours = flux_colours(flux_scale(vertex_flux))[:, :3]
            texture.set_data(simplices.composited(
                [contours.picture(points, topology, vertex_flux, colours, view, levels=levels), *panel_lines]))
        with charts.style():
            image = capture(figure)
        yield image
    plt.close(figure)
