"""Swept edge ribbons, magnetic flux density contours, and side-by-side animations depict planar Lie advection."""

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
from numga.render import Camera, contours, drawing, flow, simplices
from examples.util.panels import offset_panel, overlay

Topology = core.Topology

primal_colour = "#286580"
accent_colour = "#d77542"
ink_colour = "#173c4b"
teal_colour = "#148d88"
triangulation_colour = "#b8c4c9"
wall_colour = "#52616a"
flux_colours = LinearSegmentedColormap.from_list(
    "lie_flux", ("#184a63", "#5b9ab5", "#f6f4ee", "#df7a47", "#a6341b"))
n_flux_contours = 8
flux_ink = np.append(to_rgb(ink_colour), 0.68)
# Velocity arrows are bolder than chart strokes and edge arrows are twice as bold, each tipped with a chevron;
# swept ribbons use dotted outlines and interior triangulation edges match the frame stroke at low opacity.
velocity_width, velocity_chevron = 1.5 * charts.LINE_PX, 9.0
edge_arrow_width, edge_chevron = 2 * charts.LINE_PX, 10.0
ribbon_dots = (1.0, 2.0)
ribbon_opacity, triangle_opacity = 0.72, 0.88
triangle_colour = np.array(to_rgb("#f6f4ee"))
displaced_dashes = (5.5, 2.4)
corner_radius, corner_ring_radius = 2.5, 3.0
triangulation_alpha = 0.16


# --- plumbing -------------------------------------------------------------------------
def rgb_colour(colour: str) -> np.ndarray:
    """A named or hex `colour` string converts to an RGB float array (`[3]`)."""
    return np.array(to_rgb(colour))


def panel_viewport(ax: plt.Axes, points: core.Point, margin_ratio: float) -> None:
    """Square equal-aspect limits around `points` frame `ax` with axes and ticks hidden."""
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


def quantity_axes(ax: plt.Axes) -> None:
    """Minimal left and bottom spines frame quantity plots on `ax` with top and right spines hidden."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.78")
    ax.tick_params(color="0.78")


def vertex_averages(topology: core.Topology, areas: core.Scalar, face_field: core.Scalar) -> np.ndarray:
    """Area-weighted face averages (`[..., V]`) interpolate `face_field` onto vertices of `topology` for smooth shading."""
    corner_mean = topology.incidence[2, 0].mean(axis=0)
    vertex_field = (corner_mean.adjugate() * (areas * face_field)) / (corner_mean.adjugate() * areas)
    return np.asarray(vertex_field.batch().to_array())


def disk_lines(topology: core.Topology, points: core.Point, camera: Camera) -> list[np.ndarray]:
    """Faint interior triangulation edges and the boundary rim overlay `topology` at `points` through `camera`."""
    return [simplices.strokes(points, topology, rgb_colour(triangulation_colour), charts.FRAME_PX,
                              camera, opacity=triangulation_alpha),
            simplices.strokes(points, topology.border, rgb_colour(wall_colour), charts.LINE_PX, camera)]


@charts.style()
def cartan_ribbon_figure(corners: core.Point, corner_potentials: core.Plane,
                         corner_velocities: core.Point, ribbon_step: float,
                         endpoint_differences: core.Scalar, swept_fluxes: core.Scalar) -> plt.Figure:
    """Swept edge parallelogram ribbons on a moving triangle stand beside a bar chart of the Cartan rate terms."""
    corner_points = corners.batch()                                             # [3] Point-
    displaced_corners = corner_points + ribbon_step * corner_velocities.batch() # [3] Point-
    # Each edge sweeps a parallelogram ribbon between its endpoints and their displaced copies.
    tails, heads = core.triangle.corners[2, 1]
    ribbons = stack((corner_points[tails], corner_points[heads],
                     displaced_corners[heads], displaced_corners[tails]), axis=-1)  # [3, 4] Point-
    middles = (core.triangle.incidence[2, 1] * (core.triangle.incidence[1, 0].mean(axis=0) * corners)).batch()[:, 0]  # [3] Point-
    edges = (core.triangle.boundary[2, 1] * (core.triangle.d[0] * corners)).batch()[:, 0]  # [3] Point-
    edge_arrows = stack((middles - 0.16 * edges, middles + 0.16 * edges), axis=-1).reshape(-1)  # [6] Point-
    velocity_arrows = stack((corner_points, displaced_corners), axis=-1).reshape(-1)  # [6] Point-
    arrows = Topology.chains([2, 2, 2])
    ribbon = Topology(4, np.array([[0, 1, 2], [0, 2, 3]]))
    endpoint_vals = np.asarray(endpoint_differences.batch()[:, 0].to_array())
    swept_vals = np.asarray(swept_fluxes.batch()[:, 0].to_array())
    total_vals = endpoint_vals + swept_vals

    figure, (geom_ax, bar_ax) = charts.subplots(600, 270, ncols=2, width_ratios=(1, 1.2), layout="constrained")

    panel_viewport(geom_ax, stack((corner_points, displaced_corners)), 0.16)
    camera = offset_panel(geom_ax, core.mv.x, core.mv.y)
    flux_scale = max(np.abs(swept_vals).max(), 1e-8)
    ribbon_colours = flux_colours(0.5 + 0.42 * swept_vals / flux_scale)[:, :3]  # [3, 3]
    # Streak length shows the potential's strength across the triangle from its linearly interpolated corner values.
    directions = corner_potentials.batch().dual()                               # [3] Point-
    peak_strength = float(directions.dual().norm().max().to_array())
    ribbon_pictures = [picture for index in range(3) for picture in (
        contours.picture(ribbons[index], ribbon, np.zeros(4), ribbon_colours[index], camera, opacity=ribbon_opacity),
        simplices.strokes(ribbons[index], ribbon.border, rgb_colour(wall_colour), charts.FRAME_PX, camera,
                          dashes=ribbon_dots, cap=simplices.Cap.BUTT, opacity=ribbon_opacity))]
    overlay(geom_ax, [
        *ribbon_pictures,
        contours.picture(corner_points, core.triangle, np.zeros(3), triangle_colour, camera, opacity=triangle_opacity),
        simplices.strokes(corner_points, core.triangle.border, rgb_colour(wall_colour), charts.LINE_PX, camera,
                          opacity=triangle_opacity),
        flow.layer(corner_points, core.triangle, directions, camera, peak_strength, rgb_colour(primal_colour)),
        simplices.strokes(displaced_corners, core.triangle.border, rgb_colour(ink_colour), charts.LINE_PX, camera,
                          dashes=displaced_dashes, cap=simplices.Cap.BUTT),
        simplices.strokes(edge_arrows, arrows, rgb_colour(wall_colour), edge_arrow_width, camera, chevrons=np.ones(3),
                          chevron=edge_chevron),
        simplices.strokes(velocity_arrows, arrows, rgb_colour(accent_colour), velocity_width, camera, chevrons=np.ones(3),
                          chevron=velocity_chevron),
        simplices.markers(concatenate((corner_points, displaced_corners)), np.ones(3), corner_ring_radius, camera),
        simplices.markers(concatenate((corner_points, displaced_corners)),
                          np.repeat(np.stack((rgb_colour(wall_colour), rgb_colour(ink_colour))), 3, axis=0),
                          corner_radius, camera)])

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
    quantity_axes(bar_ax)
    return figure


def flux_levels(flux_limit: float) -> tuple[contours.Level, ...]:
    """Symmetric positive and negative iso-flux contour levels span up to `0.88 * flux_limit`."""
    half_levels = np.linspace(0.16 * flux_limit, 0.88 * flux_limit, n_flux_contours // 2)
    return tuple(contours.Level(level, flux_ink, charts.LINE_PX)
                 for level in np.concatenate((-half_levels[::-1], half_levels)))


@charts.style()
def advection_comparison_figure(topology: core.Topology, vertices: core.Point,
                                flux_density: core.Scalar, times: np.ndarray,
                                energies: core.Scalar, total_fluxes: core.Scalar) -> plt.Figure:
    """Final ideal and resistive magnetic flux densities stand beside relative energy and flux histories."""
    V, E, F = topology.n_elements
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)                # [3] Point-[F]
    areas = (core.time & ((sides[0] & sides[1]) / 2)).norm()                    # Scalar+[F]
    vertex_fluxes = vertex_averages(topology, areas, flux_density)
    flux_limit = float(np.percentile(np.abs(vertex_fluxes[0]), 96))

    figure = plt.figure(figsize=(720 / charts.DPI, 250 / charts.DPI), layout="constrained")
    disks, traces = figure.subfigures(1, 2, width_ratios=(2, 0.9))
    left_ax, mid_ax = disks.subplots(1, 2)
    trace_ax = traces.subplots()

    for ax in (left_ax, mid_ax):
        panel_viewport(ax, vertices, 0.06)

    energy_vals = np.asarray(energies.to_array())
    flux_vals = np.asarray(total_fluxes.to_array())
    trace_ax.plot(times, energy_vals[:, 0] / energy_vals[0, 0], color=ink_colour)
    trace_ax.plot(times, energy_vals[:, 1] / energy_vals[0, 0], color=accent_colour)
    trace_ax.plot(times, flux_vals[:, 1] / flux_vals[0, 1], color=teal_colour, linestyle="--")
    trace_ax.set(xlim=(times[0], times[-1]), ylim=(0.0, 1.08),
                 xticks=[0.0, round(float(times[-1]), 2)], yticks=[0.0, 1.0],
                 xlabel="time", ylabel=r"relative $E_B(t),\, \Phi(t)$")
    quantity_axes(trace_ax)

    points = vertices.batch()                                                   # [V] Point-
    flux_scale = Normalize(-flux_limit, flux_limit)
    for ax, vertex_flux in zip((left_ax, mid_ax), vertex_fluxes):
        camera = offset_panel(ax, core.mv.x, core.mv.y)
        colours = flux_colours(flux_scale(vertex_flux))[:, :3]
        overlay(ax, [contours.picture(points, topology, vertex_flux, colours, camera,
                                      levels=flux_levels(flux_limit)),
                     *disk_lines(topology, points, camera)])
    return figure


def advection_frames(topology: core.Topology, vertices: core.Point,
                     flux_density: core.ScalarMap,
                     cochains: Iterable[core.Scalar]) -> Iterator[np.ndarray]:
    """Yields side-by-side ideal and resistive magnetic flux density frames at a fixed colour scale."""
    V, E, F = topology.n_elements
    cochains = iter(cochains)
    first_cochain = next(cochains)
    points = vertices.batch()                                                   # [V] Point-

    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)                # [3] Point-[F]
    areas = (core.time & ((sides[0] & sides[1]) / 2)).norm()                    # Scalar+[F]
    first_vertex_flux = vertex_averages(topology, areas, flux_density(first_cochain))
    flux_limit = float(np.percentile(np.abs(first_vertex_flux[0]), 95))
    levels = flux_levels(flux_limit)
    flux_scale = Normalize(-flux_limit, flux_limit)

    with charts.style():
        figure, axes = charts.subplots(612, 306, ncols=2)
        figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98, wspace=0.04)
        for ax in axes:
            panel_viewport(ax, vertices, 0.06)
        views = [offset_panel(ax, core.mv.x, core.mv.y) for ax in axes]
        lines = [disk_lines(topology, points, view) for view in views]
        textures = [overlay(ax, panel_lines) for ax, panel_lines in zip(axes, lines)]
    for cochain in chain((first_cochain,), cochains):
        vertex_fluxes = vertex_averages(topology, areas, flux_density(cochain))
        for texture, view, panel_lines, vertex_flux in zip(textures, views, lines, vertex_fluxes):
            colours = flux_colours(flux_scale(vertex_flux))[:, :3]
            texture.set_data(drawing.composited(
                [contours.picture(points, topology, vertex_flux, colours, view, levels=levels), *panel_lines]))
        with charts.style():
            image = capture(figure)
        yield image
    plt.close(figure)
