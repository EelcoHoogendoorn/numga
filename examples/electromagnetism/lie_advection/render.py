"""Swept edge ribbons, magnetic flux density fields, and side-by-side Lie advection animations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import stack
from examples.electromagnetism.lie_advection import core
from examples.util import charts
from examples.util.animation import capture

primal_colour = "#286580"
accent_colour = "#d77542"
ink_colour = "#173c4b"
teal_colour = "#148d88"
mesh_colour = "#b8c4c9"
wall_colour = "#52616a"
flux_colours = LinearSegmentedColormap.from_list(
    "lie_flux", ("#184a63", "#5b9ab5", "#f6f4ee", "#df7a47", "#a6341b"))
n_flux_contours = 8
# The velocity arrows bolder than the chart's lines; the swept ribbons' dotted outlines and the mesh as thin as its
# frame, the mesh faint.
velocity_width = charts.points(1.5 * charts.LINE_PX)
mesh_alpha = 0.16


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


def plain_axes(ax: plt.Axes) -> None:
    """Minimal left and bottom spines for quantity plots."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.78")
    ax.tick_params(color="0.78")


def vertex_averages(mesh: core.Mesh, areas: core.Scalar, face_field: core.Scalar) -> np.ndarray:
    """Area-weighted vertex average of a scalar face field for smooth Gouraud shading."""
    corner_mean = mesh.incidence[2, 0].mean(axis=0)
    vertex_field = (~corner_mean * (areas * face_field)) / (~corner_mean * areas)
    return np.asarray(vertex_field.batch().to_array())


@charts.style()
def draw_cartan_ribbon(corners: core.Point, points: core.Point, sample_forms: core.Plane,
                       corner_velocities: core.Point, ribbon_step: float,
                       endpoint_differences: core.Scalar, swept_fluxes: core.Scalar) -> plt.Figure:
    """Swept edge parallelogram ribbons on a moving triangle beside the Cartan rate decomposition."""
    corner_xy = coordinates(corners)
    displaced_xy = coordinates(corners + ribbon_step * corner_velocities)
    sample_xy = coordinates(points)
    field_xy = coordinates(sample_forms.dual())
    endpoint_vals = np.asarray(endpoint_differences.to_array())
    swept_vals = np.asarray(swept_fluxes.to_array())
    total_vals = endpoint_vals + swept_vals

    # Three quarters of the column: the triangle in a square beside the bars of its three edges and their sum.
    figure, (geom_ax, bar_ax) = charts.subplots(600, 270, ncols=2, width_ratios=(1, 1.2), layout="constrained")

    all_places = np.vstack((corner_xy, displaced_xy))
    frame_panel(geom_ax, all_places, 0.16)
    flux_scale = max(np.abs(swept_vals).max(), 1e-8)
    for index in range(3):
        first, second = (index + 1) % 3, (index + 2) % 3
        ribbon_xy = np.array([corner_xy[first], corner_xy[second],
                              displaced_xy[second], displaced_xy[first]])
        colour = flux_colours(0.5 + 0.42 * swept_vals[index] / flux_scale)
        geom_ax.add_patch(Polygon(ribbon_xy, facecolor=colour, edgecolor=wall_colour,
                                  linewidth=charts.FRAME_WIDTH, linestyle=":", alpha=0.72, zorder=1))

    geom_ax.add_patch(Polygon(corner_xy, facecolor="#f6f4ee", edgecolor=wall_colour, alpha=0.88, zorder=2))
    geom_ax.add_patch(Polygon(displaced_xy, fill=False, edgecolor=ink_colour, linestyle="--", zorder=4))
    max_norm = np.linalg.norm(field_xy, axis=-1).max()
    geom_ax.quiver(sample_xy[:, 0], sample_xy[:, 1], field_xy[:, 0], field_xy[:, 1],
                   color=primal_colour, angles="xy", scale_units="xy", scale=6.2 * max_norm,
                   width=0.0085, headwidth=3.6, headlength=4.3, pivot="mid", zorder=3)

    for index in range(3):
        first, second = (index + 1) % 3, (index + 2) % 3
        middle = (corner_xy[first] + corner_xy[second]) / 2
        edge_vec = corner_xy[second] - corner_xy[first]
        geom_ax.annotate("", xy=middle + 0.16 * edge_vec, xytext=middle - 0.16 * edge_vec,
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


def draw_disk_panel(ax: plt.Axes, triangulation: Triangulation, places: np.ndarray,
                    walls: np.ndarray, vertex_flux: np.ndarray,
                    flux_levels: np.ndarray, flux_limit: float) -> None:
    """One disk panel shaded by magnetic flux density with evolving iso-flux contours."""
    frame_panel(ax, places, 0.06)
    ax.tripcolor(triangulation, vertex_flux, shading="gouraud", cmap=flux_colours,
                 norm=Normalize(-flux_limit, flux_limit), zorder=1)
    ax.triplot(triangulation, color=mesh_colour, linewidth=charts.FRAME_WIDTH, alpha=mesh_alpha, zorder=2)
    active = flux_levels[(flux_levels > vertex_flux.min()) & (flux_levels < vertex_flux.max())]
    if len(active):
        ax.tricontour(triangulation, vertex_flux, levels=active,
                      colors=ink_colour, linestyles="solid", alpha=0.68, zorder=3)
    ax.add_collection(LineCollection(walls, colors=wall_colour, zorder=4))


@charts.style()
def draw_advection_comparison(mesh: core.Mesh, vertices: core.Point,
                              flux_density: core.Scalar, times: np.ndarray,
                              energies: core.Scalar, total_fluxes: core.Scalar) -> plt.Figure:
    """Ideal versus resistive magnetic flux advection alongside magnetic energy and total flux traces."""
    places = coordinates(vertices)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = places[mesh.edges[mesh.boundary_edges]]

    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)
    areas = (core.time & (sides[0] & sides[1]) / 2).norm()
    vertex_fluxes = vertex_averages(mesh, areas, flux_density)
    flux_limit = float(np.percentile(np.abs(vertex_fluxes[0]), 96))
    half_levels = np.linspace(0.16 * flux_limit, 0.88 * flux_limit, n_flux_contours // 2)
    flux_levels = np.concatenate((-half_levels[::-1], half_levels))

    # Nine tenths of the column: the two disks square at the chart's full height, the traces beside them in a
    # layout of their own, so that the room for the traces' labels does not shrink the disks.
    figure = plt.figure(figsize=(720 / charts.DPI, 250 / charts.DPI), layout="constrained")
    disks, traces = figure.subfigures(1, 2, width_ratios=(2, 0.9))
    left_ax, mid_ax = disks.subplots(1, 2)
    trace_ax = traces.subplots()

    for index, ax in enumerate((left_ax, mid_ax)):
        draw_disk_panel(ax, triangulation, places, walls, vertex_fluxes[index],
                        flux_levels, flux_limit)

    energy_vals = np.asarray(energies.to_array())
    flux_vals = np.asarray(total_fluxes.to_array())
    trace_ax.plot(times, energy_vals[:, 0] / energy_vals[0, 0], color=ink_colour)
    trace_ax.plot(times, energy_vals[:, 1] / energy_vals[0, 0], color=accent_colour)
    trace_ax.plot(times, flux_vals[:, 1] / flux_vals[0, 1], color=teal_colour, linestyle="--")
    trace_ax.set(xlim=(times[0], times[-1]), ylim=(0.0, 1.08),
                 xticks=[0.0, round(float(times[-1]), 2)], yticks=[0.0, 1.0],
                 xlabel="time", ylabel=r"relative $E_B(t),\, \Phi(t)$")
    plain_axes(trace_ax)
    return figure


def animate_advection(mesh: core.Mesh, vertices: core.Point,
                      frames: Iterable[core.Scalar]
                      ) -> Iterator[np.ndarray]:
    """Side-by-side ideal and resistive magnetic flux winding frames captured at fixed colour scale."""
    frames = iter(frames)
    first = next(frames)
    places = coordinates(vertices)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = places[mesh.edges[mesh.boundary_edges]]

    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)
    areas = (core.time & (sides[0] & sides[1]) / 2).norm()
    first_vertex_flux = vertex_averages(mesh, areas, first)
    flux_limit = float(np.percentile(np.abs(first_vertex_flux[0]), 95))
    half_levels = np.linspace(0.16 * flux_limit, 0.88 * flux_limit, n_flux_contours // 2)
    flux_levels = np.concatenate((-half_levels[::-1], half_levels))

    for flux_density in chain((first,), frames):
        vertex_fluxes = vertex_averages(mesh, areas, flux_density)
        # The disks drawn in the charts' strokes, as in the comparison of their final states.
        with charts.style():
            figure, axes = charts.subplots(612, 306, ncols=2)
            figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98, wspace=0.04)
            for index, ax in enumerate(axes):
                draw_disk_panel(ax, triangulation, places, walls, vertex_fluxes[index],
                                flux_levels, flux_limit)
            image = capture(figure)
        plt.close(figure)
        yield image
