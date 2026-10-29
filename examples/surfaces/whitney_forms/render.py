"""Barycentric hat ramps, edge Whitney 1-forms, and anisotropic conduction fields."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import stack
from examples.util import charts
from examples.surfaces.whitney_forms import core

primal_colour = "#286580"
accent_colour = "#d77542"
mesh_colour = "#b8c4c9"
wall_colour = "#52616a"
flow_colour = "#173c4b"
fibre_colour = "#7f8f96"
hat_colours = LinearSegmentedColormap.from_list(
    "whitney_hat", ("#f6f4ee", "#8fb8c9", "#286580"))
potential_colours = LinearSegmentedColormap.from_list(
    "whitney_potential", ("#286580", "#f5f2eb", "#d77542"))
n_contours = 7
n_streamlines = 15
# The pixel sizes of a row of four triangles and of the two disks side by side.
row_size, flow_size = (780, 195), (600, 290)


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


def draw_hat_0forms(corners: core.Point, gradients: core.Plane,
                    star_mesh: core.Mesh, star_vertices: core.Point,
                    vertex_hat: core.Scalar, star_gradients: core.Plane) -> plt.Figure:
    """The three corner hat ramps with their constant gradients, and the assembled vertex pyramid."""
    corner_xy = coordinates(corners)
    gradient_xy = coordinates(gradients.dual())
    single_tri = Triangulation(*corner_xy.T, triangles=np.array([[0, 1, 2]]))
    centroid = corner_xy.mean(axis=0)
    levels = np.linspace(0.0, 1.0, n_contours + 2)[1:-1]

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        frame_panel(ax, corner_xy, 0.22)
        values = np.eye(3)[index]
        ax.tripcolor(single_tri, values, shading="gouraud", cmap=hat_colours,
                     norm=Normalize(0.0, 1.0), zorder=1)
        ax.tricontour(single_tri, values, levels=levels, colors=primal_colour,
                      linewidths=0.75, alpha=0.55, zorder=2)
        ax.add_patch(Polygon(corner_xy, fill=False, edgecolor=wall_colour, linewidth=1.2, zorder=3))
        opposite = corner_xy[[(index + 1) % 3, (index + 2) % 3]]
        ax.plot(*opposite.T, color=wall_colour, linewidth=2.2, solid_capstyle="round", zorder=4)
        ax.scatter(*corner_xy.T, s=18, color=wall_colour, zorder=5)
        ax.scatter(*corner_xy[index], s=48, color=accent_colour, edgecolor="white", linewidth=0.9, zorder=6)
        direction = 0.24 * gradient_xy[index] / np.linalg.norm(gradient_xy, axis=-1).max()
        ax.annotate("", xy=centroid + direction, xytext=centroid - direction,
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=accent_colour, linewidth=2.0),
                    zorder=7)

    ax = axes[3]
    star_xy = coordinates(star_vertices)
    star_tri = Triangulation(*star_xy.T, triangles=star_mesh.faces)
    hat_values = np.asarray(vertex_hat.batch().to_array())
    star_grad_xy = coordinates(star_gradients.dual())
    star_centres = star_xy[star_mesh.faces].mean(axis=1)
    frame_panel(ax, star_xy, 0.14)
    ax.tripcolor(star_tri, hat_values, shading="gouraud", cmap=hat_colours,
                 norm=Normalize(0.0, 1.0), zorder=1)
    ax.tricontour(star_tri, hat_values, levels=levels, colors=primal_colour,
                  linewidths=0.7, alpha=0.55, zorder=2)
    ax.triplot(star_tri, color=wall_colour, linewidth=0.65, alpha=0.65, zorder=3)
    ax.scatter(*star_xy[0], s=48, color=accent_colour, edgecolor="white", linewidth=0.9, zorder=6)
    active = np.linalg.norm(star_grad_xy, axis=-1) > 1e-8
    scale = 0.16 / np.linalg.norm(star_grad_xy[active], axis=-1).max()
    for centre, vector in zip(star_centres[active], star_grad_xy[active]):
        delta = scale * vector
        ax.annotate("", xy=centre + delta, xytext=centre - delta,
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=10, color=accent_colour, linewidth=1.6),
                    zorder=7)
    return figure


def draw_edge_1forms(corners: core.Point, points: core.Point, edge_forms: core.Plane,
                     left_corners: core.Point, left_points: core.Point, left_shared: core.Plane,
                     right_corners: core.Point, right_points: core.Point, right_shared: core.Plane) -> plt.Figure:
    """The three edge Whitney 1-forms on a single triangle and the shared-edge basis field across two faces."""
    corner_xy = coordinates(corners)
    sample_xy = coordinates(points)
    field_xy = coordinates(edge_forms.dual())
    max_norm = np.linalg.norm(field_xy, axis=-1).max()

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        frame_panel(ax, corner_xy, 0.22)
        ax.add_patch(Polygon(corner_xy, facecolor="#f6f4ee", edgecolor=wall_colour, linewidth=1.1, zorder=1))
        ax.quiver(sample_xy[:, 0], sample_xy[:, 1], field_xy[index, :, 0], field_xy[index, :, 1],
                  color=primal_colour, angles="xy", scale_units="xy", scale=6.2 * max_norm,
                  width=0.009, headwidth=3.8, headlength=4.5, pivot="mid", zorder=3)
        start, end = corner_xy[(index + 1) % 3], corner_xy[(index + 2) % 3]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=accent_colour, linewidth=2.4, zorder=4)
        middle, edge_vec = (start + end) / 2, end - start
        ax.annotate("", xy=middle + 0.18 * edge_vec, xytext=middle - 0.18 * edge_vec,
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=accent_colour, linewidth=2.0),
                    zorder=5)
        ax.scatter(*corner_xy[index], s=42, color=primal_colour, edgecolor="white", linewidth=0.9, zorder=6)

    ax = axes[3]
    left_xy, right_xy = coordinates(left_corners), coordinates(right_corners)
    both_corners = np.vstack((left_xy, right_xy))
    frame_panel(ax, both_corners, 0.16)
    for tri_xy in (left_xy, right_xy):
        ax.add_patch(Polygon(tri_xy, facecolor="#f6f4ee", edgecolor=wall_colour, linewidth=1.1, zorder=1))
    diamond_samples = np.vstack((coordinates(left_points), coordinates(right_points)))
    diamond_vectors = np.vstack((coordinates(left_shared.dual()), coordinates(right_shared.dual())))
    diamond_max = np.linalg.norm(diamond_vectors, axis=-1).max()
    ax.quiver(diamond_samples[:, 0], diamond_samples[:, 1],
              diamond_vectors[:, 0], diamond_vectors[:, 1],
              color=primal_colour, angles="xy", scale_units="xy", scale=5.5 * diamond_max,
              width=0.0085, headwidth=3.8, headlength=4.5, pivot="mid", zorder=3)
    start, end = left_xy[0], left_xy[1]
    ax.plot([start[0], end[0]], [start[1], end[1]], color=accent_colour, linewidth=2.4, zorder=4)
    middle, edge_vec = (start + end) / 2, end - start
    ax.annotate("", xy=middle + 0.2 * edge_vec, xytext=middle - 0.2 * edge_vec,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=accent_colour, linewidth=2.0),
                zorder=5)
    ax.scatter(*left_xy[2], s=38, color=primal_colour, edgecolor="white", linewidth=0.9, zorder=6)
    ax.scatter(*right_xy[2], s=38, color=primal_colour, edgecolor="white", linewidth=0.9, zorder=6)
    return figure


def draw_anisotropic_flow(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                          potential: core.Scalar, streamfunction: core.Scalar,
                          current: core.Point, fibre: core.Point) -> plt.Figure:
    """Isotropic versus tilted-anisotropic potential contours, streamlines, and current vectors."""
    places = coordinates(vertices)
    centre_xy = coordinates(centres)
    fibre_xy = coordinates(fibre)
    potentials = np.asarray(potential.batch().to_array())
    streams = np.asarray(streamfunction.batch().to_array())
    currents = coordinates(current)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = places[mesh.edges[mesh.boundary_edges]]
    potential_levels = np.linspace(-0.85, 0.85, n_contours + 2)[1:-1]

    # Select well-spaced interior face centroids so current arrows neither clump nor cross the rim.
    arrow_faces = []
    for face in np.argsort(np.linalg.norm(centre_xy, axis=-1)):
        if np.linalg.norm(centre_xy[face]) < 0.88 and all(
            np.linalg.norm(centre_xy[face] - centre_xy[chosen]) > 0.18 for chosen in arrow_faces
        ):
            arrow_faces.append(face)

    figure, axes = charts.subplots(*flow_size, ncols=2)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98, wspace=0.05)

    for index, ax in enumerate(axes):
        frame_panel(ax, places, 0.08)
        ax.tripcolor(triangulation, potentials[index], shading="gouraud", cmap=potential_colours,
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
        if index == 1:
            # Faint background chords indicate the tilted high-conductivity fibre orientation.
            normal_xy = np.array([-fibre_xy[1], fibre_xy[0]])
            offsets = np.linspace(-0.85, 0.85, 9)
            half_chords = np.sqrt(np.maximum(0.0, 1.0 - offsets**2))
            chords = np.stack((offsets[:, None] * normal_xy - half_chords[:, None] * fibre_xy,
                               offsets[:, None] * normal_xy + half_chords[:, None] * fibre_xy), axis=1)
            ax.add_collection(LineCollection(chords, colors=fibre_colour, linewidths=0.65,
                                             linestyles="dotted", alpha=0.55, zorder=2))
    return figure
