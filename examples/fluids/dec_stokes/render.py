"""Primal and dual mesh geometry, pressure, vorticity and tracers in a creeping channel flow."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import stack
from examples.fluids.dec_stokes import core
from examples.util.animation import capture


PRIMAL_COLOUR = "#286580"
DUAL_COLOUR = "#d77542"
MESH_COLOUR = "#c2ccd0"
WALL_COLOUR = "#52616a"
FLOW_COLOUR = "#173c4b"
PRESSURE_COLOURS = LinearSegmentedColormap.from_list(
    "dec_pressure", ("#8fb8c9", "#f5f2eb", "#df9b79"))
VORTICITY_COLOURS = LinearSegmentedColormap.from_list(
    "dec_vorticity", ("#286580", "#f5f4ef", "#df6435"))
N_STREAMLINES = 19
N_TRAIL_SAMPLES = 9
FIGURE_WIDTH, FIGURE_HEIGHT, DPI = 10.0, 3.5, 100


def coordinates(points: core.Point) -> np.ndarray:
    """Drawing coordinates of intrinsic positions and velocities."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def channel_axes(mesh: core.Mesh, vertices: core.Point, n_rows: int):
    """Equal-aspect channel panels, with the wall geometry framing each panel."""
    places = coordinates(vertices)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    figure, axes = plt.subplots(n_rows, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT * n_rows),
                                dpi=DPI, squeeze=False)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, hspace=0.07)
    low, high = places.min(axis=0), places.max(axis=0)
    margin = 0.025 * (high[1] - low[1])
    walls = places[mesh.edges[mesh.boundary_edges]]
    for ax in axes[:, 0]:
        ax.set(xlim=(low[0] - margin, high[0] + margin),
               ylim=(low[1] - margin, high[1] + margin), aspect="equal")
        ax.set_axis_off()
        ax.add_collection(LineCollection(walls, colors=WALL_COLOUR, linewidths=1.1, zorder=5))
    return figure, axes[:, 0], triangulation


def draw_mesh(mesh: core.Mesh, vertices: core.Point) -> plt.Figure:
    """The channel and its obstacle, with the triangles resolving their boundaries."""
    figure, axes, triangulation = channel_axes(mesh, vertices, 1)
    axes[0].triplot(triangulation, color=MESH_COLOUR, linewidth=0.45)
    return figure


def draw_complex(mesh: core.Mesh, vertices: core.Point, circumcentres: core.Point,
                 dual_edges: core.Point) -> plt.Figure:
    """An interior vertex and its dual polygon, with a crossing primal and dual edge."""
    places = coordinates(vertices)
    centres = coordinates(circumcentres)
    dual_directions = coordinates(dual_edges)
    low, high = places.min(axis=0), places.max(axis=0)
    target = low + np.array([0.65, 0.5]) * (high - low)
    interior = np.setdiff1d(np.arange(len(places)), mesh.edges[mesh.boundary_edges])
    vertex = interior[np.argmin(np.linalg.norm(places[interior] - target, axis=1))]
    star = np.flatnonzero(np.any(mesh.faces == vertex, axis=1))
    neighbours = np.unique(mesh.faces[star])
    radius = 1.8 * np.linalg.norm(places[neighbours] - places[vertex], axis=1).max()
    visible = np.linalg.norm(places - places[vertex], axis=1) < 1.6 * radius
    edges = np.flatnonzero(np.all(visible[mesh.edges], axis=1))

    # A dual segment joins adjacent face centres; on the boundary it ends at the edge midpoint.
    edge_faces = [[] for _ in mesh.edges]
    for face, boundary in enumerate(mesh.face_edges):
        for edge in boundary:
            edge_faces[edge].append(face)
    dual_segments = []
    for edge in edges:
        incident = edge_faces[edge]
        first = centres[incident[0]]
        second = centres[incident[1]] if len(incident) == 2 else places[mesh.edges[edge]].mean(axis=0)
        dual_segments.append([first, second])

    figure, ax = plt.subplots(figsize=(4.6, 4.6), dpi=DPI)
    figure.subplots_adjust(0.02, 0.02, 0.98, 0.98)
    ax.set(xlim=places[vertex, 0] + np.array([-radius, radius]),
           ylim=places[vertex, 1] + np.array([-radius, radius]), aspect="equal")
    ax.set_axis_off()
    ax.add_collection(LineCollection(places[mesh.edges[edges]], colors=PRIMAL_COLOUR,
                                     linewidths=0.85, alpha=0.6))
    ax.add_collection(LineCollection(dual_segments, colors=DUAL_COLOUR,
                                     linewidths=0.9, alpha=0.75))
    dual_vertices = centres[star]
    order = np.argsort(np.arctan2(*(dual_vertices - places[vertex])[:, ::-1].T))
    ax.add_patch(Polygon(dual_vertices[order], facecolor=DUAL_COLOUR, edgecolor=DUAL_COLOUR,
                         linewidth=1.8, alpha=0.17))
    face = star[np.argmin(np.linalg.norm(centres[star] - places[vertex], axis=1))]
    ax.add_patch(Polygon(places[mesh.faces[face]], facecolor=PRIMAL_COLOUR,
                         edgecolor=PRIMAL_COLOUR, linewidth=1.7, alpha=0.13))
    ax.scatter(*places[visible].T, s=9, color=PRIMAL_COLOUR, zorder=3)
    ax.scatter(*centres[np.linalg.norm(centres - places[vertex], axis=1) < 1.5 * radius].T,
               s=9, color=DUAL_COLOUR, zorder=3)
    ax.scatter(*places[vertex], s=44, color=PRIMAL_COLOUR, edgecolor="white", linewidth=0.8, zorder=5)
    ax.scatter(*centres[face], s=35, color=DUAL_COLOUR, edgecolor="white", linewidth=0.7, zorder=5)

    # The arrows carry the actual orientation of the two crossing geometric elements.
    boundary = mesh.face_edges[face]
    face_edges = places[mesh.edges[boundary]]
    edge = boundary[np.argmax(np.linalg.norm(face_edges[:, 1] - face_edges[:, 0], axis=1))]
    ends = places[mesh.edges[edge]]
    incident = edge_faces[edge]
    dual_middle = centres[incident].mean(axis=0)
    for middle, direction, colour in ((ends.mean(axis=0), ends[1] - ends[0], PRIMAL_COLOUR),
                                       (dual_middle, dual_directions[edge], DUAL_COLOUR)):
        ax.annotate("", xy=middle + 0.35 * direction, xytext=middle - 0.35 * direction,
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=colour, linewidth=2),
                    zorder=6)
    return figure


def draw_fields(mesh: core.Mesh, vertices: core.Point, pressure: core.Scalar, vorticity: core.Bivector,
                streamfunction: core.Scalar) -> plt.Figure:
    """Pressure above and signed vorticity below, with the same flow paths on both."""
    figure, axes, triangulation = channel_axes(mesh, vertices, 2)
    values = np.asarray(pressure.batch().to_array())
    mv = vorticity.context.multivector
    rotation = np.asarray((mv.xy.dual() & vorticity).batch().to_array())
    potential = np.asarray(streamfunction.batch().to_array())
    levels = np.linspace(potential.min(), potential.max(), N_STREAMLINES + 2)[1:-1]
    extent = np.quantile(np.abs(rotation), 0.97)
    axes[0].tripcolor(triangulation, facecolors=values, shading="flat", cmap=PRESSURE_COLOURS,
                      norm=Normalize(values.min(), values.max()), zorder=1)
    axes[1].tripcolor(triangulation, rotation, shading="gouraud", cmap=VORTICITY_COLOURS,
                      norm=Normalize(-extent, extent), zorder=1)
    for ax in axes:
        ax.tricontour(triangulation, potential, levels=levels, linestyles="solid",
                      colors=FLOW_COLOUR, linewidths=0.65, alpha=0.65, zorder=3)
    return figure


def animate_flow(mesh: core.Mesh, vertices: core.Point, pressure: core.Scalar, streamfunction: core.Scalar,
                 velocity: core.Point, n_frames: int) -> Iterator[np.ndarray]:
    """Tracers travelling along the conserved flux's streamlines over the steady pressure field."""
    figure, axes, triangulation = channel_axes(mesh, vertices, 1)
    ax = axes[0]
    values = np.asarray(pressure.batch().to_array())
    potential = np.asarray(streamfunction.batch().to_array())
    velocity_xy = coordinates(velocity)
    speeds = np.linalg.norm(velocity_xy, axis=-1)
    levels = np.linspace(potential.min(), potential.max(), N_STREAMLINES + 2)[1:-1]
    ax.tripcolor(triangulation, facecolors=values, shading="flat", cmap=PRESSURE_COLOURS,
                  norm=Normalize(values.min(), values.max()), zorder=1)
    contours = ax.tricontour(triangulation, potential, levels=levels, linestyles="solid",
                            colors=FLOW_COLOUR, linewidths=0.45, alpha=0.22, zorder=2)
    finder = triangulation.get_trifinder()
    paths, clocks = [], []
    for segments in contours.allsegs:
        for path in segments:
            if len(path) < 2:
                continue
            differences = np.diff(path, axis=0)
            midpoints = (path[1:] + path[:-1]) / 2
            faces = finder(*midpoints.T)
            # Contours may be listed in either sense; the velocity determines travel direction.
            if np.sum(differences * velocity_xy[faces]) < 0:
                path = path[::-1]
                differences, faces = -differences[::-1], faces[::-1]
            durations = np.linalg.norm(differences, axis=1) / speeds[faces]
            paths.append(path)
            clocks.append(np.concatenate(([0], np.cumsum(durations))))
    period = np.median([clock[-1] for clock in clocks]) / 6
    trail_time = 0.2 * period
    trails = LineCollection([], colors=FLOW_COLOUR, linewidths=1.8, capstyle="round", zorder=4)
    ax.add_collection(trails)
    heads = ax.scatter([], [], s=7, color=FLOW_COLOUR, zorder=4)
    for frame in range(n_frames):
        elapsed = frame * period / n_frames
        segments, tips = [], []
        for index, (path, clock) in enumerate(zip(paths, clocks)):
            arrivals = np.arange(-1, int(np.ceil(clock[-1] / period)) + 1) * period
            arrivals += elapsed + period * ((index * 0.61803398875) % 1)
            arrivals = arrivals[(arrivals >= 0) & (arrivals <= clock[-1])]
            samples = np.maximum(0, arrivals[:, None] - np.linspace(trail_time, 0, N_TRAIL_SAMPLES))
            strokes = np.stack([np.interp(samples, clock, path[:, axis]) for axis in range(2)], axis=-1)
            segments.extend(strokes)
            tips.extend(strokes[:, -1])
        trails.set_segments(segments)
        heads.set_offsets(tips)
        yield capture(figure)
    plt.close(figure)
