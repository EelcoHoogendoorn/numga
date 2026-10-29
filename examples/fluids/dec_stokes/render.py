"""Primal and dual mesh geometry, pressure, vorticity and tracers in a creeping channel flow."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import concatenate, stack
from examples.fluids.dec_stokes import core
from examples.util.animation import capture


primal_colour = "#286580"
dual_colour = "#d77542"
mesh_colour = "#c2ccd0"
wall_colour = "#52616a"
flow_colour = "#173c4b"
pressure_colours = LinearSegmentedColormap.from_list(
    "dec_pressure", ("#8fb8c9", "#f5f2eb", "#df9b79"))
vorticity_colours = LinearSegmentedColormap.from_list(
    "dec_vorticity", ("#286580", "#f5f4ef", "#df6435"))
n_streamlines = 19
n_trail_samples = 9
figure_width, figure_height, dpi = 5.0, 1.75, 100
mv = core.mv
# The planes x = 0 and y = 0: a point's pairings with them are its places across the picture, and a place so far along
# each is their duals so weighted.
axes = stack((mv.x, mv.y))                                                   # [2] Vector+
# Where the complex's vertex lies, as shares of the channel's extent along x and y; how far its arrows reach either
# way of their middles, in lengths of the edges they show.
target_shares = np.array([0.65, 0.5])
arrow_reach = 0.35


def coordinates(points: core.Point) -> np.ndarray:
    """Drawing coordinates of intrinsic positions and velocities."""
    return np.asarray((axes & points.batch()[..., None]).to_array())


def channel_axes(mesh: core.Mesh, vertices: core.Point, n_rows: int):
    """Equal-aspect channel panels, with the wall geometry framing each panel."""
    places = coordinates(vertices)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    figure, panels = plt.subplots(n_rows, 1, figsize=(figure_width, figure_height * n_rows),
                                  dpi=dpi, squeeze=False)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, hspace=0.07)
    low, high = places.min(axis=0), places.max(axis=0)
    margin = 0.025 * (high[1] - low[1])
    # The walls are the boundary edges, each from its tail to its head.
    tails, heads = mesh.reading(mesh.exterior(1), 1) * (mesh.incidence[1, 0] * vertices)   # Point-[walls], Point-[walls]
    walls = np.stack((coordinates(tails), coordinates(heads)), axis=1)
    for ax in panels[:, 0]:
        ax.set(xlim=(low[0] - margin, high[0] + margin),
               ylim=(low[1] - margin, high[1] + margin), aspect="equal")
        ax.set_axis_off()
        ax.add_collection(LineCollection(walls, colors=wall_colour, linewidths=0.55, zorder=5))
    return figure, panels[:, 0], triangulation


def draw_mesh(mesh: core.Mesh, vertices: core.Point) -> plt.Figure:
    """The channel and its obstacle, with the triangles resolving their boundaries."""
    figure, panels, triangulation = channel_axes(mesh, vertices, 1)
    panels[0].triplot(triangulation, color=mesh_colour, linewidth=0.25)
    return figure


def draw_complex(mesh: core.Mesh, vertices: core.Point, circumcentres: core.Point,
                 dual_edges: core.Point) -> plt.Figure:
    """An interior vertex and its dual polygon, with a crossing primal and dual edge."""
    places = coordinates(vertices)
    low, high = places.min(axis=0), places.max(axis=0)
    target = (axes.dual() * (low + target_shares * (high - low))).sum()      # [] Point-
    interior = mesh.interior(0)
    interior_points = (mesh.reading(interior) * vertices).batch()          # [interior] Point-
    vertex = interior[(interior_points - target).norm().argmin()]
    centre = (mesh.reading(np.array([vertex])) * vertices).batch()[0]       # [] Point-
    # The faces at the vertex, and their corners: what the corner incidence and its reverse reach from a field that
    # is one at the vertex alone.
    corners = mesh.incidence[2, 0].sum(axis=0)                                   # Scalar+[faces <- vertices]
    marked = ~mesh.reading(np.array([vertex])) * mv.scalar([[1]]).field()        # Scalar+[vertices]
    at_vertex = corners * marked                                                 # Scalar+[faces]
    star = np.flatnonzero(at_vertex.batch() > 0)
    neighbours = np.flatnonzero((~corners * at_vertex).batch() > 0)
    reach = ((mesh.reading(neighbours) * vertices).batch() - centre).norm()     # [neighbours] Scalar+
    radius = 1.8 * float(reach.to_array().max())
    visible = (vertices.batch() - centre).norm() < 1.6 * radius
    # The edges with both ends visible: the edge-vertex incidence counts an edge's visible ends.
    visible_ends = mesh.incidence[1, 0].sum(axis=0) * mv.scalar(visible[:, None] * 1.0).field()   # Scalar+[edges]
    shown = mesh.reading(np.flatnonzero(visible_ends.batch() > 1.5), 1)          # Scalar+[shown <- edges]

    # A dual segment joins the circumcentres of the faces either side of an interior edge; on the boundary it runs
    # from the one face's circumcentre to the edge's midpoint. Summed onto the edges through the face-edge incidence,
    # the circumcentres give a boundary edge its one face's.
    inside, rim = mesh.reading(mesh.interior_edges, 1), mesh.reading(mesh.exterior(1), 1)
    on_left, on_right = mesh.reading(mesh.edge_faces.T, 2) * circumcentres     # Point-[interior edges], Point-[interior edges]
    beside = (~mesh.incidence[2, 1] * circumcentres).sum(axis=0)                # Point-[edges]
    midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices                    # Point-[edges]
    dual_starts = ~inside * on_left + ~rim * (rim * beside)                      # Point-[edges]
    dual_ends = ~inside * on_right + ~rim * (rim * midpoints)                    # Point-[edges]

    figure, ax = plt.subplots(figsize=(3.1, 3.1), dpi=dpi)
    figure.subplots_adjust(0.02, 0.02, 0.98, 0.98)
    middle = coordinates(centre)
    ax.set(xlim=middle[0] + np.array([-radius, radius]),
           ylim=middle[1] + np.array([-radius, radius]), aspect="equal")
    ax.set_axis_off()
    primal_tails, primal_heads = shown * (mesh.incidence[1, 0] * vertices)       # Point-[shown], Point-[shown]
    ax.add_collection(LineCollection(np.stack((coordinates(primal_tails), coordinates(primal_heads)), axis=1),
                                     colors=primal_colour, linewidths=0.57, alpha=0.6))
    ax.add_collection(LineCollection(np.stack((coordinates(shown * dual_starts), coordinates(shown * dual_ends)), axis=1),
                                     colors=dual_colour, linewidths=0.6, alpha=0.75))
    star_centres = (mesh.reading(star, 2) * circumcentres).batch()             # [star] Point-
    spokes = star_centres - centre                                             # [star] Point-
    # Each spoke's turn from the first, read from the log of their product, orders the dual cell's corners around the
    # vertex.
    turns = (spokes[0] * spokes).normalized().log()                            # [star] Bivector+
    order = (mv.xy.dual() & turns).argsort()
    ax.add_patch(Polygon(coordinates(star_centres[order]), facecolor=dual_colour, edgecolor=dual_colour,
                         linewidth=1.2, alpha=0.17))
    nearest = spokes.norm().argmin()
    face = star[nearest]
    face_corners = mesh.reading(np.array([face]), 2) * (mesh.incidence[2, 0] * vertices)   # [3] Point-[1]
    ax.add_patch(Polygon(coordinates(face_corners)[:, 0], facecolor=primal_colour,
                         edgecolor=primal_colour, linewidth=1.15, alpha=0.13))
    ax.scatter(*places[visible].T, s=4, color=primal_colour, zorder=3)
    centres = circumcentres.batch()                                            # [faces] Point-
    ax.scatter(*coordinates(centres[(centres - centre).norm() < 1.5 * radius]).T,
               s=4, color=dual_colour, zorder=3)
    ax.scatter(*middle, s=20, color=primal_colour, edgecolor="white", linewidth=0.55, zorder=5)
    ax.scatter(*coordinates(star_centres[nearest]), s=16, color=dual_colour, edgecolor="white", linewidth=0.47,
               zorder=5)

    # The arrows carry the actual orientation of the two crossing geometric elements: the face's longest edge from
    # its midpoint, and its dual edge from the middle of its segment.
    edges = mesh.d[0] * vertices                                               # Point-[edges]
    sides = mesh.face_edges[face]
    longest = sides[(mesh.reading(sides, 1) * edges).batch().norm().argmax()]
    edge = mesh.reading(np.array([longest]), 1)                                # Scalar+[1 <- edges]
    for middle, direction, colour in ((edge * midpoints, edge * edges, primal_colour),
                                       (edge * (dual_starts + dual_ends) / 2, edge * dual_edges, dual_colour)):
        tail, head = middle - arrow_reach * direction, middle + arrow_reach * direction   # Point-[1], Point-[1]
        ax.annotate("", xy=coordinates(head)[0], xytext=coordinates(tail)[0],
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=8, color=colour, linewidth=1.35),
                    zorder=6)
    return figure


def draw_fields(mesh: core.Mesh, vertices: core.Point, pressure: core.Scalar, vorticity: core.Bivector,
                streamfunction: core.Scalar) -> plt.Figure:
    """Pressure above and signed vorticity below, with the same flow paths on both."""
    figure, panels, triangulation = channel_axes(mesh, vertices, 2)
    values = np.asarray(pressure.batch().to_array())
    rotation = np.asarray((mv.xy.dual() & vorticity).batch().to_array())
    potential = np.asarray(streamfunction.batch().to_array())
    levels = np.linspace(potential.min(), potential.max(), n_streamlines + 2)[1:-1]
    extent = np.quantile(np.abs(rotation), 0.97)
    panels[0].tripcolor(triangulation, facecolors=values, shading="flat", cmap=pressure_colours,
                        norm=Normalize(values.min(), values.max()), zorder=1)
    panels[1].tripcolor(triangulation, rotation, shading="gouraud", cmap=vorticity_colours,
                        norm=Normalize(-extent, extent), zorder=1)
    for ax in panels:
        ax.tricontour(triangulation, potential, levels=levels, linestyles="solid",
                      colors=flow_colour, linewidths=0.33, alpha=0.65, zorder=3)
    return figure


def animate_flow(mesh: core.Mesh, vertices: core.Point, pressure: core.Scalar, streamfunction: core.Scalar,
                 velocity: core.Point, n_frames: int) -> Iterator[np.ndarray]:
    """Tracers travelling along the conserved flux's streamlines over the steady pressure field."""
    figure, panels, triangulation = channel_axes(mesh, vertices, 1)
    ax = panels[0]
    values = np.asarray(pressure.batch().to_array())
    potential = np.asarray(streamfunction.batch().to_array())
    levels = np.linspace(potential.min(), potential.max(), n_streamlines + 2)[1:-1]
    ax.tripcolor(triangulation, facecolors=values, shading="flat", cmap=pressure_colours,
                 norm=Normalize(values.min(), values.max()), zorder=1)
    contours = ax.tricontour(triangulation, potential, levels=levels, linestyles="solid",
                             colors=flow_colour, linewidths=0.23, alpha=0.22, zorder=2)
    finder = triangulation.get_trifinder()
    lines, clocks = [], []
    for segments in contours.allsegs:
        for path in segments:
            if len(path) < 2:
                continue
            # The streamline's corners as points of the plane, its steps between them, and the face each step's
            # middle lies in.
            line = (axes.dual() * path).sum(axis=-1)                           # [corners] Point-
            steps = line[1:] - line[:-1]                                       # [steps] Point-
            faces = finder(*coordinates((line[1:] + line[:-1]) / 2).T)
            flow = (mesh.reading(faces, 2) * velocity).batch()                 # [steps] Point-
            # Contours may be listed in either sense; the velocity along the steps determines travel direction.
            if steps.dot(flow).sum(axis=0) < 0:
                line, steps, flow = line[::-1], -steps[::-1], flow[::-1]
            # Each step takes its length over the speed of the flow through its face.
            durations = (steps.norm() / flow.norm()).to_array()
            lines.append(line)
            clocks.append(np.concatenate(([0], np.cumsum(durations))))
    # All streamlines' corners in one batch, each line's first corner at its offset.
    offsets = np.cumsum([0] + [len(clock) for clock in clocks])[:-1]
    corners = concatenate(lines)                                               # [corners] Point-
    period = np.median([clock[-1] for clock in clocks]) / 6
    trail_time = 0.2 * period
    trails = LineCollection([], colors=flow_colour, linewidths=0.9, capstyle="round", zorder=4)
    ax.add_collection(trails)
    heads = ax.scatter([], [], s=1.75, color=flow_colour, zorder=4)
    for frame in range(n_frames):
        elapsed = frame * period / n_frames
        stroke_steps, stroke_shares = [], []
        for index, (offset, clock) in enumerate(zip(offsets, clocks)):
            arrivals = np.arange(-1, int(np.ceil(clock[-1] / period)) + 1) * period
            arrivals += elapsed + period * ((index * 0.61803398875) % 1)
            arrivals = arrivals[(arrivals >= 0) & (arrivals <= clock[-1])]
            samples = np.maximum(0, arrivals[:, None] - np.linspace(trail_time, 0, n_trail_samples))
            # The step of the streamline each sample's time falls in, and how far along it.
            step = np.clip(np.searchsorted(clock, samples, side="right") - 1, 0, len(clock) - 2)
            stroke_steps.append(offset + step)
            stroke_shares.append((samples - clock[step]) / (clock[step + 1] - clock[step]))
        step, share = np.concatenate(stroke_steps), np.concatenate(stroke_shares)   # [strokes, trail]
        # Each tracer's trail: points along the streamline's steps, at the shares their times reach.
        strokes = coordinates(corners[step] + (corners[step + 1] - corners[step]) * share)
        trails.set_segments(strokes)
        heads.set_offsets(strokes[:, -1])
        yield capture(figure)
    plt.close(figure)
