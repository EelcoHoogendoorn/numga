"""Primal and dual mesh geometry, pressure, vorticity and tracers in a creeping channel flow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import concatenate, stack
from examples.fluids.dec_stokes import core
from examples.util.wgsl import contours, flow, simplices
from examples.util.wgsl.camera import Camera, flat_panel


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
figure_width, figure_height, dpi = 5.0, 1.75, 100
mv = core.mv
# The planes x = 0 and y = 0: a point's pairings with them are its places across the picture, and a place so far along
# each is their duals so weighted.
axes = stack((mv.x, mv.y))                                                   # [2] Vector+
# Where the complex's vertex lies, as shares of the channel's extent along x and y; how far its arrows reach either
# way of their middles, in lengths of the primal edge.
target_shares = np.array([0.65, 0.5])
arrow_reach = 0.35
# Widths, radii and lengths in pixels: the walls and the mesh's edges; the complex's primal and dual edges, cells,
# vertices, circumcentres, and arrows and the strokes of the chevrons at their tips.
wall_width, mesh_width = 0.76, 0.35
primal_width, dual_width, cell_width, vertex_radius = 0.8, 0.83, 1.6, 1.4
marked_radius, marked_ring = 3.1, 0.76
arrow_width, arrow_chevron = 1.9, 7.6


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """A named colour's red, green and blue."""
    return np.array(to_rgb(colour))


def face_corners(mesh: core.Mesh, vertices: core.Point) -> tuple[core.Point, core.Mesh]:
    """Each face with its own three corners, for a field shaded per face: the corners, corner after corner of every
    face, `[corners * faces]`, and the faces among them."""
    n_faces = len(mesh.faces)
    corners = (mesh.incidence[2, 0] * vertices).batch().reshape(-1)          # [corners * faces] Point-
    split = core.Mesh(3 * n_faces, np.arange(3 * n_faces).reshape(3, n_faces).T)
    return corners, split


def channel_panels(mesh: core.Mesh, vertices: core.Point, n_rows: int) -> tuple[plt.Figure, list[plt.Axes]]:
    """Equal-aspect channel panels framing the channel."""
    readings = axes[:, None] & vertices.batch()                               # [2, vertices] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                  # [2] Scalar+, [2] Scalar+
    margin = 0.025 * (high[1] - low[1])                                       # [] Scalar+
    limits = stack((low - margin, high + margin), axis=-1).to_array()         # [2, 2]
    figure, panels = plt.subplots(n_rows, 1, figsize=(figure_width, figure_height * n_rows),
                                  dpi=dpi, squeeze=False)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, hspace=0.07)
    for ax in panels[:, 0]:
        ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
        ax.set_axis_off()
    return figure, list(panels[:, 0])


def walls(mesh: core.Mesh, vertices: core.Point, camera: Camera) -> np.ndarray:
    """The channel's walls, the mesh's boundary edges, as strokes."""
    return simplices.strokes(vertices.batch(), core.Mesh(mesh.n_vertices, mesh.edges[mesh.exterior(1)]),
                             rgb(wall_colour), wall_width, camera)


def draw_mesh(mesh: core.Mesh, vertices: core.Point) -> plt.Figure:
    """The channel and its obstacle, with the triangles resolving their boundaries."""
    figure, panels = channel_panels(mesh, vertices, 1)
    camera = flat_panel(panels[0], core.ga)
    simplices.overlay(panels[0], [simplices.strokes(vertices.batch(), mesh, rgb(mesh_colour), mesh_width, camera),
                                  walls(mesh, vertices, camera)])
    return figure


def draw_complex(mesh: core.Mesh, vertices: core.Point, circumcentres: core.Point,
                 dual_edges: core.Point) -> plt.Figure:
    """An interior vertex and its dual polygon, with a crossing primal and dual edge."""
    readings = axes[:, None] & vertices.batch()
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
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
    radius = 1.8 * reach.max()
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
    # The panel about the vertex, the radius either way.
    middle = axes & centre                                                     # [2] Scalar+
    limits = stack((middle - radius, middle + radius), axis=-1).to_array()     # [2, 2]
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.set_axis_off()
    camera = flat_panel(ax, core.ga)
    n_shown = shown.shape[0]
    # The shown primal and dual edges, each a chain of two points, its tail and its head.
    pairs = core.Mesh.chains([2] * n_shown)
    primal_tails, primal_heads = (shown * (mesh.incidence[1, 0] * vertices)).batch()   # [shown] Point-, [shown] Point-
    primal_ends = stack((primal_tails, primal_heads), axis=-1).reshape(-1)              # [shown * 2] Point-
    dual_ends_shown = stack(((shown * dual_starts).batch(), (shown * dual_ends).batch()), axis=-1).reshape(-1)   # [shown * 2] Point-
    star_centres = (mesh.reading(star, 2) * circumcentres).batch()             # [star] Point-
    spokes = star_centres - centre                                             # [star] Point-
    # Each spoke's turn from the first, read from the log of their product, orders the dual cell's corners around the
    # vertex.
    turns = (spokes[0] * spokes).normalized().log()                            # [star] Bivector+
    order = (mv.xy.dual() & turns).argsort()
    cell = star_centres[np.append(order, order[0])]                            # [star + 1] Point-
    nearest = spokes.norm().argmin()
    face = star[nearest]
    face_corners = (mesh.reading(np.array([face]), 2) * (mesh.incidence[2, 0] * vertices)).batch()[:, 0]   # [3] Point-
    centres = circumcentres.batch()                                            # [faces] Point-

    # The arrows carry the actual orientation of the two crossing geometric elements: the face's longest edge from
    # its midpoint, and its dual edge from the middle of its segment. Orientation alone, so both at one length, a
    # share of the primal edge's: a dual edge can be all but nothing where neighbouring circumcentres almost meet.
    edges = mesh.d[0] * vertices                                               # Point-[edges]
    sides = mesh.face_edges[face]
    longest = sides[(mesh.reading(sides, 1) * edges).batch().norm().argmax()]
    edge = mesh.reading(np.array([longest]), 1)                                # Scalar+[1 <- edges]
    primal, dual = edge * edges, edge * dual_edges                             # Point-[1], Point-[1]
    reach = arrow_reach * primal.norm()                                        # Scalar+[1]
    # The two arrows, the primal's and the dual's, each a stroke either way of its middle ending in a chevron.
    middles = concatenate(((edge * midpoints).batch(), (edge * (dual_starts + dual_ends) / 2).batch()))   # [2] Point-
    directions = concatenate((primal.batch(), dual.batch()))                   # [2] Point-
    along = reach.batch() * directions / directions.norm()                     # [2] Point-
    tips = middles + along                                                     # [2] Point-
    arrow_colours = np.stack((rgb(primal_colour), rgb(dual_colour)))
    arrows = simplices.strokes(stack((middles - along, tips), axis=-1).reshape(-1), core.Mesh.chains([2, 2]),
                               arrow_colours, arrow_width, camera, chevrons=np.ones(2), chevron=arrow_chevron)
    simplices.overlay(ax, [
        simplices.strokes(primal_ends, pairs, rgb(primal_colour), primal_width, camera, opacity=0.6),
        simplices.strokes(dual_ends_shown, pairs, rgb(dual_colour), dual_width, camera, opacity=0.75),
        contours.picture(star_centres, core.Mesh.triangulate(star_centres), np.zeros(len(star)), rgb(dual_colour),
                         camera, opacity=0.17),
        simplices.strokes(cell, core.Mesh.chains([len(cell)]), rgb(dual_colour), cell_width, camera, opacity=0.17),
        contours.picture(face_corners, core.Mesh(3, np.array([[0, 1, 2]])), np.zeros(3), rgb(primal_colour), camera,
                         opacity=0.13),
        simplices.strokes(face_corners[[0, 1, 2, 0]], core.Mesh.chains([4]), rgb(primal_colour), cell_width, camera,
                          opacity=0.13),
        simplices.markers(vertices.batch()[visible], rgb(primal_colour), vertex_radius, camera),
        simplices.markers(centres[(centres - centre).norm() < 1.5 * radius], rgb(dual_colour), vertex_radius, camera),
        simplices.markers(stack((centre, star_centres[nearest])), np.ones(3), marked_radius, camera),
        simplices.markers(stack((centre, star_centres[nearest])), np.stack((rgb(primal_colour), rgb(dual_colour))),
                          marked_radius - marked_ring, camera),
        arrows])
    return figure


def draw_fields(mesh: core.Mesh, vertices: core.Point, pressure: core.Scalar, vorticity: core.Bivector,
                streamfunction: core.Scalar) -> plt.Figure:
    """Pressure above and signed vorticity below, with the same flow paths on both."""
    figure, panels = channel_panels(mesh, vertices, 2)
    values = np.asarray(pressure.batch().to_array())
    rotation = np.asarray((mv.xy.dual() & vorticity).batch().to_array())
    potential = np.asarray(streamfunction.batch().to_array())
    # The flow paths, the streamfunction's levels at even steps between its values on the walls.
    paths = tuple(contours.Level(level, np.append(to_rgb(flow_colour), 0.45), 1.0)
                  for level in np.linspace(potential.min(), potential.max(), n_streamlines + 2)[1:-1])
    extent = np.quantile(np.abs(rotation), 0.97)
    # The pressure, constant on each face, shades each face's own corners alike; the streamfunction at them is the one
    # at their vertices.
    face_places, split = face_corners(mesh, vertices)
    corner_potential = np.asarray((mesh.incidence[2, 0] * streamfunction).batch().to_array()).reshape(-1)
    pressure_colours_at = np.tile(pressure_colours(Normalize(values.min(), values.max())(values))[:, :3], (3, 1))
    places = vertices.batch()                                                 # [vertices] Point-
    cameras = [flat_panel(ax, core.ga) for ax in panels]
    simplices.overlay(panels[0], [contours.picture(face_places, split, corner_potential, pressure_colours_at, cameras[0],
                                                   levels=paths),
                                  walls(mesh, vertices, cameras[0])])
    simplices.overlay(panels[1], [contours.picture(places, mesh, potential,
                                                   vorticity_colours(Normalize(-extent, extent)(rotation))[:, :3],
                                                   cameras[1], levels=paths),
                                  walls(mesh, vertices, cameras[1])])
    return figure


def draw_flow(mesh: core.Mesh, vertices: core.Point, pressure: core.Scalar, velocity: core.Point) -> plt.Figure:
    """The steady flow as streaks along its velocity, constant on each face, over the pressure."""
    figure, panels = channel_panels(mesh, vertices, 1)
    camera = flat_panel(panels[0], core.ga)
    values = pressure.batch().to_array()
    # The pressure and the velocity, constant on each face, at each face's own corners alike.
    face_places, split = face_corners(mesh, vertices)
    shades = np.tile(pressure_colours(Normalize(values.min(), values.max())(values))[:, :3], (3, 1))
    face_velocity = velocity.batch()                                          # [faces] Point-
    corner_velocity = concatenate((face_velocity, face_velocity, face_velocity))   # [corners * faces] Point-
    reference_speed = float(face_velocity.norm().max().to_array())
    simplices.overlay(panels[0], [contours.picture(face_places, split, np.zeros(len(shades)), shades, camera),
                                  flow.layer(face_places, split, corner_velocity, camera, reference_speed,
                                             rgb(flow_colour)),
                                  walls(mesh, vertices, camera)])
    return figure
