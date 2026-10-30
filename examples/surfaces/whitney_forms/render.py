"""Barycentric hat ramps, edge Whitney 1-forms, the forms on a tetrahedron, and anisotropic conduction fields."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import where, stack
from examples.util import charts
from examples.util.wgsl import flow, image_figure, meshes, solids, strokes
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
# The pixel sizes of a row of four triangles and of the two disks side by side.
row_size, flow_size = (780, 195), (600, 290)
# The tetrahedron's three views, each square, and the gap between them, in pixels.
n_tet_pixels, n_tet_gap = 300, 10
tet_elevation, tet_azimuth = np.radians(23), np.radians(38)
# The rings about the edge across the first edge: their radii, and the samples around each.
ring_radii = np.array([0.20, 0.31, 0.42])
n_ring_samples = 65
background = np.ones(3)
cream = np.array([0.93, 0.94, 0.96])
light_slate = np.array([0.58, 0.65, 0.70])
warm_tint = np.array([0.96, 0.78, 0.64])
gold = np.array([0.95, 0.68, 0.16])
teal = np.array([0.08, 0.68, 0.56])
primal_rgb, accent_rgb = np.array(to_rgb(primal_colour)), np.array(to_rgb(accent_colour))
ring_colours = (gold, accent_rgb, teal)
# The unit volume of space, an intrinsic scalar: a direction times it is the plane square to it, the plane a turn about
# it turns in.
unit_volume = core.space_mv.antipseudoscalar([1.0])                          # [] Antipseudoscalar-


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Drawing coordinates of intrinsic positions and directions."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def outward_wall(mesh: core.SpaceMesh) -> tuple[np.ndarray, np.ndarray]:
    """The faces on the wall of a solid mesh, `[wall]`, and their corners `[wall, 3]`, wound outward. The wall is the
    boundary of the solid, the sum of its tetrahedra's outward faces: one on a wall face that runs outward as stored,
    minus one on a wall face that runs inward, nothing inside."""
    wall = mesh.exterior(2)
    solid = core.space_mesh_module.as_scalar(np.ones(mesh.n_elements[3])).field()
    sense = (~mesh.d[2] * solid).batch()[wall]
    return wall, np.where(sense[:, None] > 0, mesh.faces[wall], mesh.faces[wall][:, ::-1])


def level_lines(corners: core.Point, values: core.Scalar, levels: np.ndarray) -> core.Point:
    """The level lines of a field linear across triangles, `[ends, lines]`: in each triangle, of corners
    `[corners, faces]` and the field's values there `[corners, faces]`, the segment between the two sides where the
    field passes a level, for every level `[levels]` it passes."""
    following = corners[[1, 2, 0]]                                            # [corners, faces] Point-
    start, end = values[..., None], values[[1, 2, 0], :, None]                # [corners, faces, 1]
    # How far along each side, from its corner to the next, the field reaches each level, on a side where it changes.
    mv = values.context.multivector
    changing = (end - start).abs() > 0
    share = (levels - start) / where(changing, end - start, mv.scalar([np.inf]))    # [corners, faces, levels] Scalar+
    crossed = (share >= 0) & (share < 1) & changing                           # [corners, faces, levels]
    meeting = corners[..., None] + (following - corners)[..., None] * where(crossed, share, mv.scalar([0]))   # [corners, faces, levels] Point-
    # The two sides a level crosses, in the faces where it crosses any.
    order = np.argsort(~crossed, axis=0, kind="stable")[:2]                   # [ends, faces, levels]
    face, level = np.indices(order.shape[1:])
    return meeting[order, face, level][:, crossed.any(axis=0)]                # [ends, lines] Point-


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


def draw_hat_0forms(corners: core.Point, gradients: core.Plane,
                    star_mesh: core.Mesh, star_vertices: core.Point,
                    vertex_hat: core.Scalar, star_gradients: core.Plane) -> plt.Figure:
    """The three corner hat ramps with their constant gradients, and the assembled vertex pyramid."""
    corner_xy = coordinates(corners)
    single_tri = Triangulation(*corner_xy.T, triangles=np.array([[0, 1, 2]]))
    levels = np.linspace(0.0, 1.0, n_contours + 2)[1:-1]
    # Each gradient drawn through the triangle's centroid along the direction its hat rises, the longest a fixed
    # length.
    centroid = corners.mean(axis=0)                                           # [] Point-
    rises = gradients.dual()                                                  # [corners] Point-
    lengths = rises.norm().reshape(-1)
    reaches = rises * (0.24 / lengths.max())                  # [corners] Point-
    rise_tails, rise_tips = coordinates(centroid - reaches), coordinates(centroid + reaches)   # [corners, 2], [corners, 2]

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        frame_panel(ax, corners, 0.22)
        # The hat of this corner: one there, nothing at the others.
        values = np.eye(3)[index]
        ax.tripcolor(single_tri, values, shading="gouraud", cmap=hat_colours,
                     norm=Normalize(0.0, 1.0), zorder=1)
        lines = level_lines(corners[:, None], corners.context.multivector.scalar(values[:, None, None]), levels)        # [ends, lines] Point-
        ax.add_collection(LineCollection(coordinates(lines).transpose(1, 0, 2), colors=primal_colour,
                                         linewidths=0.75, alpha=0.55, zorder=2))
        ax.add_patch(Polygon(corner_xy, fill=False, edgecolor=wall_colour, linewidth=1.2, zorder=3))
        opposite = corner_xy[[(index + 1) % 3, (index + 2) % 3]]
        ax.plot(*opposite.T, color=wall_colour, linewidth=2.2, solid_capstyle="round", zorder=4)
        ax.scatter(*corner_xy.T, s=18, color=wall_colour, zorder=5)
        ax.scatter(*corner_xy[index], s=48, color=accent_colour, edgecolor="white", linewidth=0.9, zorder=6)
        ax.annotate("", xy=rise_tips[index], xytext=rise_tails[index],
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=accent_colour, linewidth=2.0),
                    zorder=7)

    ax = axes[3]
    star_xy = coordinates(star_vertices)
    star_tri = Triangulation(*star_xy.T, triangles=star_mesh.faces)
    hat_values = np.asarray(vertex_hat.batch().to_array())
    frame_panel(ax, star_vertices, 0.14)
    ax.tripcolor(star_tri, hat_values, shading="gouraud", cmap=hat_colours,
                 norm=Normalize(0.0, 1.0), zorder=1)
    # The hat's level lines across the faces, from its values at their corners.
    corner_hats = (star_mesh.incidence[2, 0] * vertex_hat).batch()   # [corners, faces]
    star_lines = level_lines((star_mesh.incidence[2, 0] * star_vertices).batch(), corner_hats, levels)   # [ends, lines] Point-
    ax.add_collection(LineCollection(coordinates(star_lines).transpose(1, 0, 2), colors=primal_colour,
                                     linewidths=0.7, alpha=0.55, zorder=2))
    ax.triplot(star_tri, color=wall_colour, linewidth=0.65, alpha=0.65, zorder=3)
    ax.scatter(*star_xy[0], s=48, color=accent_colour, edgecolor="white", linewidth=0.9, zorder=6)
    # The gradient on each face where the hat rises, through the face's centroid, the longest a fixed length.
    star_rises = star_gradients.dual().batch()                                # [faces] Point-
    rising = star_rises.norm() > 1e-8                                         # [faces]
    star_lengths = star_rises[rising].norm().reshape(-1)
    star_reaches = star_rises[rising] * (0.16 / star_lengths.max())   # [rising] Point-
    star_centres = (star_mesh.incidence[2, 0].mean(axis=0) * star_vertices).batch()[rising]   # [rising] Point-
    for tail, tip in zip(coordinates(star_centres - star_reaches), coordinates(star_centres + star_reaches)):
        ax.annotate("", xy=tip, xytext=tail,
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=10, color=accent_colour, linewidth=1.6),
                    zorder=7)
    return figure


def draw_edge_1forms(corners: core.Point, corner_forms: core.Plane,
                     diamond: core.Mesh, diamond_vertices: core.Point, shared_corner_forms: core.Plane) -> plt.Figure:
    """Whitney edge forms as feathered streaks, with each face's corner values kept separate across the seam."""
    corner_xy = coordinates(corners)
    triangle = meshes.Mesh(3, [[0, 1, 2]])
    positions = solids.origin + solids.context.include_from_algebra(corners)
    directions = solids.context.include_from_algebra(corner_forms.dual())
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    # Each side, from the corner after its own to the one after that, with an arrow about its middle along it.
    side_starts, side_ends = corners[[1, 2, 0]], corners[[2, 0, 1]]           # [corners] Point-, Point-
    side_middles, alongs = (side_starts + side_ends) / 2, side_ends - side_starts   # [corners] Point-, Point-
    start_xy, end_xy = coordinates(side_starts), coordinates(side_ends)       # [corners, 2], [corners, 2]
    along_tails, along_tips = coordinates(side_middles - 0.18 * alongs), coordinates(side_middles + 0.18 * alongs)

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        frame_panel(ax, corners, 0.22)
        ax.add_patch(Polygon(corner_xy, facecolor="#f6f4ee", edgecolor=wall_colour, linewidth=1.1, zorder=1))
        flow.overlay(ax, positions, triangle, directions[index], reference_speed, primal_rgb)
        ax.plot(*np.stack([start_xy[index], end_xy[index]]).T, color=accent_colour, linewidth=2.4, zorder=4)
        ax.annotate("", xy=along_tips[index], xytext=along_tails[index],
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=accent_colour, linewidth=2.0),
                    zorder=5)
        ax.scatter(*corner_xy[index], s=42, color=primal_colour, edgecolor="white", linewidth=0.9, zorder=6)

    ax = axes[3]
    places = coordinates(diamond_vertices)
    frame_panel(ax, diamond_vertices, 0.16)
    for tri_xy in coordinates(diamond.incidence[2, 0] * diamond_vertices).transpose(1, 0, 2):
        ax.add_patch(Polygon(tri_xy, facecolor="#f6f4ee", edgecolor=wall_colour, linewidth=1.1, zorder=1))
    # Separate corners retain the two faces' different normal components along their shared edge.
    diamond_corners = (diamond.incidence[2, 0] * diamond_vertices).batch()
    positions = solids.origin + solids.context.include_from_algebra(diamond_corners.reshape(-1))
    directions = solids.context.include_from_algebra(shared_corner_forms.dual().batch().reshape(-1))
    n_faces = len(diamond.faces)
    split_mesh = meshes.Mesh(3 * n_faces, np.arange(3 * n_faces).reshape(3, n_faces).T)
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    flow.overlay(ax, positions, split_mesh, directions, reference_speed, primal_rgb)
    # The shared edge, from its tail to its head, with an arrow about its middle along it.
    shared = diamond.reading(diamond.interior_edges[:1], 1)                   # Scalar+[shared <- edges]
    tail, head = (shared * (diamond.incidence[1, 0] * diamond_vertices)).batch()[:, 0]   # [] Point-, Point-
    middle, along = (tail + head) / 2, head - tail                            # [] Point-, Point-
    ax.plot(*coordinates(stack([tail, head])).T, color=accent_colour, linewidth=2.4, zorder=4)
    ax.annotate("", xy=coordinates(middle + 0.2 * along), xytext=coordinates(middle - 0.2 * along),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=accent_colour, linewidth=2.0),
                zorder=5)
    # The corners across the shared edge, in the face on its left and in the face on its right.
    for apex in coordinates(diamond.reading(diamond.edge_opposites[0]) * diamond_vertices):
        ax.scatter(*apex, s=38, color=primal_colour, edgecolor="white", linewidth=0.9, zorder=6)
    return figure


def draw_tet_whitney(mesh: core.SpaceMesh, vertices: core.SpacePoint, points: core.SpacePoint,
                     edge_forms: core.SpacePlane, face_forms: core.SpaceBivector, volume: core.Volume) -> plt.Figure:
    """Three views of a mesh of one tetrahedron: the Whitney 1-form of its first edge, the 2-form of its first face, and
    the linkage of that edge with the edge across it."""
    # The tetrahedron about its centroid, and the samples spread away from it so that their glyphs' directions stay
    # legible.
    centroid = vertices.sites.mean()                                          # [] Point-
    corners = (vertices - centroid).batch()                                   # [vertices] Point-
    samples = (points - centroid).batch().reshape(-1) * 1.22                  # [samples] Point-
    places = solids.origin + solids.context.include_from_algebra(corners)                      # [vertices] Point-
    edge_field = edge_forms[0].dual().batch().reshape(-1)                     # [samples] Point-
    # A face form times the volume's sense is the direction its flux crosses.
    face_field = (face_forms[0] * (volume / volume.norm())).batch().reshape(-1)   # [samples] Point-

    # The tetrahedron's first edge, the edge across it, the corner the first face form radiates from, and the face
    # facing that corner, as vertices.
    tet = mesh.tets[0]
    edge, across = tet[mesh.corners(3, 1)[0]], tet[core.complements(mesh, 1)[0]]
    apex, face = tet[0], tet[mesh.corners(3, 2)[0]]
    wall, wall_faces = outward_wall(mesh)

    view = solids.View(solids.aim(solids.origin, tet_elevation, tet_azimuth), np.full(2, 0.86), n_tet_pixels)

    shell_colours = np.broadcast_to(cream, wall_faces.shape + (3,))
    shell = meshes.samples(places, wall_faces, shell_colours, meshes.flat(places, wall_faces), view,
                           face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    tails, heads = mesh.incidence[1, 0] * (vertices - centroid)               # Point-[edges], Point-[edges]
    wire = strokes.segments(solids.origin + solids.context.include_from_algebra(tails.batch()),
                            solids.origin + solids.context.include_from_algebra(heads.batch()), light_slate, 1.5, view)

    # The Whitney 1-form of the first edge, circulating around the edge across it, the longest glyph a fixed length.
    edge_lengths = edge_field.norm().reshape(-1)
    edge_reaches = edge_field * (0.30 / edge_lengths.max())   # [samples] Point-
    e_starts = solids.origin + solids.context.include_from_algebra(samples - 0.45 * edge_reaches)   # [samples] Point-
    e_ends = solids.origin + solids.context.include_from_algebra(samples + 0.55 * edge_reaches)     # [samples] Point-
    e_shafts, e_heads = solids.arrows(e_starts, e_ends, 0.017, 0.065, 0.038)
    active_rod, active_balls = solids.edges(places[edge[:1]], places[edge[1:]], 0.024)
    opp_rod, opp_balls = solids.edges(places[across[:1]], places[across[1:]], 0.020)
    # Each of the two edges, its middle and the way along it, with an arrow about its middle.
    edge_middle, edge_along = corners[edge].mean(axis=0), corners[edge[1]] - corners[edge[0]]   # [] Point-, Point-
    edge_arrow_s, edge_arrow_h = solids.arrows(
        (solids.origin + solids.context.include_from_algebra(edge_middle - 0.18 * edge_along))[None],
        (solids.origin + solids.context.include_from_algebra(edge_middle + 0.22 * edge_along))[None],
        0.025, 0.090, 0.054,
    )
    panel_edge = solids.draw(
        [(e_shafts, accent_rgb), (e_heads, accent_rgb), (active_rod, accent_rgb), (active_balls, accent_rgb),
         (edge_arrow_s, accent_rgb), (edge_arrow_h, accent_rgb), (opp_rod, primal_rgb), (opp_balls, primal_rgb)],
        view, background, inks=[], translucent=[], meshes=[(shell, 0.24), (wire, 1.0)],
    )

    # The Whitney 2-form of the first face, radiating from the corner it faces through that face.
    face_lengths = face_field.norm().reshape(-1)
    face_reaches = face_field * (0.30 / face_lengths.max())   # [samples] Point-
    f_starts = solids.origin + solids.context.include_from_algebra(samples - 0.45 * face_reaches)   # [samples] Point-
    f_ends = solids.origin + solids.context.include_from_algebra(samples + 0.55 * face_reaches)     # [samples] Point-
    f_shafts, f_heads = solids.arrows(f_starts, f_ends, 0.017, 0.065, 0.038)
    face_colours = np.broadcast_to(cream, wall_faces.shape + (3,)).copy()
    face_colours[wall == mesh.boundary_indices[2][0, 0]] = warm_tint
    face_mesh = meshes.samples(places, wall_faces, face_colours, meshes.flat(places, wall_faces), view,
                               face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    apex_ball = solids.ball(places[[apex]], 0.046)[:, None]
    face_rim = strokes.segments(places[face], places[np.roll(face, -1)], accent_rgb, 2.8, view)
    panel_face = solids.draw(
        [(f_shafts, primal_rgb), (f_heads, primal_rgb), (apex_ball, accent_rgb)],
        view, background, inks=[], translucent=[], meshes=[(face_mesh, 0.30), (wire, 1.0), (face_rim, 1.0)],
    )

    # Rings around the edge across the first edge, which the first edge threads.
    across_middle, across_along = corners[across].mean(axis=0), corners[across[1]] - corners[across[0]]   # [] Point-, Point-
    across_axis = across_along.normalized()                                   # [] Point-
    opp_arrow_s, opp_arrow_h = solids.arrows(
        (solids.origin + solids.context.include_from_algebra(across_middle - 0.18 * across_along))[None],
        (solids.origin + solids.context.include_from_algebra(across_middle + 0.22 * across_along))[None],
        0.025, 0.090, 0.054,
    )
    # Each ring starts from the edge across towards the first edge's middle, square to the edge across, and turns
    # about it in the plane square to it.
    towards = edge_middle - across_middle                                     # [] Point-
    radial = (towards - across_axis * towards.dot(across_axis)).normalized()  # [] Point-
    ring_turns = ((across_axis * unit_volume) * (-np.linspace(0.0, 2 * np.pi, n_ring_samples) / 2)).exp()   # [samples] Rotor+
    ring_points = across_middle * 0.62 + ring_radii[:, None] * (ring_turns >> radial)   # [rings, samples] Point-
    ring_places = solids.origin + solids.context.include_from_algebra(ring_points)            # [rings, samples] Point-
    rings = strokes.polylines(list(zip(ring_places, ring_colours)), 3.2, view)
    panel_link = solids.draw(
        [(active_rod, accent_rgb), (active_balls, accent_rgb), (edge_arrow_s, accent_rgb), (edge_arrow_h, accent_rgb),
         (opp_rod, primal_rgb), (opp_balls, primal_rgb), (opp_arrow_s, primal_rgb), (opp_arrow_h, primal_rgb)],
        view, background, inks=[], translucent=[], meshes=[(shell, 0.22), (wire, 1.0), (rings, 1.0)],
    )

    gap = np.full((n_tet_pixels, n_tet_gap, 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate((panel_edge, gap, panel_face, gap, panel_link), axis=1))


def draw_anisotropic_flow(mesh: core.Mesh, vertices: core.Point,
                          potential: core.Scalar,
                          current: core.Point, fibre: core.Point) -> plt.Figure:
    """Isotropic and tilted-anisotropic conduction share one potential palette and current-length scale."""
    places = coordinates(vertices)
    potentials = np.asarray(potential.batch().to_array())
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    walls = coordinates(mesh.reading(mesh.boundary_edges, 1) * (mesh.incidence[1, 0] * vertices)).transpose(1, 0, 2)
    potential_levels = np.linspace(-0.85, 0.85, n_contours + 2)[1:-1]

    # Area-weighted vertex currents give a continuous drawing of the facewise solution.
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)
    areas = ((sides[0] & sides[1]) / 2).norm()                              # Scalar+[faces]
    corners = ~mesh.incidence[2, 0].mean(axis=0)
    vertex_current = (corners * (areas * current)) / (corners * areas)       # [cases] Point-[vertices]
    reference_speed = float(np.percentile(vertex_current.norm().batch().to_array(), 95))
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())   # [vertices] Point-
    directions = solids.context.include_from_algebra(vertex_current.batch())        # [cases, vertices] Point-

    # Faint chords of the unit disk along the fibre, across it at even offsets: each reaches the disk's edge the
    # half-chord of its offset either way.
    across = (core.mv.xy * (-np.pi / 4)).exp() >> fibre                       # [] Point-
    offsets = np.linspace(-0.85, 0.85, 9)
    half_chords = np.sqrt(1.0 - offsets**2)
    chord_ends = across * offsets + fibre * (np.array([-1.0, 1.0])[:, None] * half_chords)   # [ends, chords] Point-
    chords = coordinates(chord_ends).transpose(1, 0, 2)                       # [chords, ends, 2]

    figure, axes = charts.subplots(*flow_size, ncols=2)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98, wspace=0.05)

    for index, ax in enumerate(axes):
        frame_panel(ax, vertices, 0.08)
        ax.tripcolor(triangulation, potentials[index], shading="gouraud", cmap=potential_colours,
                     norm=Normalize(-1.0, 1.0), zorder=1)
        ax.triplot(triangulation, color=mesh_colour, linewidth=0.35, alpha=0.45, zorder=2)
        ax.tricontour(triangulation, potentials[index], levels=potential_levels,
                      colors=wall_colour, linestyles="dashed", linewidths=0.75, alpha=0.65, zorder=3)
        flow.overlay(ax, points, mesh, directions[index], reference_speed, np.array(to_rgb(flow_colour)))
        ax.add_collection(LineCollection(walls, colors=wall_colour, linewidths=1.3, zorder=6))
        if index == 1:
            # Faint background chords indicate the tilted high-conductivity fibre orientation.
            ax.add_collection(LineCollection(chords, colors=fibre_colour, linewidths=0.65,
                                             linestyles="dotted", alpha=0.55, zorder=2))
    return figure
