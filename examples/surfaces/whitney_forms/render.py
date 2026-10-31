"""Barycentric hat ramps, edge Whitney 1-forms, the forms on a tetrahedron, and anisotropic conduction fields."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import stack
from examples.util import charts
from examples.util.wgsl import contours, flow, image_figure, meshes, simplices, solids
from examples.util.wgsl.camera import Camera, flat_panel
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
# In pixels: the triangles' outlines, their sides drawn bold, the disks' boundaries, the meshes' edges and the fibre's
# chords; the arrows' width and their heads' length; the dots, small, highlighted and on the diamond, and their white
# rings. The fibre's chords dotted, a dot and a gap.
outline_width, side_width, edge_width, wall_width, mesh_width, chord_width = 1.6, 3.2, 3.0, 1.8, 1.0, 1.0
arrow_width, arrow_head = 2.8, 8.0
small_dot, highlight_dot, apex_dot, halo = 2.9, 4.8, 4.3, 1.25
chord_dots = (1.0, 2.0)
paper_colour = "#f6f4ee"
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
def outward_wall(mesh: core.SpaceMesh) -> tuple[np.ndarray, np.ndarray]:
    """The faces on the wall of a solid mesh, `[wall]`, and their corners `[wall, 3]`, wound outward. The wall is the
    boundary of the solid, the sum of its tetrahedra's outward faces: one on a wall face that runs outward as stored,
    minus one on a wall face that runs inward, nothing inside."""
    wall = mesh.exterior(2)
    solid = core.space_mesh_module.as_scalar(np.ones(mesh.n_elements[3])).field()
    sense = (~mesh.d[2] * solid).batch()[wall]
    return wall, np.where(sense[:, None] > 0, mesh.faces[wall], mesh.faces[wall][:, ::-1])


def frame_panel(ax: plt.Axes, points: core.Point, margin_ratio: float) -> Camera:
    """Square equal-aspect viewport around the points with no axes or ticks; its camera, looking down on x and y."""
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
    return flat_panel(ax, points.algebra)


def ink(colour: str) -> np.ndarray:
    """A colour's red, green and blue."""
    return np.array(to_rgb(colour))


def ringed(points: core.Point, colour: str, radius: float, camera: Camera) -> list[np.ndarray]:
    """Dots at points, `radius` pixels, ringed in white: their pictures."""
    return [simplices.markers(points, np.ones(3), radius + halo, camera),
            simplices.markers(points, ink(colour), radius, camera)]


def arrows(tails: core.Point, tips: core.Point, width: float, head: float, camera: Camera) -> np.ndarray:
    """Arrows in the accent from tails to tips `[arrows]`: their picture."""
    n_arrows = tails.shape[0]
    return simplices.strokes(stack((tails, tips), axis=-1).reshape(-1), simplices.Mesh.chains([2] * n_arrows),
                             accent_rgb, width, camera, heads=np.ones(n_arrows), head=head, cap=simplices.Cap.BUTT)


def draw_hat_0forms(corners: core.Point, gradients: core.Plane,
                    star_mesh: core.Mesh, star_vertices: core.Point, vertex_hat: core.Scalar) -> plt.Figure:
    """The three corner hat ramps with their constant gradients, and the assembled vertex pyramid by its levels."""
    triangle = meshes.Mesh(3, [[0, 1, 2]])
    # The hats' levels at even steps between nothing and one.
    levels = tuple(contours.Level(level, np.append(primal_rgb, 0.55), 1.0)
                   for level in np.linspace(0.0, 1.0, n_contours + 2)[1:-1])
    # Each gradient drawn through the triangle's centroid along the direction its hat rises, the longest a fixed
    # length.
    centroid = corners.mean(axis=0)                                           # [] Point-
    rises = gradients.dual()                                                  # [corners] Point-
    lengths = rises.norm().reshape(-1)
    reaches = rises * (0.24 / lengths.max())                  # [corners] Point-

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    tails, heads = triangle.corners[2, 1]
    for index in range(3):
        ax = axes[index]
        camera = frame_panel(ax, corners, 0.22)
        # The hat of this corner: one there, nothing at the others.
        values = np.eye(3)[index]
        simplices.overlay(ax, [
            contours.picture(corners, triangle, values, hat_colours(values)[:, :3], camera, levels=levels),
            simplices.strokes(corners, triangle, ink(wall_colour), outline_width, camera),
            simplices.strokes(corners[[tails[index], heads[index]]], simplices.Mesh.chains([2]),
                              ink(wall_colour), edge_width, camera),
            simplices.markers(corners, ink(wall_colour), small_dot, camera),
            *ringed(corners[index][None], accent_colour, highlight_dot, camera),
            arrows((centroid - reaches[index])[None], (centroid + reaches[index])[None], arrow_width, arrow_head,
                   camera)])

    ax = axes[3]
    hat_values = np.asarray(vertex_hat.batch().to_array())
    star_places = star_vertices.batch()                                       # [vertices] Point-
    camera = frame_panel(ax, star_vertices, 0.14)
    # The assembled hat's gradient, a field over the star, reads in its level lines.
    simplices.overlay(ax, [
        contours.picture(star_places, star_mesh, hat_values, hat_colours(hat_values)[:, :3], camera, levels=levels),
        simplices.strokes(star_places, star_mesh, ink(wall_colour), mesh_width, camera, opacity=0.65),
        *ringed(star_places[:1], accent_colour, highlight_dot, camera)])
    return figure


def draw_edge_1forms(corners: core.Point, corner_forms: core.Plane,
                     diamond: core.Mesh, diamond_vertices: core.Point, shared_corner_forms: core.Plane) -> plt.Figure:
    """Whitney edge forms as feathered streaks, with each face's corner values kept separate across the seam."""
    triangle = meshes.Mesh(3, [[0, 1, 2]])
    directions = corner_forms.dual()                                          # [corners] Point-
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    # Each side, from the corner after its own to the one after that, with an arrow about its middle along it.
    first, second = core.complements(core.triangle, 0).T
    side_starts, side_ends = corners[first], corners[second]                  # [corners] Point-, Point-
    side_middles, alongs = (side_starts + side_ends) / 2, side_ends - side_starts   # [corners] Point-, Point-
    along_tails, along_tips = side_middles - 0.18 * alongs, side_middles + 0.18 * alongs   # [corners] Point-, Point-

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        camera = frame_panel(ax, corners, 0.22)
        simplices.overlay(ax, [
            contours.picture(corners, triangle, np.zeros(3), ink(paper_colour), camera),
            simplices.strokes(corners, triangle, ink(wall_colour), charts.LINE_PX, camera),
            flow.layer(corners, triangle, directions[index], camera, reference_speed, primal_rgb),
            simplices.strokes(stack((side_starts[index], side_ends[index])), simplices.Mesh.chains([2]), accent_rgb,
                              side_width, camera, cap=simplices.Cap.BUTT),
            arrows(along_tails[index][None], along_tips[index][None], arrow_width, arrow_head, camera),
            *ringed(corners[index][None], primal_colour, highlight_dot, camera)])

    ax = axes[3]
    camera = frame_panel(ax, diamond_vertices, 0.16)
    places = diamond_vertices.batch()                                         # [vertices] Point-
    # Separate corners retain the two faces' different normal components along their shared edge.
    diamond_corners = (diamond.incidence[2, 0] * diamond_vertices).batch()
    positions = diamond_corners.reshape(-1)
    directions = shared_corner_forms.dual().batch().reshape(-1)
    n_faces = len(diamond.faces)
    split_mesh = meshes.Mesh(3 * n_faces, np.arange(3 * n_faces).reshape(3, n_faces).T)
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    # The shared edge, from its tail to its head, with an arrow about its middle along it.
    shared = diamond.reading(diamond.interior_edges[:1], 1)                   # Scalar+[shared <- edges]
    tail, head = (shared * (diamond.incidence[1, 0] * diamond_vertices)).batch()[:, 0]   # [] Point-, Point-
    middle, along = (tail + head) / 2, head - tail                            # [] Point-, Point-
    # The corners across the shared edge, in the face on its left and in the face on its right.
    apexes = (diamond.reading(diamond.edge_opposites[0]) * diamond_vertices).batch()   # [sides] Point-
    simplices.overlay(ax, [
        contours.picture(places, diamond, np.zeros(len(places)), ink(paper_colour), camera),
        simplices.strokes(places, diamond, ink(wall_colour), charts.LINE_PX, camera),
        flow.layer(positions, split_mesh, directions, camera, reference_speed, primal_rgb),
        simplices.strokes(stack((tail, head)), simplices.Mesh.chains([2]), accent_rgb, side_width, camera,
                          cap=simplices.Cap.BUTT),
        arrows((middle - 0.2 * along)[None], (middle + 0.2 * along)[None], arrow_width, arrow_head, camera),
        *ringed(apexes, primal_colour, apex_dot, camera)])
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
    edge, across = tet[mesh.corners[3, 1][:, 0]], tet[core.complements(mesh, 1)[0]]
    apex, face = tet[0], tet[mesh.corners[3, 2][:, 0]]
    wall, wall_faces = outward_wall(mesh)

    view = solids.View(solids.aim(solids.origin, tet_elevation, tet_azimuth), np.full(2, 0.86), n_tet_pixels)

    shell_colours = np.broadcast_to(cream, wall_faces.shape + (3,))
    shell = meshes.samples(places, wall_faces, shell_colours, meshes.flat(places, wall_faces), view,
                           face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    # The tetrahedron's wireframe: its mesh's own edges.
    wire = simplices.tubes(places, simplices.Mesh(len(places), mesh.edges), light_slate, 1.5, view, solids.gloss)

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
    # The face's rim: an edge from each of its corners to the next, round the face.
    face_rim = simplices.tubes(places, simplices.Mesh(len(places), np.stack((face, np.roll(face, -1)), axis=-1)),
                               accent_rgb, 2.8, view, solids.gloss)
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
    # Each ring a closed chain of edges, each edge in its ring's colour.
    rings = simplices.tubes(ring_places.reshape(-1), simplices.Mesh.chains([n_ring_samples] * len(ring_places)),
                            np.repeat(np.stack(ring_colours), n_ring_samples - 1, axis=0), 3.2, view, solids.gloss)
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
    potentials = np.asarray(potential.batch().to_array())
    potential_levels = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.45), 1.0)
                             for level in np.linspace(-0.85, 0.85, n_contours + 2)[1:-1])

    # Area-weighted vertex currents give a continuous drawing of the facewise solution.
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)
    areas = ((sides[0] & sides[1]) / 2).norm()                              # Scalar+[faces]
    corners = ~mesh.incidence[2, 0].mean(axis=0)
    vertex_current = (corners * (areas * current)) / (corners * areas)       # [cases] Point-[vertices]
    reference_speed = float(np.percentile(vertex_current.norm().batch().to_array(), 95))
    points = vertices.batch()                                                 # [vertices] Point-
    directions = vertex_current.batch()                                       # [cases, vertices] Point-
    # The disk's boundary, the mesh of its boundary's edges.
    rim = simplices.Mesh(len(points), mesh.edges[mesh.boundary_edges])

    # Faint chords of the unit disk along the fibre, across it at even offsets: each reaches the disk's edge the
    # half-chord of its offset either way.
    across = (core.mv.xy * (-np.pi / 4)).exp() >> fibre                       # [] Point-
    offsets = np.linspace(-0.85, 0.85, 9)
    half_chords = np.sqrt(1.0 - offsets**2)
    chord_ends = across * offsets[:, None] + fibre * (half_chords[:, None] * np.array([-1.0, 1.0]))   # [chords, ends] Point-

    figure, axes = charts.subplots(*flow_size, ncols=2)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98, wspace=0.05)

    for index, ax in enumerate(axes):
        camera = frame_panel(ax, vertices, 0.08)
        colours = potential_colours(Normalize(-1.0, 1.0)(potentials[index]))[:, :3]
        # Faint background chords indicate the tilted high-conductivity fibre orientation.
        fibres = [simplices.strokes(chord_ends.reshape(-1), simplices.Mesh.chains([2] * len(offsets)),
                                    ink(fibre_colour), chord_width, camera, dashes=chord_dots, opacity=0.55)
                  ] if index == 1 else []
        simplices.overlay(ax, [
            contours.picture(points, mesh, potentials[index], colours, camera, levels=potential_levels),
            simplices.strokes(points, mesh, ink(mesh_colour), mesh_width, camera, opacity=0.45),
            *fibres,
            flow.layer(points, mesh, directions[index], camera, reference_speed, ink(flow_colour)),
            simplices.strokes(points, rim, ink(wall_colour), wall_width, camera)])
    return figure
