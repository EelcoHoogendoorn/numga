"""Barycentric hat ramps, edge Whitney 1-forms, the forms on a tetrahedron, and anisotropic conduction fields."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import stack
from examples.util import charts
from examples.util.wgsl import contours, flow, image_figure, meshes, simplices, solids
from examples.util.wgsl.camera import Camera, offset_panel
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
# chords; the arrows' width; the dots, small, highlighted and on the diamond, and their white rings. The fibre's chords
# dotted, a dot and a gap. The length of the strokes of the chevron at an arrow's tip, in widths of the arrow.
outline_width, side_width, edge_width, wall_width, mesh_width, chord_width = 1.6, 3.2, 3.0, 1.8, 1.0, 1.0
arrow_width = 2.8
small_dot, highlight_dot, apex_dot, halo = 2.9, 4.8, 4.3, 1.25
chord_dots = (1.0, 2.0)
chevron_length = 4.0
paper_colour = "#f6f4ee"
# The pixel sizes of a row of four triangles and of the two disks side by side.
row_size, flow_size = (780, 195), (600, 290)
# The tetrahedron's three views, each square, and the gap between them, in pixels.
n_tet_pixels, n_tet_gap = 300, 10
# The direction the edge form and the linkage are seen from, and the azimuth the face form is seen from, out past
# its face.
tet_elevation, tet_azimuth, face_azimuth = np.radians(23), np.radians(38), np.radians(80)
# How much each of the edge form's level surfaces hides.
page_opacity = 0.2
# In pixels: the level surfaces' rims and the field lines. In the tetrahedron's units: the radius of the balls where the
# field lines cross the face. The length of the strokes of the chevron at the head of the first edge and of the edge
# across it, in widths of the edge.
page_width, flux_width = 1.6, 2.0
flux_tip = 0.026
edge_chevron = 3.0
# The rings about the edge across the first edge: their radii, and the samples around each.
ring_radii = np.array([0.20, 0.31, 0.42])
n_ring_samples = 64
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
def outward_wall(topology: core.Topology) -> tuple[np.ndarray, np.ndarray]:
    """The faces on the wall of a solid mesh, `[wall]`, and their corners `[wall, 3]`, wound outward. The wall is the
    boundary of the solid, the sum of its tetrahedra's outward faces: one on a wall face that runs outward as stored,
    minus one on a wall face that runs inward, nothing inside."""
    wall = topology.exterior[2]
    solid = core.space_mv.scalar(np.ones(topology.n_elements[3])[..., None]).field()
    sense = (topology.reading(wall, 2) * (~topology.d[2] * solid)).batch()
    return wall, np.where(sense[:, None] > 0, topology.elements[2][wall], topology.elements[2][wall][:, ::-1])


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
    return offset_panel(ax, mv.x, mv.y)


def ink(colour: str) -> np.ndarray:
    """A colour's red, green and blue."""
    return np.array(to_rgb(colour))


def ringed(points: core.Point, colour: str, radius: float, camera: Camera) -> list[np.ndarray]:
    """Dots at points, `radius` pixels, ringed in white: their pictures."""
    return [simplices.markers(points, np.ones(3), radius + halo, camera),
            simplices.markers(points, ink(colour), radius, camera)]


def arrows(tails: core.Point, tips: core.Point, width: float, camera: Camera) -> np.ndarray:
    """Arrows in the accent from tails to tips `[arrows]`, each a stroke `width` pixels wide ending in a chevron: their
    picture."""
    n_arrows = tails.shape[0]
    return simplices.strokes(stack((tails, tips), axis=-1).reshape(-1), simplices.Topology.chains([2] * n_arrows),
                             accent_rgb, width, camera, chevrons=np.ones(n_arrows), chevron=chevron_length * width,
                             cap=simplices.Cap.BUTT)


def draw_hat_0forms(corners: core.Point, gradients: core.Plane,
                    topology: core.Topology, star_vertices: core.Point, vertex_hat: core.Scalar) -> plt.Figure:
    """The three corner hat ramps with their constant gradients, and the assembled vertex pyramid by its levels."""
    triangle = meshes.Topology(3, [[0, 1, 2]])
    # The hats' levels at even steps between nothing and one.
    levels = tuple(contours.Level(level, np.append(primal_rgb, 0.55), 1.0)
                   for level in np.linspace(0.0, 1.0, n_contours + 2)[1:-1])
    # Each gradient drawn through the triangle's centroid along the direction its hat rises, the longest a fixed
    # length.
    centroid = corners.mean(axis=0)                                           # [] Point-
    rises = gradients.dual()                                                  # [corners] Point-
    lengths = rises.norm().reshape(-1)
    reaches = rises * (0.24 / lengths.max())                                  # [corners] Point-

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
            simplices.strokes(corners[[tails[index], heads[index]]], simplices.Topology.chains([2]),
                              ink(wall_colour), edge_width, camera),
            simplices.markers(corners, ink(wall_colour), small_dot, camera),
            *ringed(corners[index][None], accent_colour, highlight_dot, camera),
            arrows((centroid - reaches[index])[None], (centroid + reaches[index])[None], arrow_width, camera)])

    ax = axes[3]
    hat_values = np.asarray(vertex_hat.batch().to_array())
    star_places = star_vertices.batch()                                       # [vertices] Point-
    camera = frame_panel(ax, star_vertices, 0.14)
    # The assembled hat's gradient, a field over the star, reads in its level lines.
    simplices.overlay(ax, [
        contours.picture(star_places, topology, hat_values, hat_colours(hat_values)[:, :3], camera, levels=levels),
        simplices.strokes(star_places, topology, ink(wall_colour), mesh_width, camera, opacity=0.65),
        *ringed(star_places[:1], accent_colour, highlight_dot, camera)])
    return figure


def draw_edge_1forms(corners: core.Point, corner_forms: core.Plane,
                     diamond: core.Topology, diamond_vertices: core.Point, shared_corner_forms: core.Plane) -> plt.Figure:
    """Whitney edge forms as feathered streaks, with each face's corner values kept separate across the seam."""
    triangle = meshes.Topology(3, [[0, 1, 2]])
    directions = corner_forms.dual()                                          # [corners] Point-
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    # Each side, from the corner after its own to the one after that, an arrow ending in a chevron.
    first, second = core.complements(core.triangle, 0).T
    side_starts, side_ends = corners[first], corners[second]                  # [corners] Point-, Point-

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        camera = frame_panel(ax, corners, 0.22)
        simplices.overlay(ax, [
            contours.picture(corners, triangle, np.zeros(3), ink(paper_colour), camera),
            simplices.strokes(corners, triangle, ink(wall_colour), charts.LINE_PX, camera),
            flow.layer(corners, triangle, directions[index], camera, reference_speed, primal_rgb),
            arrows(side_starts[index][None], side_ends[index][None], side_width, camera),
            *ringed(corners[index][None], primal_colour, highlight_dot, camera)])

    ax = axes[3]
    camera = frame_panel(ax, diamond_vertices, 0.16)
    places = diamond_vertices.batch()                                         # [vertices] Point-
    # Separate corners retain the two faces' different normal components along their shared edge.
    diamond_corners = (diamond.incidence[2, 0] * diamond_vertices).batch()
    positions = diamond_corners.reshape(-1)
    directions = shared_corner_forms.dual().batch().reshape(-1)
    n_faces = diamond.n_elements[2]
    topology = meshes.Topology(3 * n_faces, np.arange(3 * n_faces).reshape(3, n_faces).T)
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    # The shared edge, from its tail to its head, an arrow ending in a chevron.
    shared = diamond.reading(diamond.interior[1][:1], 1)                      # Scalar+[shared <- edges]
    tail, head = (shared * (diamond.incidence[1, 0] * diamond_vertices)).batch()[:, 0]   # [] Point-, Point-
    # The corners across the shared edge, in the face on its left and in the face on its right.
    apexes = (diamond.reading(diamond.facet_opposites[0], 0) * diamond_vertices).batch()  # [sides] Point-
    simplices.overlay(ax, [
        contours.picture(places, diamond, np.zeros(len(places)), ink(paper_colour), camera),
        simplices.strokes(places, diamond, ink(wall_colour), charts.LINE_PX, camera),
        flow.layer(positions, topology, directions, camera, reference_speed, primal_rgb),
        arrows(tail[None], head[None], side_width, camera),
        *ringed(apexes, primal_colour, apex_dot, camera)])
    return figure


def draw_tet_whitney(topology: core.Topology, vertices: core.SpacePoint, pages: core.SpacePoint,
                     field_lines: core.SpacePoint) -> plt.Figure:
    """Three views of a mesh of one tetrahedron: the Whitney 1-form of its first edge by its level surfaces, triangles
    `pages` `[tet_edges, pages, 3] Point-[tets]`; the 2-form of its first face by its field lines `[corners, lines, 2]
    Point-[tets]`, from the corner the face faces to the face; and the linkage of that edge with the edge across it."""
    # The tetrahedron about its centroid.
    centroid = vertices.sites.mean()                                          # [] Point-
    corners = (vertices - centroid).batch()                                   # [vertices] Point-
    places = solids.origin + solids.context.include_from_algebra(corners)     # [vertices] Point-

    # The tetrahedron's corners in its own order, about the centroid and in the drawing; and its first edge, the edge
    # across it, the corner the first face form radiates from, and the face facing that corner, as its corners.
    tet_corners = (topology.incidence[3, 0] * (vertices - centroid)).batch()[:, 0]   # [4] Point-
    tet_places = solids.origin + solids.context.include_from_algebra(tet_corners)    # [4] Point-
    edge, across = topology.corners[3, 1][:, 0], core.complements(topology, 1)[0]
    apex, face = 0, core.complements(topology, 0)[0]
    wall, wall_faces = outward_wall(topology)

    view = solids.View(solids.aim(solids.origin, tet_elevation, tet_azimuth), np.full(2, 0.86), n_tet_pixels)
    face_view = view._replace(camera=solids.aim(solids.origin, tet_elevation, face_azimuth))

    shell_colours = np.broadcast_to(cream, wall_faces.shape + (3,))
    shell = meshes.samples(places, wall_faces, shell_colours, meshes.flat(places, wall_faces), view,
                           face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    # The tetrahedron's wireframe: its mesh's own edges.
    wire = simplices.tubes(places, topology, light_slate, 1.5, view, solids.gloss)
    # Each of the two edges an arrow, a tube ending in a chevron, as wide as a rod of its radius in the tetrahedron's
    # units; the edge across, where its sense is not shown, also a plain rod.
    active_width, opp_width = 2 * 0.024 / view.pixel, 2 * 0.020 / view.pixel
    active_arrow = simplices.tubes(tet_places[edge], simplices.Topology.chains([2]), accent_rgb, active_width, view,
                                   solids.gloss, chevrons=np.ones(1), chevron=edge_chevron * active_width)
    opp_arrow = simplices.tubes(tet_places[across], simplices.Topology.chains([2]), primal_rgb, opp_width, view, solids.gloss,
                                chevrons=np.ones(1), chevron=edge_chevron * opp_width)
    opp_rod, opp_balls = solids.edges(tet_places[across[:1]], tet_places[across[1:]], 0.020)

    # The first edge's pages about the centroid, and each page's corners among them, in each tetrahedron.
    first_pages = (pages[0] - centroid).batch()                              # [pages, 3, tets] Point-
    page_places = solids.origin + solids.context.include_from_algebra(first_pages.reshape(-1))   # [page_corners] Point-
    page_triangles = np.arange(len(page_places)).reshape(first_pages.shape).swapaxes(1, 2)   # [pages, tets, 3]
    # Each page a mesh of its own, so that pages seen through each other each add their tint.
    page_fills = [(meshes.samples(page_places, triangles, np.broadcast_to(warm_tint, triangles.shape + (3,)),
                                  meshes.flat(page_places, triangles), view, face_values=np.zeros(len(triangles)),
                                  colour=meshes.plain, gloss=solids.gloss), page_opacity) for triangles in page_triangles]
    # Each page's two sides off the edge across, the form's level lines on the two faces at the first edge.
    page_sides = np.stack((page_triangles[..., :2], page_triangles[..., 1:]), axis=-2).reshape(-1, 2)
    page_rims = simplices.tubes(page_places, simplices.Topology(len(page_places), page_sides), accent_rgb, page_width,
                                view, solids.gloss)
    panel_edge = solids.draw(
        [(opp_rod, primal_rgb), (opp_balls, primal_rgb)],
        view, background, inks=[], translucent=[],
        meshes=[(shell, 0.24), (wire, 1.0), *page_fills, (page_rims, 1.0), (active_arrow, 1.0)],
    )

    # The first face's field lines about the centroid, each line's ends among them, in each tetrahedron; the flux runs
    # out through the face, where a ball marks each line.
    first_lines = (field_lines[0] - centroid).batch()                        # [lines, 2, tets] Point-
    line_places = solids.origin + solids.context.include_from_algebra(first_lines.reshape(-1))   # [line_ends] Point-
    line_ends = np.arange(len(line_places)).reshape(first_lines.shape).swapaxes(1, 2).reshape(-1, 2)   # [lines * tets, 2]
    flux_lines = simplices.tubes(line_places, simplices.Topology(len(line_places), line_ends), primal_rgb, flux_width,
                                 face_view, solids.gloss)
    flux_tips = solids.ball(line_places[line_ends[:, 1]], flux_tip)[:, None]
    face_colours = np.broadcast_to(cream, wall_faces.shape + (3,)).copy()
    face_colours[wall == topology.boundary_indices[2][0, 0]] = warm_tint
    face_mesh = meshes.samples(places, wall_faces, face_colours, meshes.flat(places, wall_faces), face_view,
                               face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    apex_ball = solids.ball(tet_places[[apex]], 0.046)[:, None]
    # The face's rim: an edge from each of its corners to the next, round the face.
    face_rim = simplices.tubes(tet_places[face], core.triangle.border, accent_rgb, 2.8, face_view, solids.gloss)
    face_wire = simplices.tubes(places, topology, light_slate, 1.5, face_view, solids.gloss)
    panel_face = solids.draw(
        [(flux_tips, primal_rgb), (apex_ball, accent_rgb)],
        face_view, background, inks=[], translucent=[],
        meshes=[(face_mesh, 0.30), (face_wire, 1.0), (face_rim, 1.0), (flux_lines, 1.0)],
    )

    # Rings around the edge across the first edge, which the first edge threads.
    edge_middle, across_middle = tet_corners[edge].mean(axis=0), tet_corners[across].mean(axis=0)  # [] Point-, Point-
    across_axis = (tet_corners[across[1]] - tet_corners[across[0]]).normalized()  # [] Point-
    # Each ring starts from the edge across towards the first edge's middle, square to the edge across, and turns
    # about it in the plane square to it.
    towards = edge_middle - across_middle                                     # [] Point-
    radial = (towards - across_axis * towards.dot(across_axis)).normalized()  # [] Point-
    ring_turns = ((across_axis * unit_volume) * (-np.linspace(0.0, 2 * np.pi, n_ring_samples, endpoint=False) / 2)).exp()   # [samples] Rotor+
    ring_points = across_middle * 0.62 + ring_radii[:, None] * (ring_turns >> radial)   # [rings, samples] Point-
    ring_places = solids.origin + solids.context.include_from_algebra(ring_points)      # [rings, samples] Point-
    # Each ring a closed loop of edges, each edge in its ring's colour.
    rings = simplices.tubes(ring_places.reshape(-1), simplices.Topology.loops([n_ring_samples] * len(ring_places)),
                            np.repeat(np.stack(ring_colours), n_ring_samples, axis=0), 3.2, view, solids.gloss)
    panel_link = solids.draw(
        [], view, background, inks=[], translucent=[],
        meshes=[(shell, 0.22), (wire, 1.0), (rings, 1.0), (active_arrow, 1.0), (opp_arrow, 1.0)],
    )

    gap = np.full((n_tet_pixels, n_tet_gap, 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate((panel_edge, gap, panel_face, gap, panel_link), axis=1))


def draw_anisotropic_flow(topology: core.Topology, vertices: core.Point,
                          potential: core.Scalar,
                          current: core.Point, fibre: core.Point) -> plt.Figure:
    """Isotropic and tilted-anisotropic conduction share one potential palette and current-length scale."""
    potentials = np.asarray(potential.batch().to_array())
    potential_levels = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.45), 1.0)
                             for level in np.linspace(-0.85, 0.85, n_contours + 2)[1:-1])

    # Area-weighted vertex currents give a continuous drawing of the facewise solution.
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)
    areas = ((sides[0] & sides[1]) / 2).norm()                              # Scalar+[faces]
    corners = ~topology.incidence[2, 0].mean(axis=0)
    vertex_current = (corners * (areas * current)) / (corners * areas)       # [cases] Point-[vertices]
    reference_speed = float(np.percentile(vertex_current.norm().batch().to_array(), 95))
    points = vertices.batch()                                                 # [vertices] Point-
    directions = vertex_current.batch()                                       # [cases, vertices] Point-
    # The disk's boundary, the mesh of its boundary's edges.
    rim = topology.border

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
        fibres = [simplices.strokes(chord_ends.reshape(-1), simplices.Topology.chains([2] * len(offsets)),
                                    ink(fibre_colour), chord_width, camera, dashes=chord_dots, opacity=0.55)
                  ] if index == 1 else []
        simplices.overlay(ax, [
            contours.picture(points, topology, potentials[index], colours, camera, levels=potential_levels),
            simplices.strokes(points, topology, ink(mesh_colour), mesh_width, camera, opacity=0.45),
            *fibres,
            flow.layer(points, topology, directions[index], camera, reference_speed, ink(flow_colour)),
            simplices.strokes(points, rim, ink(wall_colour), wall_width, camera)])
    return figure
