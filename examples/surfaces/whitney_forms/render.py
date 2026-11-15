"""Planar contour and flow panels together with 3D tetrahedral views display Whitney 0-, 1-, and 2-forms and steady anisotropic conduction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import stack
from numga.render import Camera, contours, flow, meshes, simplices, solids
from numga.topology import Topology
from examples.surfaces.whitney_forms import core
from examples.util import charts
from examples.util.panels import image_figure, offset_panel, overlay

primal_colour = "#286580"
accent_colour = "#d77542"
edge_colour = "#b8c4c9"
wall_colour = "#52616a"
flow_colour = "#173c4b"
fibre_colour = "#7f8f96"
form_colours = LinearSegmentedColormap.from_list(
    "whitney_form", ("#f6f4ee", "#8fb8c9", "#286580"))
potential_colours = LinearSegmentedColormap.from_list(
    "whitney_potential", ("#286580", "#f5f2eb", "#d77542"))
n_contours = 7
n_chords = 9
outline_width, side_width, edge_width, wall_width, wire_width, chord_width = 1.6, 3.2, 3.0, 1.8, 1.0, 1.0
arrow_width = 2.8
small_dot, highlight_dot, apex_dot, halo = 2.9, 4.8, 4.3, 1.25
chord_dots = (1.0, 2.0)
chevron_length = 4.0
paper_colour = "#f6f4ee"
row_size, flow_size = (780, 195), (600, 290)
n_tet_pixels, n_tet_gap = 300, 10
tet_elevation, tet_azimuth, face_azimuth = np.radians(23), np.radians(38), np.radians(80)
page_opacity = 0.2
page_width, flux_width = 1.6, 2.0
flux_tip = 0.026
edge_chevron = 3.0
ring_radii = np.array([0.20, 0.31, 0.42])
n_rings = len(ring_radii)
n_ring_samples = 64
background = np.ones(3)
cream = np.array([0.93, 0.94, 0.96])
light_slate = np.array([0.58, 0.65, 0.70])
warm_tint = np.array([0.96, 0.78, 0.64])
gold = np.array([0.95, 0.68, 0.16])
teal = np.array([0.08, 0.68, 0.56])
primal_rgb, accent_rgb = np.array(to_rgb(primal_colour)), np.array(to_rgb(accent_colour))
ring_colours = (gold, accent_rgb, teal)
# Multiplying an intrinsic line by the unit volume element yields the extrinsic rate bivector about that line.
unit_volume = (-core.space_mv.xyzw).dual()                                    # Antipseudoscalar-


# --- plumbing -------------------------------------------------------------------------
def outward_wall(topology: Topology) -> tuple[np.ndarray, np.ndarray]:
    """Boundary coboundary signs select the exterior faces of a tetrahedral complex and orient each face outward."""
    V, E, F, T = topology.n_elements
    wall = topology.exterior[2]
    solid = core.space_mv.scalar(np.ones((T, 1))).field()                    # Scalar+[T]
    sense = (topology.reading(wall, 2) * (topology.d[2].adjugate() * solid)).to_array()
    return wall, np.where(sense[:, None] > 0, topology.elements[2][wall], topology.elements[2][wall][:, ::-1])


def panel_camera(ax: plt.Axes, points: core.Point, margin_ratio: float) -> Camera:
    """An equal-aspect square viewport frames `points` without axes and defines its planar camera."""
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
    """A hex colour string converts to a three-channel RGB array (`[3]`)."""
    return np.array(to_rgb(colour))


def ringed_markers(points: core.Point, colour: str, radius: float, camera: Camera) -> list[np.ndarray]:
    """White-haloed circular markers of `radius` pixels highlight `points` in `colour`."""
    return [simplices.markers(points, np.ones(3), radius + halo, camera),
            simplices.markers(points, ink(colour), radius, camera)]


def arrows(tails: core.Point, tips: core.Point, width: float, camera: Camera) -> np.ndarray:
    """Chevron-tipped strokes of `width` pixels connect `tails` to `tips` in the accent colour."""
    n_arrows = tails.shape[0]
    return simplices.strokes(stack((tails, tips), axis=-1).reshape(-1), Topology.chains([2] * n_arrows),
                             accent_rgb, width, camera, chevrons=np.ones(n_arrows), chevron=chevron_length * width,
                             cap=simplices.Cap.BUTT)


def vertex_forms_figure(
    triangle: Topology,
    triangle_vertices: core.Point,
    gradients: core.Plane,
    star_topology: Topology,
    star_vertices: core.Point,
    vertex_form: core.Scalar,
) -> plt.Figure:
    """Four planar panels show the three corner vertex 0-forms with their simplex gradients alongside an assembled vertex star."""
    V_star, E_star, F_star = star_topology.n_elements
    corners = triangle_vertices.batch()                                       # [3] Point-
    face_gradients = gradients.batch()[:, 0]                                  # [3] Plane+
    levels = tuple(contours.Level(level, np.append(primal_rgb, 0.55), 1.0)
                   for level in np.linspace(0.0, 1.0, n_contours + 2)[1:-1])
    centroid = corners.mean(axis=0)                                           # [] Point-
    rises = face_gradients.dual()                                             # [3] Point-
    lengths = face_gradients.norm()                                           # [3] Scalar+
    reaches = rises * (0.24 / lengths.to_array().max())                       # [3] Point-

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    tails, heads = triangle.corners[2, 1]
    for index in range(3):
        ax = axes[index]
        camera = panel_camera(ax, corners, 0.22)
        values = np.eye(3)[index]
        overlay(ax, [
            contours.picture(corners, triangle, values, form_colours(values)[:, :3], camera, levels=levels),
            simplices.strokes(corners, triangle, ink(wall_colour), outline_width, camera),
            simplices.strokes(corners[[tails[index], heads[index]]], Topology.chains([2]),
                              ink(wall_colour), edge_width, camera),
            simplices.markers(corners, ink(wall_colour), small_dot, camera),
            *ringed_markers(corners[index][None], accent_colour, highlight_dot, camera),
            arrows((centroid - reaches[index])[None], (centroid + reaches[index])[None], arrow_width, camera)])

    ax = axes[3]
    star_values = np.asarray(vertex_form.batch().to_array())
    star_places = star_vertices.batch()                                       # [V_star] Point-
    camera = panel_camera(ax, star_vertices, 0.14)
    overlay(ax, [
        contours.picture(star_places, star_topology, star_values, form_colours(star_values)[:, :3], camera, levels=levels),
        simplices.strokes(star_places, star_topology, ink(wall_colour), wire_width, camera, opacity=0.65),
        *ringed_markers(star_places[:1], accent_colour, highlight_dot, camera)])
    return figure


def edge_forms_figure(
    triangle: Topology,
    triangle_vertices: core.Point,
    corner_forms: core.Plane,
    diamond: Topology,
    diamond_vertices: core.Point,
    shared_corner_forms: core.Plane,
) -> plt.Figure:
    """Four planar panels show the three local edge Whitney 1-forms and tangential continuity across a two-triangle diamond seam."""
    V_diamond, E_diamond, F_diamond = diamond.n_elements
    corners = triangle_vertices.batch()                                       # [3] Point-
    face_forms = corner_forms.batch()[..., 0]                                 # [3, 3] Plane+
    directions = face_forms.dual()                                            # [3, 3] Point-
    reference_speed = float(np.percentile(face_forms.norm().to_array(), 95))
    first, second = np.where(triangle.orientation[1] > 0, triangle.complements[2, 0], triangle.complements[2, 0][::-1])
    side_starts, side_ends = corners[first], corners[second]                  # [3] Point-, Point-

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index in range(3):
        ax = axes[index]
        camera = panel_camera(ax, corners, 0.22)
        overlay(ax, [
            contours.picture(corners, triangle, np.zeros(3), ink(paper_colour), camera),
            simplices.strokes(corners, triangle, ink(wall_colour), charts.LINE_PX, camera),
            flow.layer(corners, triangle, directions[index], camera, reference_speed, primal_rgb),
            arrows(side_starts[index][None], side_ends[index][None], side_width, camera),
            *ringed_markers(corners[index][None], primal_colour, highlight_dot, camera)])

    ax = axes[3]
    camera = panel_camera(ax, diamond_vertices, 0.16)
    places = diamond_vertices.batch()                                         # [V_diamond] Point-
    # Keeping the two triangles' corners separate preserves the normal jump across the shared interior edge.
    diamond_corners = (diamond.incidence[2, 0] * diamond_vertices).batch()    # [3, F_diamond] Point-
    positions = diamond_corners.reshape(-1)
    seam_directions = shared_corner_forms.dual().batch().reshape(-1)
    split_topology = Topology(3 * F_diamond, np.arange(3 * F_diamond).reshape(3, F_diamond).T)
    seam_speed = float(np.percentile(shared_corner_forms.batch().norm().to_array(), 95))
    shared_reading = diamond.reading(diamond.interior[1][:1], 1)              # Scalar+[1 <- E_diamond]
    tail, head = (shared_reading * (diamond.incidence[1, 0] * diamond_vertices)).batch()[:, 0]  # [] Point-, Point-
    apexes = (diamond.reading(diamond.facet_opposites[0], 0) * diamond_vertices).batch()  # [2] Point-
    overlay(ax, [
        contours.picture(places, diamond, np.zeros(V_diamond), ink(paper_colour), camera),
        simplices.strokes(places, diamond, ink(wall_colour), charts.LINE_PX, camera),
        flow.layer(positions, split_topology, seam_directions, camera, seam_speed, primal_rgb),
        arrows(tail[None], head[None], side_width, camera),
        *ringed_markers(apexes, primal_colour, apex_dot, camera)])
    return figure


def tetrahedron_forms_figure(
    topology: Topology,
    vertices: core.SpacePoint,
    pages: core.SpacePoint,
    field_lines: core.SpacePoint,
) -> plt.Figure:
    """Three perspective panels display edge 1-form level pages, face 2-form flux lines, and opposite-edge linkage on a tetrahedron."""
    V, E, F, T = topology.n_elements
    n_pages = pages.shape[1]
    n_lines = field_lines.shape[1]
    centroid = vertices.sites.mean()                                          # [] Point-
    corners = (vertices - centroid).batch()                                   # [V] Point-
    places = solids.origin + solids.context.include_from_algebra(corners)     # [V] Point-

    tet_corners = (topology.incidence[3, 0] * (vertices - centroid)).batch()[:, 0]  # [4] Point-
    tet_places = solids.origin + solids.context.include_from_algebra(tet_corners)   # [4] Point-
    first_edge, opposite_edge = topology.corners[3, 1][:, 0], topology.complements[3, 1][:, 0]
    apex, face = 0, topology.complements[3, 0][:, 0]
    wall, wall_faces = outward_wall(topology)

    view = solids.View(solids.aim(solids.origin, tet_elevation, tet_azimuth), np.full(2, 0.86), n_tet_pixels)
    face_view = view._replace(camera=solids.aim(solids.origin, tet_elevation, face_azimuth))

    shell_colours = np.broadcast_to(cream, wall_faces.shape + (3,))
    shell = meshes.samples(places, wall_faces, shell_colours, meshes.flat(places, wall_faces), view,
                           face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    wire = simplices.tubes(places, topology, light_slate, 1.5, view, solids.gloss)
    active_width, opp_width = 2 * 0.024 / view.pixel, 2 * 0.020 / view.pixel
    active_arrow = simplices.tubes(tet_places[first_edge], Topology.chains([2]), accent_rgb, active_width, view,
                                   solids.gloss, chevrons=np.ones(1), chevron=edge_chevron * active_width)
    opp_arrow = simplices.tubes(tet_places[opposite_edge], Topology.chains([2]), primal_rgb, opp_width, view,
                                solids.gloss, chevrons=np.ones(1), chevron=edge_chevron * opp_width)
    opp_rod, opp_balls = solids.edges(tet_places[opposite_edge[:1]], tet_places[opposite_edge[1:]], 0.020)

    first_pages = (pages[0] - centroid).batch()                               # [n_pages, 3, T] Point-
    page_places = solids.origin + solids.context.include_from_algebra(first_pages.reshape(-1))  # [n_pages * 3 * T] Point-
    page_triangles = np.arange(len(page_places)).reshape(first_pages.shape).swapaxes(1, 2)  # [n_pages, T, 3]
    page_fills = [(meshes.samples(page_places, triangles, np.broadcast_to(warm_tint, triangles.shape + (3,)),
                                  meshes.flat(page_places, triangles), view, face_values=np.zeros(len(triangles)),
                                  colour=meshes.plain, gloss=solids.gloss), page_opacity) for triangles in page_triangles]
    page_sides = np.stack((page_triangles[..., :2], page_triangles[..., 1:]), axis=-2).reshape(-1, 2)
    page_rims = simplices.tubes(page_places, Topology(len(page_places), page_sides), accent_rgb, page_width,
                                view, solids.gloss)
    panel_edge = solids.draw(
        [(opp_rod, primal_rgb), (opp_balls, primal_rgb)],
        view, background, inks=[], translucent=[],
        meshes=[(shell, 0.24), (wire, 1.0), *page_fills, (page_rims, 1.0), (active_arrow, 1.0)],
    )

    first_lines = (field_lines[0] - centroid).batch()                         # [n_lines, 2, T] Point-
    line_places = solids.origin + solids.context.include_from_algebra(first_lines.reshape(-1))  # [n_lines * 2 * T] Point-
    line_ends = np.arange(len(line_places)).reshape(first_lines.shape).swapaxes(1, 2).reshape(-1, 2)  # [n_lines * T, 2]
    flux_lines = simplices.tubes(line_places, Topology(len(line_places), line_ends), primal_rgb, flux_width,
                                 face_view, solids.gloss)
    flux_tips = solids.ball(line_places[line_ends[:, 1]], flux_tip)[:, None]
    face_colours = np.broadcast_to(cream, wall_faces.shape + (3,)).copy()
    face_colours[wall == topology.boundary_indices[2][0, 0]] = warm_tint
    face_surface = meshes.samples(places, wall_faces, face_colours, meshes.flat(places, wall_faces), face_view,
                                  face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    apex_ball = solids.ball(tet_places[[apex]], 0.046)[:, None]
    face_rim = simplices.tubes(tet_places[face], Topology.loops([3]), accent_rgb, 2.8, face_view, solids.gloss)
    face_wire = simplices.tubes(places, topology, light_slate, 1.5, face_view, solids.gloss)
    panel_face = solids.draw(
        [(flux_tips, primal_rgb), (apex_ball, accent_rgb)],
        face_view, background, inks=[], translucent=[],
        meshes=[(face_surface, 0.30), (face_wire, 1.0), (face_rim, 1.0), (flux_lines, 1.0)],
    )

    edge_middle, opposite_middle = tet_corners[first_edge].mean(axis=0), tet_corners[opposite_edge].mean(axis=0)  # [] Point-, Point-
    ring_centre = centroid + opposite_middle * 0.62                           # [] Point-
    opposite_step = tet_corners[opposite_edge[1]] - tet_corners[opposite_edge[0]]  # [] Point-
    ring_axis = (ring_centre & opposite_step).normalized()                    # [] Line-
    # Intersecting the plane along the opposite edge with the orthogonal plane at ring_centre places the starting point.
    plane_cut = ((ring_centre + edge_middle - opposite_middle) & opposite_step) ^ (ring_centre | ring_axis)  # [] Point-
    start = plane_cut / (core.space_mv.w & plane_cut)                         # [] Point-
    radial = (start - ring_centre) / (ring_centre & start).norm()             # [] Point-
    ring_turns = ((ring_axis * unit_volume) * (-np.linspace(0.0, 2 * np.pi, n_ring_samples, endpoint=False) / 2)).exp()  # [n_ring_samples] Motor+
    radial_circle = ring_turns >> radial                                      # [n_ring_samples] Point-
    ring_points = ring_centre + ring_radii[:, None] * radial_circle           # [n_rings, n_ring_samples] Point-
    ring_places = solids.origin + solids.context.include_from_algebra(ring_points - centroid)  # [n_rings, n_ring_samples] Point-
    rings = simplices.tubes(ring_places.reshape(-1), Topology.loops([n_ring_samples] * n_rings),
                            np.repeat(np.stack(ring_colours), n_ring_samples, axis=0), 3.2, view, solids.gloss)
    panel_link = solids.draw(
        [], view, background, inks=[], translucent=[],
        meshes=[(shell, 0.22), (wire, 1.0), (rings, 1.0), (active_arrow, 1.0), (opp_arrow, 1.0)],
    )

    gap = np.full((n_tet_pixels, n_tet_gap, 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate((panel_edge, gap, panel_face, gap, panel_link), axis=1))


def conduction_figure(
    topology: Topology,
    vertices: core.Point,
    potential: core.Scalar,
    current: core.Point,
    fibre: core.Point,
) -> plt.Figure:
    """Side-by-side disk panels compare isotropic and tilted-anisotropic steady conduction on a shared potential palette and streak scale."""
    V, E, F = topology.n_elements
    n_cases = potential.shape[0]
    potentials = np.asarray(potential.batch().to_array())
    potential_levels = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.45), 1.0)
                             for level in np.linspace(-0.85, 0.85, n_contours + 2)[1:-1])

    # Area-weighted averaging onto vertices produces continuous streak fields from piecewise-constant face currents.
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [3] Point-[F]
    areas = ((sides[0] & sides[1]) / 2).antinorm()                            # Scalar+[F]
    vertex_averaging = topology.incidence[2, 0].mean(axis=0).adjugate()       # Scalar+[V <- F]
    vertex_current = (vertex_averaging * (areas * current)) / (vertex_averaging * areas)  # [n_cases] Point-[V]
    reference_speed = float(np.percentile(vertex_current.antinorm().batch().to_array(), 95))
    points = vertices.batch()                                                 # [V] Point-
    directions = vertex_current.batch()                                       # [n_cases, V] Point-
    rim = topology.border

    quarter_turn = (core.mv.xy * (-np.pi / 4)).exp()                          # [] Rotor+
    transverse = quarter_turn >> fibre                                        # [] Point-
    offsets = np.linspace(-0.85, 0.85, n_chords)                              # [n_chords]
    half_chords = np.sqrt(1 - offsets**2)                                     # [n_chords]
    chord_ends = transverse * offsets[:, None] + fibre * (half_chords[:, None] * np.array([-1, 1]))  # [n_chords, 2] Point-

    figure, axes = charts.subplots(*flow_size, ncols=2)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98, wspace=0.05)

    for index, ax in enumerate(axes):
        camera = panel_camera(ax, vertices, 0.08)
        colours = potential_colours(Normalize(-1.0, 1.0)(potentials[index]))[:, :3]
        fibres = [simplices.strokes(chord_ends.reshape(-1), Topology.chains([2] * n_chords),
                                    ink(fibre_colour), chord_width, camera, dashes=chord_dots, opacity=0.55)
                  ] if index == 1 else []
        overlay(ax, [
            contours.picture(points, topology, potentials[index], colours, camera, levels=potential_levels),
            simplices.strokes(points, topology, ink(edge_colour), wire_width, camera, opacity=0.45),
            *fibres,
            flow.layer(points, topology, directions[index], camera, reference_speed, ink(flow_colour)),
            simplices.strokes(points, rim, ink(wall_colour), wall_width, camera)])
    return figure
