"""Barycentric hat ramps, edge Whitney 1-forms, the forms on a tetrahedron, and anisotropic conduction fields."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from numga import stack
from examples.util import charts
from examples.util.wgsl import image_figure, meshes, solids, strokes
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
# The tetrahedron's three views, each square, and the gap between them, in pixels.
n_tet_pixels, n_tet_gap = 300, 10
tet_elevation, tet_azimuth = np.radians(23), np.radians(38)
background = np.ones(3)
cream = np.array([0.93, 0.94, 0.96])
light_slate = np.array([0.58, 0.65, 0.70])
warm_tint = np.array([0.96, 0.78, 0.64])
gold = np.array([0.95, 0.68, 0.16])
teal = np.array([0.08, 0.68, 0.56])
primal_rgb, accent_rgb = np.array(to_rgb(primal_colour)), np.array(to_rgb(accent_colour))


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Drawing coordinates of intrinsic positions and directions."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def space_coordinates(points: core.SpacePoint) -> np.ndarray:
    """Drawing coordinates of intrinsic positions and directions in space."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y, mv.z)) & points.batch()[..., None]).to_array())


def outward_wall(mesh: core.SpaceMesh) -> tuple[np.ndarray, np.ndarray]:
    """The faces on the wall of a solid mesh, `[wall]`, and their corners `[wall, 3]`, wound outward. The wall is the
    boundary of the solid, the sum of its tetrahedra's outward faces: one on a wall face that runs outward as stored,
    minus one on a wall face that runs inward, nothing inside."""
    wall = mesh.exterior(2)
    solid = core.space_mesh_module.as_scalar(np.ones(mesh.n_elements[3])).field()
    sense = (~mesh.d[2] * solid).to_array()[wall]
    return wall, np.where(sense[:, None] > 0, mesh.faces[wall], mesh.faces[wall][:, ::-1])


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
    star_centres = coordinates(star_mesh.incidence[2, 0].mean(axis=0) * star_vertices)
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
                     diamond: core.Mesh, diamond_vertices: core.Point, diamond_points: core.Point,
                     shared_forms: core.Plane) -> plt.Figure:
    """The three edge Whitney 1-forms on a single triangle, and the shared edge's basis field `[samples] Plane+[faces]`
    at the samples `[samples] Point-[faces]` of the two faces of the diamond."""
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
    places = coordinates(diamond_vertices)
    frame_panel(ax, places, 0.16)
    for tri_xy in places[diamond.faces]:
        ax.add_patch(Polygon(tri_xy, facecolor="#f6f4ee", edgecolor=wall_colour, linewidth=1.1, zorder=1))
    # Face by face, the samples and the field at them.
    diamond_samples = coordinates(diamond_points).transpose(1, 0, 2).reshape(-1, 2)
    diamond_vectors = coordinates(shared_forms.dual()).transpose(1, 0, 2).reshape(-1, 2)
    diamond_max = np.linalg.norm(diamond_vectors, axis=-1).max()
    ax.quiver(diamond_samples[:, 0], diamond_samples[:, 1],
              diamond_vectors[:, 0], diamond_vectors[:, 1],
              color=primal_colour, angles="xy", scale_units="xy", scale=5.5 * diamond_max,
              width=0.0085, headwidth=3.8, headlength=4.5, pivot="mid", zorder=3)
    start, end = places[diamond.edges[diamond.interior_edges[0]]]
    ax.plot([start[0], end[0]], [start[1], end[1]], color=accent_colour, linewidth=2.4, zorder=4)
    middle, edge_vec = (start + end) / 2, end - start
    ax.annotate("", xy=middle + 0.2 * edge_vec, xytext=middle - 0.2 * edge_vec,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color=accent_colour, linewidth=2.0),
                zorder=5)
    # The corners across the shared edge, in the face on its left and in the face on its right.
    for apex in places[diamond.edge_opposites[0]]:
        ax.scatter(*apex, s=38, color=primal_colour, edgecolor="white", linewidth=0.9, zorder=6)
    return figure


def draw_tet_whitney(mesh: core.SpaceMesh, vertices: core.SpacePoint, points: core.SpacePoint,
                     edge_forms: core.SpacePlane, face_forms: core.SpaceBivector, volume: core.Volume) -> plt.Figure:
    """Three views of a mesh of one tetrahedron: the Whitney 1-form of its first edge, the 2-form of its first face, and
    the linkage of that edge with the edge across it."""
    corner_xyz = space_coordinates(vertices)
    centre_xyz = corner_xyz.mean(axis=0)
    corner_xyz = corner_xyz - centre_xyz
    # Spread the glyphs around the centroid so their directions remain legible.
    sample_xyz = (space_coordinates(points).reshape(-1, 3) - centre_xyz) * 1.22

    edge_field_xyz = space_coordinates(edge_forms[0].dual()).reshape(-1, 3)
    # A face form times the volume's sense is the direction its flux crosses.
    face_field_xyz = space_coordinates(face_forms[0] * (volume / volume.norm())).reshape(-1, 3)

    # The tetrahedron's first edge, the edge across it, the corner the first face form radiates from, and the face
    # facing that corner, as vertices.
    tet = mesh.tets[0]
    edge, across = tet[mesh.corners(3, 1)[0]], tet[core.complements(mesh, 1)[0]]
    apex, face = tet[0], tet[mesh.corners(3, 2)[0]]
    wall, wall_faces = outward_wall(mesh)

    tet_pts = solids.mv.point(corner_xyz)
    view = solids.View(solids.aim(solids.origin, tet_elevation, tet_azimuth), np.full(2, 0.86), n_tet_pixels)

    shell_colours = np.broadcast_to(cream, wall_faces.shape + (3,))
    shell = meshes.samples(tet_pts, wall_faces, shell_colours, meshes.flat(tet_pts, wall_faces), view,
                           face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    wire = strokes.segments(tet_pts[mesh.edges[:, 0]], tet_pts[mesh.edges[:, 1]], light_slate, 1.5, view)

    # The Whitney 1-form of the first edge, circulating around the edge across it.
    edge_scale = 0.30 / np.linalg.norm(edge_field_xyz, axis=-1).max()
    e_starts = solids.mv.point(sample_xyz - 0.45 * edge_scale * edge_field_xyz)
    e_ends = solids.mv.point(sample_xyz + 0.55 * edge_scale * edge_field_xyz)
    e_shafts, e_heads = solids.arrows(e_starts, e_ends, 0.017, 0.065, 0.038)
    active_rod, active_balls = solids.edges(tet_pts[edge[:1]], tet_pts[edge[1:]], 0.024)
    opp_rod, opp_balls = solids.edges(tet_pts[across[:1]], tet_pts[across[1:]], 0.020)
    edge_middle = corner_xyz[edge].mean(axis=0)
    edge_direction = corner_xyz[edge[1]] - corner_xyz[edge[0]]
    edge_arrow_s, edge_arrow_h = solids.arrows(
        solids.mv.point((edge_middle - 0.18 * edge_direction)[None]),
        solids.mv.point((edge_middle + 0.22 * edge_direction)[None]),
        0.025, 0.090, 0.054,
    )
    panel_edge = solids.draw(
        [(e_shafts, accent_rgb), (e_heads, accent_rgb), (active_rod, accent_rgb), (active_balls, accent_rgb),
         (edge_arrow_s, accent_rgb), (edge_arrow_h, accent_rgb), (opp_rod, primal_rgb), (opp_balls, primal_rgb)],
        view, background, inks=[], translucent=[], meshes=[(shell, 0.24), (wire, 1.0)],
    )

    # The Whitney 2-form of the first face, radiating from the corner it faces through that face.
    face_scale = 0.30 / np.linalg.norm(face_field_xyz, axis=-1).max()
    f_starts = solids.mv.point(sample_xyz - 0.45 * face_scale * face_field_xyz)
    f_ends = solids.mv.point(sample_xyz + 0.55 * face_scale * face_field_xyz)
    f_shafts, f_heads = solids.arrows(f_starts, f_ends, 0.017, 0.065, 0.038)
    face_colours = np.broadcast_to(cream, wall_faces.shape + (3,)).copy()
    face_colours[wall == mesh.boundary_indices[2][0, 0]] = warm_tint
    face_mesh = meshes.samples(tet_pts, wall_faces, face_colours, meshes.flat(tet_pts, wall_faces), view,
                               face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    apex_ball = solids.ball(tet_pts[[apex]], 0.046)[:, None]
    face_rim = strokes.segments(tet_pts[face], tet_pts[np.roll(face, -1)], accent_rgb, 2.8, view)
    panel_face = solids.draw(
        [(f_shafts, primal_rgb), (f_heads, primal_rgb), (apex_ball, accent_rgb)],
        view, background, inks=[], translucent=[], meshes=[(face_mesh, 0.30), (wire, 1.0), (face_rim, 1.0)],
    )

    # Rings around the edge across the first edge, which the first edge threads.
    across_middle = corner_xyz[across].mean(axis=0)
    across_direction = corner_xyz[across[1]] - corner_xyz[across[0]]
    across_axis = across_direction / np.linalg.norm(across_direction)
    opp_arrow_s, opp_arrow_h = solids.arrows(
        solids.mv.point((across_middle - 0.18 * across_direction)[None]),
        solids.mv.point((across_middle + 0.22 * across_direction)[None]),
        0.025, 0.090, 0.054,
    )
    radial_1 = edge_middle - across_middle
    radial_1 = radial_1 - np.dot(radial_1, across_axis) * across_axis
    radial_1 = radial_1 / np.linalg.norm(radial_1)
    radial_2 = np.cross(across_axis, radial_1)
    ring_angles = np.linspace(0.0, 2 * np.pi, 65)
    curves = []
    for radius, colour in ((0.20, gold), (0.31, accent_rgb), (0.42, teal)):
        ring_xyz = (across_middle[None, :] * 0.62
                    + radius * (np.cos(ring_angles)[:, None] * radial_1[None, :]
                                + np.sin(ring_angles)[:, None] * radial_2[None, :]))
        curves.append((solids.mv.point(ring_xyz), colour))
    rings = strokes.polylines(curves, 3.2, view)
    panel_link = solids.draw(
        [(active_rod, accent_rgb), (active_balls, accent_rgb), (edge_arrow_s, accent_rgb), (edge_arrow_h, accent_rgb),
         (opp_rod, primal_rgb), (opp_balls, primal_rgb), (opp_arrow_s, primal_rgb), (opp_arrow_h, primal_rgb)],
        view, background, inks=[], translucent=[], meshes=[(shell, 0.22), (wire, 1.0), (rings, 1.0)],
    )

    gap = np.full((n_tet_pixels, n_tet_gap, 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate((panel_edge, gap, panel_face, gap, panel_link), axis=1))


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
