"""Tetrahedral forms and smoothed field-line tubes show magnetic linkage and selective decay."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from examples.electromagnetism.beltrami_fields import core
from examples.util import charts
from examples.util.wgsl import image_figure, meshes, show_pixels, solids, strokes

n_tet_pixels = 300
n_torus_pixels = 320
n_anim_pixels = 270
# The spectrum's width beside the two torus panels, in pixels: what the column leaves them.
n_spectrum_pixels = charts.DESIGN_WIDTH - 2 * n_torus_pixels
# The spectrum's margins, left, bottom, right and top, in pixels: room for its tick labels.
spectrum_margins = np.array([44, 26, 8, 14])
# The two shown modes, twice the size of the others, as a scatter's area in points squared.
shown_mode_size = charts.points(2 * charts.MARKER_PX) ** 2
background = np.ones(3)
sapphire = np.array([0.20, 0.48, 0.86])
teal = np.array([0.08, 0.68, 0.56])
gold = np.array([0.95, 0.68, 0.16])
vermilion = np.array([0.88, 0.32, 0.16])
dark_slate = np.array([0.22, 0.28, 0.34])
light_slate = np.array([0.58, 0.65, 0.70])
cream = np.array([0.93, 0.94, 0.96])
warm_tint = np.array([0.96, 0.78, 0.64])
tet_elevation, tet_azimuth = np.radians(23), np.radians(38)
torus_elevation, torus_azimuth = np.radians(28), np.radians(36)
top_elevation, top_azimuth = np.radians(82), np.radians(36)
# Up, and the unit prism, an intrinsic scalar, the join of a unit cube's edges and time: the patch a direction sweeps
# through time, times it, is the plane of space square to the direction.
up = core.mv.z.dual()                                                         # [] Point-
unit_prism = core.mv.antipseudoscalar([1.0])                                  # [] Antipseudoscalar-
# The glyphs' spread about the tetrahedron's middle, so their directions remain legible; the rings' radii about the
# opposite edge, their share of the way from the middle out to that edge, and their points.
spread = 1.22
ring_radii = np.array([0.20, 0.31, 0.42])
ring_reach = 0.62
n_ring_points = 64
# The torus the field lines are traced in: its radius and its tube's, which tapers their motion off the wall; the
# shells of seeds, each its share of the tube's radius, its count of seeds, and the angles around the tube and the
# axis of its first seed; the traced steps either way from a seed, their length, and every how many a point is kept.
tube_major, tube_minor = 1.0, 0.33
seed_shares = np.array([0.24, 0.52, 0.78])
n_seeds = np.array([3, 4, 5])
seed_angles_around_tube = np.array([0.20, 1.10, 2.15])
seed_angles_around_axis = np.array([0.0, np.pi / 4, np.pi / 5])
n_trace_steps, trace_step, trace_stride = 170, 0.018, 2
# The field's smoothing: its reach around the axis, as a chord of the unit circle, and across the tube; the field's
# floor in its normalisation, and the inward pull at the wall, as shares of its typical strength.
smoothing_turn, smoothing_reach = 0.58, 0.20
speed_floor, wall_pull = 0.14, 0.08


# --- plumbing -------------------------------------------------------------------------
def gpu_points(points: core.Point) -> solids.Point:
    """The GPU library's own points at the places of the example's, a field's sites read as a batch: their
    coordinates, read by pairing the intrinsic points with the frame planes, are the one hand-off from this algebra to
    the renderer's."""
    mv = points.context.multivector
    return solids.mv.point((stack((mv.x, mv.y, mv.z)) & points.batch()[..., None]).to_array())


def length(directions: core.Point) -> core.Scalar:
    """The lengths of directions of space: the size of the patch each sweeps through a unit of time."""
    return (directions & core.time).norm()


def outward_wall(mesh: core.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The faces on the wall of a solid mesh, `[wall]`, and their corners `[wall, 3]`, wound outward. The wall is the
    boundary of the solid, the sum of its tetrahedra's outward faces: one on a wall face that runs outward as stored,
    minus one on a wall face that runs inward, nothing inside."""
    wall = mesh.exterior(2)
    sense = (~mesh.d[2] * core.as_scalar(np.ones(mesh.n_elements[3])).field()).to_array()[wall]
    return wall, np.where(sense[:, None] > 0, mesh.faces[wall], mesh.faces[wall][:, ::-1])


def draw_tet_whitney(mesh: core.Mesh, vertices: core.Point, points: core.Point, edge_forms: core.Plane,
                     face_forms: core.Field, volume: core.Volume) -> plt.Figure:
    """Three 3D views of a mesh of one tetrahedron: edge 1-form, face 2-form, and opposite-edge helicity linkage."""
    # Everything about the tetrahedron's middle, the glyphs spread out from it.
    middle = vertices.batch().mean(axis=0)                                    # [] Point-
    corners = (mesh.incidence[3, 0] * vertices).batch()[:, 0] - middle       # [corners] Point-
    samples = (points.batch().reshape(-1) - middle) * spread                  # [samples] Point-

    edge_field = core.Plane.dual()(edge_forms[0]).batch().reshape(-1)         # [samples] Point-
    unit_volume = volume / (volume & core.time).norm()                        # Volume-[tets]
    face_field = (face_forms[0] * unit_volume).batch().reshape(-1)            # [samples] Point-

    # The tetrahedron's first edge, the edge across it, the corner the first face form radiates from, and the face
    # facing that corner, as its corners.
    edge, across = corners[mesh.corners(3, 1)[0]], corners[core.complements(mesh, 1)[0]]   # [2] Point-, [2] Point-
    apex, face = corners[0], corners[mesh.corners(3, 2)[0]]                   # [] Point-, [3] Point-
    wall, wall_faces = outward_wall(mesh)

    tet_pts = gpu_points(vertices.batch() - middle)
    view = solids.View(solids.aim(solids.origin, tet_elevation, tet_azimuth), np.full(2, 0.86), n_tet_pixels)

    shell_colours = np.broadcast_to(cream, wall_faces.shape + (3,))
    shell = meshes.samples(tet_pts, wall_faces, shell_colours, meshes.flat(tet_pts, wall_faces), view,
                           face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    tails, heads = mesh.incidence[1, 0] * vertices                            # Point-[edges], Point-[edges]
    wire = strokes.segments(gpu_points(tails.batch() - middle), gpu_points(heads.batch() - middle), light_slate, 1.5, view)

    # Panel 1: the Whitney 1-form of edge (0, 1), circulating around the opposite edge (2, 3).
    edge_scale = 0.30 / length(edge_field).to_array().max()
    e_starts = gpu_points(samples - edge_field * (0.45 * edge_scale))
    e_ends = gpu_points(samples + edge_field * (0.55 * edge_scale))
    e_shafts, e_heads = solids.arrows(e_starts, e_ends, 0.017, 0.065, 0.038)
    edge_pts, across_pts = gpu_points(edge), gpu_points(across)
    active_rod, active_balls = solids.edges(edge_pts[:1], edge_pts[1:], 0.024)
    opp_rod, opp_balls = solids.edges(across_pts[:1], across_pts[1:], 0.020)
    mid_01, dir_01 = edge.mean(axis=0), edge[1] - edge[0]                     # [] Point-, Point-
    edge_arrow_s, edge_arrow_h = solids.arrows(gpu_points(mid_01 - dir_01 * 0.18)[None],
                                               gpu_points(mid_01 + dir_01 * 0.22)[None], 0.025, 0.090, 0.054)
    panel_edge = solids.draw(
        [(e_shafts, vermilion), (e_heads, vermilion), (active_rod, vermilion), (active_balls, vermilion),
         (edge_arrow_s, vermilion), (edge_arrow_h, vermilion), (opp_rod, sapphire), (opp_balls, sapphire)],
        view, background, inks=[], translucent=[], meshes=[(shell, 0.24), (wire, 1.0)],
    )

    # Panel 2: the Whitney 2-form of face (1, 2, 3), radiating from the opposite corner 0 through that face.
    face_scale = 0.30 / length(face_field).to_array().max()
    f_starts = gpu_points(samples - face_field * (0.45 * face_scale))
    f_ends = gpu_points(samples + face_field * (0.55 * face_scale))
    f_shafts, f_heads = solids.arrows(f_starts, f_ends, 0.017, 0.065, 0.038)
    face_colours = np.broadcast_to(cream, wall_faces.shape + (3,)).copy()
    face_colours[wall == mesh.boundary_indices[2][0, 0]] = warm_tint
    face_mesh = meshes.samples(tet_pts, wall_faces, face_colours, meshes.flat(tet_pts, wall_faces), view,
                               face_values=np.zeros(len(wall)), colour=meshes.plain, gloss=solids.gloss)
    apex_ball = solids.ball(gpu_points(apex)[None], 0.046)[:, None]
    face_rim = strokes.segments(gpu_points(face), gpu_points(face[[1, 2, 0]]), vermilion, 2.8, view)
    panel_face = solids.draw(
        [(f_shafts, sapphire), (f_heads, sapphire), (apex_ball, vermilion)],
        view, background, inks=[], translucent=[], meshes=[(face_mesh, 0.30), (wire, 1.0), (face_rim, 1.0)],
    )

    # Panel 3: Topological helicity linkage between opposite edges (0, 1) and (2, 3).
    mid_23, dir_23 = across.mean(axis=0), across[1] - across[0]               # [] Point-, Point-
    opp_arrow_s, opp_arrow_h = solids.arrows(gpu_points(mid_23 - dir_23 * 0.18)[None],
                                             gpu_points(mid_23 + dir_23 * 0.22)[None], 0.025, 0.090, 0.054)
    # The rings turn about the opposite edge, in the plane of space square to it, starting from the direction toward
    # the first edge, less its part along the opposite edge.
    axis = dir_23 / length(dir_23)                                            # [] Point-
    toward = mid_01 - mid_23                                                  # [] Point-
    square = toward - axis * (axis | toward)                                  # [] Point-
    start = square / length(square)                                           # [] Point-
    turning = (axis & core.time) * unit_prism                                 # [] Field+
    turns = (turning * (np.linspace(0.0, 2 * np.pi, n_ring_points + 1) / 2)).exp()   # [ring points] Rotor+
    rings = mid_23 * ring_reach + (turns >> start) * ring_radii[:, None]      # [rings, ring points] Point-
    ring_pts = gpu_points(rings)
    curves = strokes.polylines([(ring_pts[ring], colour) for ring, colour in enumerate((gold, vermilion, teal))], 3.2, view)
    panel_link = solids.draw(
        [(active_rod, vermilion), (active_balls, vermilion), (edge_arrow_s, vermilion), (edge_arrow_h, vermilion),
         (opp_rod, sapphire), (opp_balls, sapphire), (opp_arrow_s, sapphire), (opp_arrow_h, sapphire)],
        view, background, inks=[], translucent=[], meshes=[(shell, 0.22), (wire, 1.0), (curves, 1.0)],
    )

    separator = np.full((n_tet_pixels, 10, 3), 255, dtype=np.uint8)
    combined = np.concatenate((panel_edge, separator, panel_face, separator, panel_link), axis=1)
    return image_figure(combined)


def field_lines(centres: core.Point, volumes: core.Volume, magnetic: core.Field) -> tuple[core.Point, np.ndarray]:
    """Smooth, strictly confined magnetic field lines inside the torus `[lines, steps]`, and the shell each starts on
    `[lines]`, traced by RK4 through the field smoothed around the axis and across the tube and tapered off the wall.
    The field at the tetrahedra's centres, each a direction, is read along its cylindrical frame, out from the axis,
    around it and up; at a point, those readings averaged with smooth weights by how near the samples lie around the
    axis and across the tube are rebuilt along the point's own frame."""
    mv = core.mv
    # The seeds: points on circles around the tube's core, turned about the axis.
    shells = np.repeat(np.arange(len(n_seeds)), n_seeds)                      # [lines]
    # Each shell's seeds evenly spaced around the tube and around the axis alike.
    spacing = np.concatenate([np.arange(count) * (2 * np.pi / count) for count in n_seeds])   # [lines]
    around_tube = (mv.xz * (-(seed_angles_around_tube[shells] + spacing) / 2)).exp()   # [lines] Rotor+
    around_axis = (mv.xy * (-(seed_angles_around_axis[shells] + spacing) / 2)).exp()   # [lines] Rotor+
    outward = mv.x.dual()                                                     # [] Point-
    seeds = around_axis >> (outward * tube_major + (around_tube >> outward) * (seed_shares[shells] * tube_minor))   # [lines] Point-

    # Symmetrize the two outer tetrahedra of each 3-tet prism wedge to remove even/odd diagonal split alternation.
    unit_volumes = volumes / (volumes & core.time).norm()                     # Volume-[tets]
    sites, fields = centres.batch().reshape(3, -1), (magnetic * unit_volumes).batch().reshape(3, -1)   # [3, wedges] Point-
    sites = concatenate(((sites[0] + sites[1]) / 2, sites[2]))                # [samples] Point-
    fields = concatenate(((fields[0] + fields[1]) / 2, fields[2]))            # [samples] Point-

    # Each sample's distance from the axis and height, its frame, and the field read along that frame.
    heights = mv.z & sites                                                    # [samples] Scalar+
    level = sites - up * heights                                              # [samples] Point-
    distances = length(level)                                                 # [samples] Scalar+
    outwards = level / distances                                              # [samples] Point-
    frames = stack([outwards, outwards.commutator(mv.xy), up.broadcast_to(outwards.shape)], axis=-1)   # [samples, 3] Point-
    readings = (frames | fields[:, None]).to_array()                          # [samples, 3]
    sample_distances, sample_heights = distances.to_array(), heights.to_array()   # [samples], [samples]
    typical = float((fields | fields).mean(axis=0).square_root().to_array())

    def velocity(points: core.Point, senses: np.ndarray) -> core.Point:
        """The unit direction of the smoothed, tapered field at points `[senses, lines]`, along each sense."""
        height = mv.z & points                                                # [senses, lines] Scalar+
        level = points - up * height                                          # [senses, lines] Point-
        distance = length(level)                                              # [senses, lines] Scalar+
        outward = level / distance                                            # [senses, lines] Point-
        # How near each sample lies around the axis, as the chord between the unit directions out to each, and
        # across the tube: the smoothing's weights, numbers that average the samples' readings.
        chords = 2 - 2 * (outward[..., None] | outwards).to_array()           # [senses, lines, samples]
        across = ((distance.to_array()[..., None] - sample_distances)**2
                  + (height.to_array()[..., None] - sample_heights)**2)       # [senses, lines, samples]
        weights = np.exp(-chords / smoothing_turn**2 - across / smoothing_reach**2)
        smoothed = senses[..., None] * (weights @ readings) / weights.sum(axis=-1, keepdims=True)   # [senses, lines, 3]
        frame = stack([outward, outward.commutator(mv.xy), up.broadcast_to(outward.shape)], axis=-1)   # [senses, lines, 3] Point-
        field = (frame * smoothed).sum(axis=-1)                               # [senses, lines] Point-

        # Smoothly taper motion out from the tube's core toward the conducting wall, so streamlines glide along flux
        # surfaces.
        offset = points - outward * tube_major                                # [senses, lines] Point-
        reach = length(offset)                                                # [senses, lines] Scalar+
        tube_out = offset / reach                                             # [senses, lines] Point-
        share = np.clip(reach.to_array() / tube_minor, 0.0, 1.0)             # [senses, lines]
        out = (field | tube_out).to_array()                                   # [senses, lines]
        tapered = out * np.cos(0.5 * np.pi * share) ** 2 - wall_pull * share**3 * typical
        field = field + tube_out * (tapered - out)                            # [senses, lines] Point-
        return field / np.sqrt((field | field).to_array() + (speed_floor * typical) ** 2)

    # Both ways from each seed at once, every `trace_stride` steps kept.
    senses = np.array([-1.0, 1.0])[:, None]                                   # [senses, 1]
    points = seeds[None].broadcast_to((2, len(shells)))                       # [senses, lines] Point-
    traced = [points]
    for step in range(1, n_trace_steps + 1):
        k1 = velocity(points, senses)
        k2 = velocity(points + k1 * (0.5 * trace_step), senses)
        k3 = velocity(points + k2 * (0.5 * trace_step), senses)
        k4 = velocity(points + k3 * trace_step, senses)
        points = points + (k1 + 2 * k2 + 2 * k3 + k4) * (trace_step / 6.0)
        if step % trace_stride == 0:
            traced.append(points)
    traced = stack(traced, axis=-1)                                           # [senses, lines, steps] Point-
    # Backward from the far end to the seed, then forward.
    return concatenate((traced[0, :, :0:-1], traced[1]), axis=-1), shells


def palette_blend(twist: np.ndarray, shell: np.ndarray) -> np.ndarray:
    """Blend a rich scientific palette by radial flux shell and local helical pitch."""
    base_colours = np.array([vermilion, teal, sapphire])[shell]               # [lines, 3]
    # Modulate toward warm gold where the field line pitches strongly upward/poloidally.
    pitch_weight = np.clip(0.5 + 0.45 * twist, 0.0, 1.0)[..., None]
    blended = (1.0 - 0.38 * pitch_weight) * base_colours[:, None, :] + (0.38 * pitch_weight) * gold[None, None, :]
    return np.clip(blended, 0.0, 1.0)


def render_torus_field(mesh: core.Mesh, vertices: core.Point, lines: core.Point, shells: np.ndarray,
                       view: solids.View) -> np.ndarray:
    """Continuous 3D magnetic field-line tubes `[lines, steps]` inside a smooth translucent toroidal cavity, each
    coloured by its shell `[lines]`."""
    torus_pts = gpu_points(vertices)
    _, b_faces = outward_wall(mesh)
    shell_colours = np.broadcast_to(cream, b_faces.shape + (3,))
    shell = meshes.samples(torus_pts, b_faces, shell_colours, meshes.smooth(torus_pts, b_faces), view,
                           face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)

    starts, ends = lines[:, :-1], lines[:, 1:]                                # [lines, segments] Point-
    # Local helical twist: how steeply each segment climbs, its unit direction's height.
    climb = (core.mv.z & (ends - starts)) / length(ends - starts)            # [lines, segments] Scalar+
    seg_colours = palette_blend(climb.to_array(), shells)                     # [lines, segments, 3]
    tubes = strokes.segments(gpu_points(starts.reshape(-1)), gpu_points(ends.reshape(-1)), seg_colours.reshape(-1, 3),
                             4.6, view)

    # Place glossy 3D cone arrowheads along each field line to show its flow direction.
    middle = lines.shape[1] // 2
    arrow_starts = lines[:, middle - 2]                                       # [lines] Point-
    heading = lines[:, middle + 2] - arrow_starts                             # [lines] Point-
    _, a_heads = solids.arrows(gpu_points(arrow_starts), gpu_points(arrow_starts + heading * (0.14 / length(heading))),
                               0.018, 0.082, 0.044)
    head_colours = np.array([vermilion, teal, sapphire])[shells]

    return solids.draw([(a_heads, head_colours)], view, background, inks=[], translucent=[],
                       meshes=[(shell, 0.16), (tubes, 1.0)])


@charts.style()
def draw_beltrami_eigenmodes(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                             volumes: core.Volume, inverse_eigenvalues: core.Scalar, mode_magnetic: core.Field,
                             shown_indices: np.ndarray) -> plt.Figure:
    """The chiral Beltrami spectrum beside 3D field-line renders of the Taylor state and a higher-order mode."""
    mu = np.asarray(inverse_eigenvalues.to_array())
    n_modes = len(mu)
    indices = np.arange(n_modes)

    view = solids.View(solids.aim(solids.origin, torus_elevation, torus_azimuth), np.full(2, 1.38), n_torus_pixels)
    taylor_img = render_torus_field(mesh, vertices, *field_lines(centres, volumes, mode_magnetic[0]), view)
    excited_img = render_torus_field(mesh, vertices, *field_lines(centres, volumes, mode_magnetic[1]), view)

    width, height = charts.DESIGN_WIDTH, n_torus_pixels
    figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI))
    left, bottom, right, top = spectrum_margins
    ax_spec = figure.add_axes([left / width, bottom / height,
                               (n_spectrum_pixels - left - right) / width, (height - bottom - top) / height])
    ax_taylor = figure.add_axes([n_spectrum_pixels / width, 0.0, n_torus_pixels / width, 1.0])
    ax_excited = figure.add_axes([(n_spectrum_pixels + n_torus_pixels) / width, 0.0, n_torus_pixels / width, 1.0])

    # The zero line as thin as the chart's frame.
    ax_spec.axhline(0.0, color="#6c7a82", linewidth=charts.FRAME_WIDTH, zorder=1)
    negative = mu < 0
    positive = mu > 0
    ax_spec.scatter(indices[negative], mu[negative], color=sapphire, alpha=0.82, zorder=2)
    ax_spec.scatter(indices[positive], mu[positive], color=vermilion, alpha=0.82, zorder=2)
    resolved = shown_indices % n_modes
    ax_spec.scatter(resolved, mu[resolved], s=shown_mode_size, color=[vermilion, teal], edgecolor="white", zorder=4)
    mu_max = float(np.abs(mu).max())
    ax_spec.set_xlim(-4, n_modes + 3)
    ax_spec.set_ylim(-1.15 * mu_max, 1.15 * mu_max)
    ax_spec.set_xticks([0, n_modes - 1])
    ax_spec.set_yticks([-round(mu_max, 2), 0.0, round(mu_max, 2)])
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.spines["left"].set_color("#52616a")
    ax_spec.spines["bottom"].set_color("#52616a")
    ax_spec.tick_params(colors="#52616a")

    for ax, img in ((ax_taylor, taylor_img), (ax_excited, excited_img)):
        show_pixels(ax, img)
    return figure


def animate_relaxation(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                       volumes: core.Volume, magnetic_frames: Iterable[core.Field]) -> Iterator[np.ndarray]:
    """Oblique and axial 3D views of magnetic field lines relaxing toward the coherent Taylor state."""
    oblique_view = solids.View(solids.aim(solids.origin, torus_elevation, torus_azimuth), np.full(2, 1.38), n_anim_pixels)
    axial_view = solids.View(solids.aim(solids.origin, top_elevation, top_azimuth), np.full(2, 1.38), n_anim_pixels)
    separator = np.full((n_anim_pixels, 8, 3), 255, dtype=np.uint8)
    for magnetic in magnetic_frames:
        lines, shells = field_lines(centres, volumes, magnetic)
        left = render_torus_field(mesh, vertices, lines, shells, oblique_view)
        right = render_torus_field(mesh, vertices, lines, shells, axial_view)
        yield np.concatenate((left, separator, right), axis=1)
