"""Tetrahedral forms and smoothed field-line tubes show magnetic linkage and selective decay."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.electromagnetism.beltrami_fields import core
from examples.util.wgsl import image_figure, meshes, show_pixels, solids, strokes

n_tet_pixels = 300
n_torus_pixels = 320
n_anim_pixels = 270
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


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Cartesian coordinates of intrinsic 3D positions and directions by pairing with the frame planes."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y, mv.z)) & points.batch()[..., None]).to_array())


def draw_tet_whitney(corners: core.Point, points: core.Point, edge_forms: core.Plane,
                     face_forms: core.Field, volume: core.Volume) -> plt.Figure:
    """Three 3D views of a single tetrahedron: edge 1-form, face 2-form, and opposite-edge helicity linkage."""
    corner_xyz = coordinates(corners)
    centre_xyz = corner_xyz.mean(axis=0)
    corner_xyz = corner_xyz - centre_xyz
    # Spread the glyphs around the centroid so their directions remain legible.
    sample_xyz = (coordinates(points) - centre_xyz) * 1.22

    edge_field_xyz = coordinates(core.Plane.dual()(edge_forms[0]))
    unit_volume = volume / (volume & core.time).norm()
    face_field_xyz = coordinates(face_forms[0] * unit_volume)

    tet_pts = solids.mv.point(corner_xyz)
    tet_faces = np.stack([core.face_first, core.face_second, core.face_third], axis=-1)
    view = solids.View(solids.aim(solids.origin, tet_elevation, tet_azimuth), np.full(2, 0.86), n_tet_pixels)

    shell_colours = np.broadcast_to(cream, tet_faces.shape + (3,))
    shell = meshes.samples(tet_pts, tet_faces, shell_colours, meshes.flat(tet_pts, tet_faces), view,
                           face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    wire = strokes.segments(tet_pts[core.edge_tails], tet_pts[core.edge_heads], light_slate, 1.5, view)

    # Panel 1: the Whitney 1-form of edge (0, 1), circulating around the opposite edge (2, 3).
    edge_scale = 0.30 / np.linalg.norm(edge_field_xyz, axis=-1).max()
    e_starts = solids.mv.point(sample_xyz - 0.45 * edge_scale * edge_field_xyz)
    e_ends = solids.mv.point(sample_xyz + 0.55 * edge_scale * edge_field_xyz)
    e_shafts, e_heads = solids.arrows(e_starts, e_ends, 0.017, 0.065, 0.038)
    active_rod, active_balls = solids.edges(tet_pts[[0]], tet_pts[[1]], 0.024)
    opp_rod, opp_balls = solids.edges(tet_pts[[2]], tet_pts[[3]], 0.020)
    mid_01 = (corner_xyz[0] + corner_xyz[1]) / 2
    dir_01 = corner_xyz[1] - corner_xyz[0]
    edge_arrow_s, edge_arrow_h = solids.arrows(
        solids.mv.point((mid_01 - 0.18 * dir_01)[None]),
        solids.mv.point((mid_01 + 0.22 * dir_01)[None]),
        0.025, 0.090, 0.054,
    )
    panel_edge = solids.draw(
        [(e_shafts, vermilion), (e_heads, vermilion), (active_rod, vermilion), (active_balls, vermilion),
         (edge_arrow_s, vermilion), (edge_arrow_h, vermilion), (opp_rod, sapphire), (opp_balls, sapphire)],
        view, background, inks=[], translucent=[], meshes=[(shell, 0.24), (wire, 1.0)],
    )

    # Panel 2: the Whitney 2-form of face (1, 2, 3), radiating from the opposite corner 0 through that face.
    face_scale = 0.30 / np.linalg.norm(face_field_xyz, axis=-1).max()
    f_starts = solids.mv.point(sample_xyz - 0.45 * face_scale * face_field_xyz)
    f_ends = solids.mv.point(sample_xyz + 0.55 * face_scale * face_field_xyz)
    f_shafts, f_heads = solids.arrows(f_starts, f_ends, 0.017, 0.065, 0.038)
    face_colours = np.broadcast_to(cream, tet_faces.shape + (3,)).copy()
    face_colours[0] = warm_tint
    face_mesh = meshes.samples(tet_pts, tet_faces, face_colours, meshes.flat(tet_pts, tet_faces), view,
                               face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    apex_ball = solids.ball(tet_pts[[0]], 0.046)[:, None]
    face_rim = strokes.segments(tet_pts[[1, 2, 3]], tet_pts[[2, 3, 1]], vermilion, 2.8, view)
    panel_face = solids.draw(
        [(f_shafts, sapphire), (f_heads, sapphire), (apex_ball, vermilion)],
        view, background, inks=[], translucent=[], meshes=[(face_mesh, 0.30), (wire, 1.0), (face_rim, 1.0)],
    )

    # Panel 3: Topological helicity linkage between opposite edges (0, 1) and (2, 3).
    mid_23 = (corner_xyz[2] + corner_xyz[3]) / 2
    dir_23 = corner_xyz[3] - corner_xyz[2]
    axis_23 = dir_23 / np.linalg.norm(dir_23)
    opp_arrow_s, opp_arrow_h = solids.arrows(
        solids.mv.point((mid_23 - 0.18 * dir_23)[None]),
        solids.mv.point((mid_23 + 0.22 * dir_23)[None]),
        0.025, 0.090, 0.054,
    )
    radial_1 = mid_01 - mid_23
    radial_1 = radial_1 - np.dot(radial_1, axis_23) * axis_23
    radial_1 = radial_1 / np.linalg.norm(radial_1)
    radial_2 = np.cross(axis_23, radial_1)
    ring_angles = np.linspace(0.0, 2 * np.pi, 65)
    curves = []
    for radius, colour in ((0.20, gold), (0.31, vermilion), (0.42, teal)):
        ring_xyz = (mid_23[None, :] * 0.62
                    + radius * (np.cos(ring_angles)[:, None] * radial_1[None, :]
                                + np.sin(ring_angles)[:, None] * radial_2[None, :]))
        curves.append((solids.mv.point(ring_xyz), colour))
    rings = strokes.polylines(curves, 3.2, view)
    panel_link = solids.draw(
        [(active_rod, vermilion), (active_balls, vermilion), (edge_arrow_s, vermilion), (edge_arrow_h, vermilion),
         (opp_rod, sapphire), (opp_balls, sapphire), (opp_arrow_s, sapphire), (opp_arrow_h, sapphire)],
        view, background, inks=[], translucent=[], meshes=[(shell, 0.22), (wire, 1.0), (rings, 1.0)],
    )

    separator = np.full((n_tet_pixels, 10, 3), 255, dtype=np.uint8)
    combined = np.concatenate((panel_edge, separator, panel_face, separator, panel_link), axis=1)
    return image_figure(combined)


def trace_torus_streamlines(centre_xyz: np.ndarray, field_xyz: np.ndarray,
                            major_radius: float = 1.0, minor_radius: float = 0.33
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Smooth, strictly confined 3D magnetic field-line trajectories inside the torus, traced by RK4."""
    seeds = []
    shell_ids = []
    for shell_idx, (r_frac, n_seeds, pol_angle, phi_offset) in enumerate((
        (0.24, 3, 0.20, 0.0),
        (0.52, 4, 1.10, np.pi / 4),
        (0.78, 5, 2.15, np.pi / 5),
    )):
        tor_angles = np.linspace(0.0, 2 * np.pi, n_seeds, endpoint=False) + phi_offset
        pol_angles = pol_angle + np.arange(n_seeds) * (2 * np.pi / n_seeds)
        r = r_frac * minor_radius
        R = major_radius + r * np.cos(pol_angles)
        seeds.append(np.stack((R * np.cos(tor_angles), R * np.sin(tor_angles), r * np.sin(pol_angles)), axis=-1))
        shell_ids.append(np.full(n_seeds, shell_idx))
    x0 = np.concatenate(seeds, axis=0)                                        # [lines, 3]
    shells = np.concatenate(shell_ids, axis=0)                                # [lines]

    # Symmetrize the two outer tetrahedra of each 3-tet prism wedge to remove even/odd diagonal split alternation.
    c3 = centre_xyz.reshape(3, -1, 3)
    f3 = field_xyz.reshape(3, -1, 3)
    sym_xyz = np.concatenate((0.5 * (c3[0] + c3[1]), c3[2]), axis=0)
    sym_field = np.concatenate((0.5 * (f3[0] + f3[1]), f3[2]), axis=0)

    c_R = np.maximum(np.hypot(sym_xyz[:, 0], sym_xyz[:, 1]), 1e-8)
    c_phi = np.arctan2(sym_xyz[:, 1], sym_xyz[:, 0])
    c_z = sym_xyz[:, 2]
    c_cos, c_sin = sym_xyz[:, 0] / c_R, sym_xyz[:, 1] / c_R
    field_cyl = np.stack((
        sym_field[:, 0] * c_cos + sym_field[:, 1] * c_sin,
        -sym_field[:, 0] * c_sin + sym_field[:, 1] * c_cos,
        sym_field[:, 2],
    ), axis=-1)                                                               # [samples, 3]
    b_scale = float(np.sqrt(np.mean(np.sum(field_cyl * field_cyl, axis=-1))))

    def velocity(pts: np.ndarray, sign: float) -> np.ndarray:
        R = np.maximum(np.hypot(pts[:, 0], pts[:, 1]), 1e-8)
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        z = pts[:, 2]
        cos_phi, sin_phi = pts[:, 0] / R, pts[:, 1] / R
        d_phi = 2.0 * np.sin((phi[:, None] - c_phi[None, :]) / 2.0)
        d_R = R[:, None] - c_R[None, :]
        d_z = z[:, None] - c_z[None, :]
        weights = np.exp(-(d_phi / 0.58) ** 2 - (d_R ** 2 + d_z ** 2) / (0.20 ** 2))
        b_cyl = sign * (weights @ field_cyl) / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)

        rho_r, rho_z = R - major_radius, z
        r = np.maximum(np.hypot(rho_r, rho_z), 1e-8)
        cos_theta, sin_theta = rho_r / r, rho_z / r
        b_rad = b_cyl[:, 0] * cos_theta + b_cyl[:, 2] * sin_theta
        b_pol = -b_cyl[:, 0] * sin_theta + b_cyl[:, 2] * cos_theta

        # Smoothly taper radial motion to zero toward the conducting wall so streamlines glide along flux surfaces.
        s = np.clip(r / minor_radius, 0.0, 1.0)
        b_rad = b_rad * np.cos(0.5 * np.pi * s) ** 2 - 0.08 * (s ** 3) * b_scale
        b_r = b_rad * cos_theta - b_pol * sin_theta
        b_phi = b_cyl[:, 1]
        b_z = b_rad * sin_theta + b_pol * cos_theta
        b = np.stack((b_r * cos_phi - b_phi * sin_phi, b_r * sin_phi + b_phi * cos_phi, b_z), axis=-1)
        norm = np.sqrt(np.sum(b * b, axis=-1, keepdims=True) + (0.14 * b_scale) ** 2)
        return b / norm

    n_half_steps = 170
    ds = 0.018

    def integrate(sign: float) -> np.ndarray:
        pts = x0.copy()
        traj = [pts.copy()]
        for step in range(1, n_half_steps + 1):
            k1 = velocity(pts, sign)
            k2 = velocity(pts + 0.5 * ds * k1, sign)
            k3 = velocity(pts + 0.5 * ds * k2, sign)
            k4 = velocity(pts + ds * k3, sign)
            pts = pts + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if step % 2 == 0:
                traj.append(pts.copy())
        return np.stack(traj, axis=1)                                         # [lines, steps, 3]

    backward = integrate(-1.0)[:, :0:-1, :]
    forward = integrate(1.0)
    trajectories = np.concatenate((backward, forward), axis=1)                # [lines, total_steps, 3]
    return trajectories, shells


def palette_blend(twist: np.ndarray, shell: np.ndarray) -> np.ndarray:
    """Blend a rich scientific palette by radial flux shell and local helical pitch."""
    base_colours = np.array([vermilion, teal, sapphire])[shell]               # [lines, 3]
    # Modulate toward warm gold where the field line pitches strongly upward/poloidally.
    pitch_weight = np.clip(0.5 + 0.45 * twist, 0.0, 1.0)[..., None]
    blended = (1.0 - 0.38 * pitch_weight) * base_colours[:, None, :] + (0.38 * pitch_weight) * gold[None, None, :]
    return np.clip(blended, 0.0, 1.0)


def render_torus_field(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                       volumes: core.Volume, magnetic: core.Field, view: solids.View) -> np.ndarray:
    """Continuous 3D magnetic field-line tubes inside a smooth translucent toroidal cavity."""
    vert_xyz = coordinates(vertices)
    centre_xyz = coordinates(centres)
    field_xyz = coordinates(magnetic * (volumes / (volumes & core.time).norm()))

    torus_pts = solids.mv.point(vert_xyz)
    # The wall's triangles, wound outward as faces of the tetrahedra they bound.
    tet_faces = np.stack([mesh.tets[:, core.face_first], mesh.tets[:, core.face_second], mesh.tets[:, core.face_third]], axis=-1)
    b_faces = tet_faces[np.isin(mesh.boundary_indices[2], mesh.exterior(2))]
    shell_colours = np.broadcast_to(cream, b_faces.shape + (3,))
    shell = meshes.samples(torus_pts, b_faces, shell_colours, meshes.smooth(torus_pts, b_faces), view,
                           face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)

    trajectories, shells = trace_torus_streamlines(centre_xyz, field_xyz)
    starts_xyz = trajectories[:, :-1, :]                                      # [lines, segs, 3]
    ends_xyz = trajectories[:, 1:, :]                                         # [lines, segs, 3]
    tangents = ends_xyz - starts_xyz
    tangents = tangents / np.maximum(np.linalg.norm(tangents, axis=-1, keepdims=True), 1e-12)

    # Local helical twist / vertical pitch component along each segment.
    seg_colours = palette_blend(tangents[..., 2], shells)                     # [lines, segs, 3]
    tubes = strokes.segments(
        solids.mv.point(starts_xyz.reshape(-1, 3)),
        solids.mv.point(ends_xyz.reshape(-1, 3)),
        seg_colours.reshape(-1, 3),
        4.6,
        view,
    )

    # Place glossy 3D cone arrowheads along each field line to show its flow direction.
    mid_idx = trajectories.shape[1] // 2
    arrow_starts = trajectories[:, mid_idx - 2, :]
    arrow_ends = trajectories[:, mid_idx + 2, :]
    arrow_dirs = arrow_ends - arrow_starts
    arrow_dirs = arrow_dirs / np.maximum(np.linalg.norm(arrow_dirs, axis=-1, keepdims=True), 1e-12)
    a_starts = solids.mv.point(arrow_starts)
    a_ends = solids.mv.point(arrow_starts + 0.14 * arrow_dirs)
    _, a_heads = solids.arrows(a_starts, a_ends, 0.018, 0.082, 0.044)
    head_colours = np.array([vermilion, teal, sapphire])[shells]

    return solids.draw([(a_heads, head_colours)], view, background, inks=[], translucent=[],
                       meshes=[(shell, 0.16), (tubes, 1.0)])


def draw_beltrami_eigenmodes(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                             volumes: core.Volume, inverse_eigenvalues: core.Scalar, mode_magnetic: core.Field,
                             shown_indices: np.ndarray) -> plt.Figure:
    """The chiral Beltrami spectrum beside 3D field-line renders of the Taylor state and a higher-order mode."""
    mu = np.asarray(inverse_eigenvalues.to_array())
    n_modes = len(mu)
    indices = np.arange(n_modes)

    view = solids.View(solids.aim(solids.origin, torus_elevation, torus_azimuth), np.full(2, 1.38), n_torus_pixels)
    taylor_img = render_torus_field(mesh, vertices, centres, volumes, mode_magnetic[0], view)
    excited_img = render_torus_field(mesh, vertices, centres, volumes, mode_magnetic[1], view)

    dpi = 100
    figure = plt.figure(figsize=(3 * n_torus_pixels / dpi, n_torus_pixels / dpi), dpi=dpi)
    ax_spec = figure.add_axes([0.055, 0.14, 0.255, 0.78])
    ax_taylor = figure.add_axes([0.335, 0.0, 0.33, 1.0])
    ax_excited = figure.add_axes([0.67, 0.0, 0.33, 1.0])

    ax_spec.axhline(0.0, color="#6c7a82", linewidth=0.9, zorder=1)
    negative = mu < 0
    positive = mu > 0
    ax_spec.scatter(indices[negative], mu[negative], s=14, color=sapphire, alpha=0.82, zorder=2)
    ax_spec.scatter(indices[positive], mu[positive], s=14, color=vermilion, alpha=0.82, zorder=2)
    resolved = shown_indices % n_modes
    ax_spec.scatter(resolved, mu[resolved], s=52, color=[vermilion, teal],
                    edgecolor="white", linewidth=1.0, zorder=4)
    mu_max = float(np.abs(mu).max())
    ax_spec.set_xlim(-4, n_modes + 3)
    ax_spec.set_ylim(-1.15 * mu_max, 1.15 * mu_max)
    ax_spec.set_xticks([0, n_modes - 1])
    ax_spec.set_yticks([-round(mu_max, 2), 0.0, round(mu_max, 2)])
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.spines["left"].set_color("#52616a")
    ax_spec.spines["bottom"].set_color("#52616a")
    ax_spec.tick_params(colors="#52616a", labelsize=8)

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
        left = render_torus_field(mesh, vertices, centres, volumes, magnetic, oblique_view)
        right = render_torus_field(mesh, vertices, centres, volumes, magnetic, axial_view)
        yield np.concatenate((left, separator, right), axis=1)
