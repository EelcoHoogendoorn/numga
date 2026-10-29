"""Tetrahedral forms and smoothed field-line tubes show magnetic linkage and selective decay."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.electromagnetism.beltrami_fields import core
from examples.util import charts
from examples.util.wgsl import meshes, show_pixels, solids, strokes

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


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Cartesian coordinates of intrinsic 3D positions and directions by pairing with the frame planes."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y, mv.z)) & points.batch()[..., None]).to_array())


def outward_wall(mesh: core.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The faces on the wall of a solid mesh, `[wall]`, and their corners `[wall, 3]`, wound outward. The wall is the
    boundary of the solid, the sum of its tetrahedra's outward faces: one on a wall face that runs outward as stored,
    minus one on a wall face that runs inward, nothing inside."""
    wall = mesh.exterior(2)
    sense = (~mesh.d[2] * core.as_scalar(np.ones(mesh.n_elements[3])).field()).to_array()[wall]
    return wall, np.where(sense[:, None] > 0, mesh.faces[wall], mesh.faces[wall][:, ::-1])


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
    _, b_faces = outward_wall(mesh)
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


@charts.style()
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
        left = render_torus_field(mesh, vertices, centres, volumes, magnetic, oblique_view)
        right = render_torus_field(mesh, vertices, centres, volumes, magnetic, axial_view)
        yield np.concatenate((left, separator, right), axis=1)
