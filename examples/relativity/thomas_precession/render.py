"""Drawing for Thomas precession and Bargmann–Michel–Telegdi spin dynamics: four-velocity loops on the
unit hyperboloid, concentric circular orbits showing the accumulated Thomas holonomy at three Lorentz
factors, and storage-ring spin precession for three gyromagnetic ratios."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.relativity.thomas_precession import core

COLOURS = ("#2a9d8f", "#e76f51", "#1d3557")
RING_RADII = (0.68, 1.0, 1.32)


def spacetime_coords(vectors: core.Vector) -> np.ndarray:
    """The `(t, x, y)` components of spacetime vectors."""
    return vectors.cast(core.ga.subspace("t x y")).kernel


def planar_coords(vectors: core.Vector) -> np.ndarray:
    """The `(x, y)` components of spatial vectors."""
    return vectors.cast(core.ga.subspace("x y")).kernel


def _draw_concentric_orbits(ax, base_xy: np.ndarray, spins_xy: np.ndarray, radii: tuple[float, ...], step_limit: int) -> None:
    """Draw three concentric circular tracks with sampled laboratory spin arrows up to `step_limit`."""
    total_steps = len(base_xy) - 1
    stride = max(1, total_steps // 18)
    sample_indices = [idx for idx in range(0, step_limit, stride)]
    for case, (radius, colour) in enumerate(zip(radii, COLOURS)):
        track = radius * base_xy
        ax.plot(track[:, 0], track[:, 1], color="#cbd5e1", linewidth=1.1, zorder=1)
        ax.plot(track[:step_limit, 0], track[:step_limit, 1], color=colour, linewidth=1.8, alpha=0.65, zorder=2)
        if sample_indices:
            pts = track[sample_indices]
            vecs = 0.22 * spins_xy[sample_indices, case]
            ax.quiver(
                pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1],
                angles="xy", scale_units="xy", scale=1.0,
                color=colour, alpha=0.45, width=0.005, zorder=3,
            )
        # Highlight initial spin and current/final spin.
        start_pt = track[0]
        start_vec = 0.26 * spins_xy[0, case]
        ax.quiver(
            [start_pt[0]], [start_pt[1]], [start_vec[0]], [start_vec[1]],
            angles="xy", scale_units="xy", scale=1.0,
            color="#64748b", width=0.007, zorder=4,
        )
        end_idx = step_limit - 1
        end_pt = track[end_idx]
        end_vec = 0.26 * spins_xy[end_idx, case]
        ax.quiver(
            [end_pt[0]], [end_pt[1]], [end_vec[0]], [end_vec[1]],
            angles="xy", scale_units="xy", scale=1.0,
            color=colour, width=0.008, zorder=5,
        )
        ax.scatter([end_pt[0]], [end_pt[1]], s=32, color=colour, edgecolors="white", linewidths=0.8, zorder=6)


def draw_thomas_summary(
    loop_pos: core.Vector,
    loop_u: core.Vector,
    loop_lab: core.Vector,
    ring_pos: core.Vector,
    ring_lab: core.Vector,
) -> plt.Figure:
    """Left: circular four-velocity orbits on the future unit hyperboloid; middle: concentric orbits
    showing the laboratory rest-frame spin and its Thomas holonomy gap at three Lorentz factors; right:
    storage-ring spin angle relative to the velocity tangent across one cyclotron period for three
    gyromagnetic ratios."""
    figure = plt.figure(figsize=(12.2, 4.0), dpi=100)
    ax_hyp = figure.add_subplot(1, 3, 1, projection="3d")
    ax_loop = figure.add_subplot(1, 3, 2)
    ax_ring = figure.add_subplot(1, 3, 3)

    # Unit hyperboloid sheet u_t = sqrt(1 + u_x^2 + u_y^2).
    rho = np.linspace(0.0, 1.85, 28)
    phi = np.linspace(0.0, 2 * np.pi, 60)
    r_grid, p_grid = np.meshgrid(rho, phi, indexing="ij")
    ux_surf = r_grid * np.cos(p_grid)
    uy_surf = r_grid * np.sin(p_grid)
    ut_surf = np.sqrt(1.0 + r_grid**2)
    ax_hyp.plot_surface(ux_surf, uy_surf, ut_surf, color="#e2e8f0", alpha=0.45, edgecolor="none", shade=True)

    u_txy = spacetime_coords(loop_u)                                            # [steps + 1, orbits, 3]
    for case, colour in enumerate(COLOURS):
        ax_hyp.plot(
            u_txy[:, case, 1], u_txy[:, case, 2], u_txy[:, case, 0],
            color=colour, linewidth=2.2,
        )
    ax_hyp.view_init(elev=24, azim=-52)
    ax_hyp.set_box_aspect((1.0, 1.0, 0.85))
    ax_hyp.set_axis_off()

    base_xy = planar_coords(loop_pos)[:, 0]                                     # [steps + 1, 2]
    loop_spins_xy = planar_coords(loop_lab)                                     # [steps + 1, orbits, 2]
    _draw_concentric_orbits(ax_loop, base_xy, loop_spins_xy, RING_RADII, len(base_xy))
    ax_loop.set_xlim(-1.7, 1.7)
    ax_loop.set_ylim(-1.7, 1.7)
    ax_loop.set_aspect("equal")
    ax_loop.set_axis_off()

    ring_base_xy = planar_coords(ring_pos)[:, 0]                                # [steps + 1, 2]
    ring_spins_xy = planar_coords(ring_lab)                                     # [steps + 1, cases, 2]
    # Unit tangent along the counterclockwise orbit is (-y, x).
    tangent_xy = np.stack([-ring_base_xy[:, 1], ring_base_xy[:, 0]], axis=-1)   # [steps + 1, 2]
    normal_xy = ring_base_xy                                                    # [steps + 1, 2]
    cos_rel = np.einsum("sc,s->sc", ring_spins_xy[:, :, 0], tangent_xy[:, 0]) + np.einsum("sc,s->sc", ring_spins_xy[:, :, 1], tangent_xy[:, 1])
    sin_rel = np.einsum("sc,s->sc", ring_spins_xy[:, :, 0], normal_xy[:, 0]) + np.einsum("sc,s->sc", ring_spins_xy[:, :, 1], normal_xy[:, 1])
    rel_angle = np.unwrap(np.arctan2(sin_rel, cos_rel), axis=0)                 # [steps + 1, cases]
    progress = np.linspace(0.0, 1.0, len(ring_base_xy))
    ax_ring.axhline(0.0, color="#cbd5e1", linewidth=1.0, zorder=0)
    for case, colour in enumerate(COLOURS):
        ax_ring.plot(progress, rel_angle[:, case], color=colour, linewidth=2.3)
    ax_ring.set_axis_off()

    figure.tight_layout()
    return figure


def animate_storage_ring(ring_pos: core.Vector, ring_lab: core.Vector, every: int) -> list[np.ndarray]:
    """Frames of the three particles (`g = 0`, `g = 2`, `g = 2.5`) orbiting the magnetic storage ring
    with their laboratory rest-frame spins."""
    base_xy = planar_coords(ring_pos)[:, 0]                                     # [steps + 1, 2]
    spins_xy = planar_coords(ring_lab)                                          # [steps + 1, cases, 2]
    images = []
    for step_end in range(1, len(base_xy) + 1, every):
        figure, ax = plt.subplots(figsize=(5.2, 5.2), dpi=85)
        _draw_concentric_orbits(ax, base_xy, spins_xy, RING_RADII, step_end)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.set_axis_off()
        figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        images.append(capture(figure))
        plt.close(figure)
    return images
