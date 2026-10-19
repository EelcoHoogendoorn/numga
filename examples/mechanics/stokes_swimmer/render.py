"""Drawing for low-Reynolds-number swimming in PGA2D: the Purcell swimmer and its surrounding Stokes
flow field beside the reciprocal flapper and their paths in hinge-angle space."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.mechanics.stokes_swimmer import core

PURCELL_COLOUR = "#1d3557"
SCALLOP_COLOUR = "#c84b31"
TRAIL_COLOUR = "#e63946"
FLOW_COLOUR = "#94a3b8"
FORCE_COLOUR = "#2a9d8f"


def point_coords(points: core.Point | core.Direction) -> np.ndarray:
    """The `(x, y)` coordinates of planar points or ideal displacements."""
    return points.dual().cast(core.ga.subspace("x y")).kernel


def force_coords(forces: core.Force) -> np.ndarray:
    """The `(x, y)` components of planar forces."""
    return forces.cast(core.ga.subspace("x y")).kernel


def _draw_swimmer(ax, beads_xy: np.ndarray, colour: str, alpha: float) -> None:
    """Draw the three-link body and its nine hydrodynamic beads."""
    # Left arm (beads 0, 1, 2), centre bar (beads 3, 4, 5), right arm (beads 6, 7, 8).
    left_hinge = 1.5 * beads_xy[3] - 0.5 * beads_xy[4]
    right_hinge = 1.5 * beads_xy[5] - 0.5 * beads_xy[4]
    chain = np.vstack([beads_xy[[0, 1, 2]], left_hinge, beads_xy[[3, 4, 5]], right_hinge, beads_xy[[6, 7, 8]]])
    ax.plot(chain[:, 0], chain[:, 1], color=colour, linewidth=2.6, alpha=alpha, zorder=3)
    ax.scatter(beads_xy[:, 0], beads_xy[:, 1], s=42, color=colour, alpha=alpha, zorder=4, edgecolors="white", linewidths=0.6)
    ax.scatter(
        [left_hinge[0], right_hinge[0]],
        [left_hinge[1], right_hinge[1]],
        s=26,
        color="white",
        edgecolors=colour,
        linewidths=1.4,
        zorder=5,
        alpha=alpha,
    )


def draw_summary(
    angles: np.ndarray,
    positions: core.Point,
    world_sites: core.Point,
    world_forces: core.Force,
    grid: core.Point,
    flow: core.Direction,
) -> plt.Figure:
    """Left: instantaneous Stokes flow around Purcell's swimmer with bead forces and centre trail;
    middle: centre displacement along x over the three cycles for both strokes; right: the two strokes
    in hinge-angle space."""
    figure, (ax_flow, ax_path, ax_hinge) = plt.subplots(1, 3, figsize=(12.2, 3.8), dpi=100)

    grid_xy = point_coords(grid)                                                # [gy, gx, 2]
    flow_xy = point_coords(flow)                                                # [gy, gx, 2]
    beads_xy = point_coords(world_sites.batch())                                # [steps, swimmers, beads, 2]
    forces_xy = force_coords(world_forces.batch())                              # [steps, swimmers, beads, 2]
    pos_xy = point_coords(positions)                                            # [steps + 1, swimmers, 2]

    ax_flow.quiver(
        grid_xy[..., 0], grid_xy[..., 1], flow_xy[..., 0], flow_xy[..., 1],
        color=FLOW_COLOUR, pivot="mid", width=0.004, zorder=1,
    )
    _draw_swimmer(ax_flow, beads_xy[0, 0], PURCELL_COLOUR, 1.0)
    ax_flow.quiver(
        beads_xy[0, 0, :, 0], beads_xy[0, 0, :, 1],
        forces_xy[0, 0, :, 0], forces_xy[0, 0, :, 1],
        color=FORCE_COLOUR, scale=1.4, width=0.007, zorder=6,
    )
    ax_flow.plot(pos_xy[:, 0, 0], pos_xy[:, 0, 1], color=TRAIL_COLOUR, linewidth=1.6, zorder=2)
    ax_flow.set_xlim(grid_xy[0, 0, 0], grid_xy[0, -1, 0])
    ax_flow.set_ylim(grid_xy[0, 0, 1], grid_xy[-1, 0, 1])
    ax_flow.set_aspect("equal")
    ax_flow.set_axis_off()

    t = np.linspace(0.0, 1.0, len(pos_xy))
    ax_path.plot(t, pos_xy[:, 0, 0], color=PURCELL_COLOUR, linewidth=2.2)
    ax_path.plot(t, pos_xy[:, 1, 0], color=SCALLOP_COLOUR, linewidth=2.0, linestyle="--")
    ax_path.axhline(0.0, color="#cbd5e1", linewidth=0.8, zorder=0)
    ax_path.set_axis_off()

    ax_hinge.plot(angles[:, 0, 0], angles[:, 0, 1], color=PURCELL_COLOUR, linewidth=2.2)
    ax_hinge.plot(angles[:, 1, 0], angles[:, 1, 1], color=SCALLOP_COLOUR, linewidth=2.0, linestyle="--")
    ax_hinge.set_aspect("equal")
    ax_hinge.set_axis_off()

    figure.tight_layout()
    return figure


def animate_swimmers(
    positions: core.Point,
    world_sites: core.Point,
    grid: core.Point,
    flows: Iterable[core.Direction],
    every: int,
) -> list[np.ndarray]:
    """Frames of Purcell's non-reciprocal swimmer (top), with the flow it drives at each frame
    `[grid_y, grid_x] Direction`, and the reciprocal flapper (bottom)."""
    beads_xy = point_coords(world_sites.batch())[::every]                       # [frames, swimmers, beads, 2]
    all_pos_xy = point_coords(positions)                                        # [steps + 1, swimmers, 2]
    grid_xy = point_coords(grid)                                                # [gy, gx, 2]

    x_min = min(float(all_pos_xy[:, 0, 0].min()), 0.0) - 2.3
    x_max = max(float(all_pos_xy[:, 0, 0].max()), 0.0) + 2.3
    images = []
    for frame, flow in enumerate(flows):
        flow_xy = point_coords(flow)                                            # [gy, gx, 2]
        figure, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.2, 4.6), dpi=85)
        step_end = frame * every + 1

        ax_top.quiver(
            grid_xy[..., 0], grid_xy[..., 1],
            flow_xy[..., 0], flow_xy[..., 1],
            color=FLOW_COLOUR, scale=3.2, width=0.003, zorder=1,
        )
        ax_top.plot(all_pos_xy[:step_end, 0, 0], all_pos_xy[:step_end, 0, 1], color=TRAIL_COLOUR, linewidth=1.6, zorder=2)
        _draw_swimmer(ax_top, beads_xy[frame, 0], PURCELL_COLOUR, 1.0)
        ax_top.set_xlim(x_min, x_max)
        ax_top.set_ylim(-1.55, 1.55)
        ax_top.set_aspect("equal")
        ax_top.set_axis_off()

        ax_bot.plot(all_pos_xy[:step_end, 1, 0], all_pos_xy[:step_end, 1, 1], color=TRAIL_COLOUR, linewidth=1.6, zorder=2)
        _draw_swimmer(ax_bot, beads_xy[frame, 1], SCALLOP_COLOUR, 1.0)
        ax_bot.set_xlim(x_min, x_max)
        ax_bot.set_ylim(-1.55, 1.55)
        ax_bot.set_aspect("equal")
        ax_bot.set_axis_off()

        figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, hspace=0.06)
        images.append(capture(figure))
        plt.close(figure)
    return images
