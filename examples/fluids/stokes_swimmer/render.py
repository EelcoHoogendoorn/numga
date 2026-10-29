"""Drawing for low-Reynolds-number swimming in PGA2D: the Purcell swimmer and its surrounding Stokes
flow field beside the reciprocal flapper and their paths in hinge-angle space."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.fluids.stokes_swimmer import core

PURCELL_COLOUR = "#1d3557"
SCALLOP_COLOUR = "#c84b31"
TRAIL_COLOUR = "#e63946"
FLOW_COLOUR = "#94a3b8"
FORCE_COLOUR = "#2a9d8f"
CROSS_COLOUR, LABEL_COLOUR = "#cbd5e1", "0.4"
# The margin around the strokes in hinge-angle space, as a share of their extent, across and up: up, the room for the
# name running down the line.
HINGE_MARGIN, HINGE_ROOM = 0.12, 0.5


def point_coords(points: core.Point) -> np.ndarray:
    """The `(x, y)` coordinates of points at unit weight, or of ideal displacements: their pairings with the
    planes x = 0 and y = 0."""
    mv = core.mv
    return (stack((mv.x, mv.y)) & points[..., None]).to_array()


def force_coords(forces: core.Force) -> np.ndarray:
    """The `(x, y)` components of forces: their pairings with the unit displacements along x and y."""
    mv = core.mv
    return (forces[..., None] & stack((mv.direction([1.0, 0.0]), mv.direction([0.0, 1.0])))).to_array()


def _cross(ax, across: str, up: str) -> None:
    """No scales: a faint cross through the origin, its lines named at their ends, each along itself,
    the line across named just past its end."""
    ax.axhline(0.0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.axvline(0.0, color=CROSS_COLOUR, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.text(1, 0, f" {across}", transform=ax.get_yaxis_transform(), ha="left", va="center", color=LABEL_COLOUR)
    ax.text(0, 1, f"{up} ", transform=ax.get_xaxis_transform(), ha="right", va="bottom", color=LABEL_COLOUR,
            rotation=90, rotation_mode="anchor")
    ax.set_axis_off()


def _zero_axes(ax, across: str, up: str) -> None:
    """Axes of a quantity through its zero and the parameter's, named at their ends, each along itself, without tick
    marks or numbers, the axis across named just past its end."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.set(xticks=[], yticks=[])
    ax.text(1, 0, f" {across}", transform=ax.get_yaxis_transform(), ha="left", va="center", color=LABEL_COLOUR)
    ax.text(0, 1, f"{up} ", transform=ax.get_xaxis_transform(), ha="right", va="bottom", color=LABEL_COLOUR,
            rotation=90, rotation_mode="anchor")


def _links(beads_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The three-link body through its nine beads `[9, 2]`: the chain through the beads and the hinges `[11, 2]`, and
    the two hinges `[2, 2]`."""
    # Left arm (beads 0, 1, 2), centre bar (beads 3, 4, 5), right arm (beads 6, 7, 8).
    left_hinge = 1.5 * beads_xy[3] - 0.5 * beads_xy[4]
    right_hinge = 1.5 * beads_xy[5] - 0.5 * beads_xy[4]
    chain = np.vstack([beads_xy[[0, 1, 2]], left_hinge, beads_xy[[3, 4, 5]], right_hinge, beads_xy[[6, 7, 8]]])
    return chain, np.stack([left_hinge, right_hinge])


def _draw_swimmer(ax, beads_xy: np.ndarray, colour: str, alpha: float) -> list[Artist]:
    """Draw the three-link body and its nine hydrodynamic beads: the chain, the beads and the hinges."""
    chain, hinges = _links(beads_xy)
    line, = ax.plot(chain[:, 0], chain[:, 1], color=colour, linewidth=2.6, alpha=alpha, zorder=3)
    beads = ax.scatter(beads_xy[:, 0], beads_xy[:, 1], s=42, color=colour, alpha=alpha, zorder=4, edgecolors="white",
                       linewidths=0.6)
    joints = ax.scatter(
        hinges[:, 0],
        hinges[:, 1],
        s=26,
        color="white",
        edgecolors=colour,
        linewidths=1.4,
        zorder=5,
        alpha=alpha,
    )
    return [line, beads, joints]


def _move_swimmer(parts: list[Artist], beads_xy: np.ndarray) -> None:
    """Move a swimmer drawn by `_draw_swimmer` to its beads."""
    line, beads, joints = parts
    chain, hinges = _links(beads_xy)
    line.set_data(chain[:, 0], chain[:, 1])
    beads.set_offsets(beads_xy)
    joints.set_offsets(hinges)


@charts.style()
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
    # The whole column: three panels side by side, the flow's arrows dense.
    figure, (ax_flow, ax_path, ax_hinge) = charts.subplots(800, 260, ncols=3)

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
    ax_flow.plot(pos_xy[:, 0, 0], pos_xy[:, 0, 1], color=TRAIL_COLOUR, zorder=2)
    ax_flow.set_xlim(grid_xy[0, 0, 0], grid_xy[0, -1, 0])
    ax_flow.set_ylim(grid_xy[0, 0, 1], grid_xy[-1, 0, 1])
    ax_flow.set_aspect("equal")
    ax_flow.set_axis_off()

    t = np.linspace(0.0, 1.0, len(pos_xy))
    ax_path.plot(t, pos_xy[:, 0, 0], color=PURCELL_COLOUR)
    ax_path.plot(t, pos_xy[:, 1, 0], color=SCALLOP_COLOUR, linestyle="--")
    _zero_axes(ax_path, "time", "$x$")

    ax_hinge.plot(angles[:, 0, 0], angles[:, 0, 1], color=PURCELL_COLOUR)
    ax_hinge.plot(angles[:, 1, 0], angles[:, 1, 1], color=SCALLOP_COLOUR, linestyle="--")
    ax_hinge.set_aspect("equal")
    # Room above the loop for the name of the line up.
    ax_hinge.margins(x=HINGE_MARGIN, y=HINGE_ROOM)
    _cross(ax_hinge, "left hinge", "right hinge")

    figure.tight_layout()
    return figure


def animate_swimmers(
    positions: core.Point,
    world_sites: core.Point,
    grid: core.Point,
    flows: Iterable[core.Direction],
    every: int,
) -> Iterator[np.ndarray]:
    """Frames of Purcell's non-reciprocal swimmer (top), with the flow it drives at each frame
    `[grid_y, grid_x] Direction`, and the reciprocal flapper (bottom)."""
    beads_xy = point_coords(world_sites.batch())[::every]                       # [frames, swimmers, beads, 2]
    all_pos_xy = point_coords(positions)                                        # [steps + 1, swimmers, 2]
    grid_xy = point_coords(grid)                                                # [gy, gx, 2]

    x_min = min(float(all_pos_xy[:, 0, 0].min()), 0.0) - 2.3
    x_max = max(float(all_pos_xy[:, 0, 0].max()), 0.0) + 2.3
    # The figure set up once; the flow, the trails and the swimmers move.
    figure, axes = plt.subplots(2, 1, figsize=(6.2, 4.6), dpi=55)
    arrows = axes[0].quiver(
        grid_xy[..., 0], grid_xy[..., 1],
        np.zeros(grid_xy.shape[:-1]), np.zeros(grid_xy.shape[:-1]),
        color=FLOW_COLOUR, scale=3.2, width=0.003, zorder=1,
    )
    trails, swimmers = [], []
    for ax, colour, beads in zip(axes, (PURCELL_COLOUR, SCALLOP_COLOUR), beads_xy[0]):
        trails += ax.plot([], [], color=TRAIL_COLOUR, linewidth=1.6, zorder=2)
        swimmers.append(_draw_swimmer(ax, beads, colour, 1.0))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal")
        ax.set_axis_off()
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, hspace=0.06)

    for frame, flow in enumerate(flows):
        flow_xy = point_coords(flow)                                            # [gy, gx, 2]
        step_end = frame * every + 1
        arrows.set_UVC(flow_xy[..., 0], flow_xy[..., 1])
        for swimmer, (trail, parts) in enumerate(zip(trails, swimmers)):
            trail.set_data(all_pos_xy[:step_end, swimmer, 0], all_pos_xy[:step_end, swimmer, 1])
            _move_swimmer(parts, beads_xy[frame, swimmer])
        yield capture(figure)
    plt.close(figure)
