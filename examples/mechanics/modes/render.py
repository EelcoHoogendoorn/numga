"""Matplotlib views of rigid-body normal modes and spring suspensions.

The mechanics supplies infinitesimal mode displacements. Outlines and springs
are displaced linearly and exaggerated for readability.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from examples.util.animation import capture
from numga import stack
from examples.mechanics.modes.core import ModeCase, Point, mv


BODY, GHOST, SUPPORT, SPRING = "#254e70", "#929ba7", "#3a414b", "#5b6470"
# A spring's coil: its places along the spring as shares of its length, straight leads at both ends, and how far
# each zigzags across it, to either side in turn.
n_coil_turns = 15
coil_along = np.r_[0.0, 0.16, np.linspace(0.20, 0.80, n_coil_turns), 0.84, 1.0]
coil_across = np.r_[0.0, 0.0, 0.035 * (-1.0) ** np.arange(n_coil_turns), 0.0, 0.0]


def coordinates(points: Point) -> np.ndarray:
    """Coordinates of points of unit weight, and of directions, `[..., 2]`: their pairings with the planes x = 0 and
    y = 0."""
    return (stack((mv.x, mv.y)) & points[..., None]).to_array()


def spring_path(start: Point, end: Point) -> np.ndarray:
    """A coil with straight leads, oriented between two endpoints."""
    along = end - start                                                      # [] Point-
    # Across the spring: its step turned a quarter turn in the plane, over its length, the size of the line through
    # its ends.
    across = along.commutator(mv.xy) / (start & end).norm()                  # [] Point-
    return coordinates(start + along * coil_along + across * coil_across)


def fixed_support(ax: plt.Axes, anchor: Point) -> None:
    """A fixed anchor, as a dot."""
    ax.scatter(*coordinates(anchor), s=22, color=SUPPORT, zorder=4)


def _new_figure(cases: list[ModeCase], header: float, top: float, hspace: float) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """A row of three mode panels per case, with `header` inches above them."""
    n = len(cases)
    fig, axes_grid = plt.subplots(n, 3, figsize=(9, 2.7 * n + header), dpi=75, facecolor="white", squeeze=False)
    axes = axes_grid.tolist()

    all_points = np.concatenate([
        np.concatenate((coordinates(case.body), coordinates(case.anchors)))
        for case in cases
    ])
    lo, hi = all_points.min(axis=0) - [0.35, 0.35], all_points.max(axis=0) + [0.25, 0.25]
    for row in axes:
        for ax in row:
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
            ax.set_aspect("equal")
            ax.set_axis_off()

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=top, wspace=0.04, hspace=hspace)
    return fig, axes


def _draw_mode(ax: plt.Axes, case: ModeCase, mode: int):
    """Draw one mode panel and return its phase-update function."""
    reference = Polygon(coordinates(case.body), closed=True, facecolor="none", edgecolor=GHOST,
                        lw=1.0, linestyle=":", zorder=1)
    ax.add_patch(reference)

    # The largest offset of a corner drawn a fifth of a unit long; a direction's length is the size of the line
    # through the origin square to it, its dual.
    scale = 0.20 / case.body_offsets[mode].dual().norm().to_array().max()

    displaced_body = case.body + case.body_offsets[mode] * scale
    body = Polygon(coordinates(displaced_body), closed=True,
                   facecolor="#d4e3ed", edgecolor=BODY, lw=1.6, zorder=3)
    ax.add_patch(body)
    springs = []
    for anchor in case.anchors:
        fixed_support(ax, anchor)
        spring, = ax.plot([], [], lw=1.3, color=SPRING, zorder=5)
        springs.append(spring)
    markers = ax.scatter([], [], s=14, color=BODY, zorder=6)

    def update(phase: float):
        s = phase * scale
        current_body = case.body + case.body_offsets[mode] * s
        current_attachments = case.attachments + case.attachment_offsets[mode] * s

        body.set_xy(coordinates(current_body))
        markers.set_offsets(coordinates(current_attachments))
        for spring, anchor, attachment in zip(springs, case.anchors, current_attachments):
            spring.set_data(*spring_path(anchor, attachment).T)
        return [body, markers, *springs]

    update(1.0)
    return update


def draw_modes(cases: list[ModeCase]) -> plt.Figure:
    """The three mode panels of each suspension case, one row per case."""
    fig, axes = _new_figure(cases, 0.0, 0.98, 0.04)
    for case, row in zip(cases, axes):
        for mode, ax in enumerate(row):
            _draw_mode(ax, case, mode)
    return fig


def animate_modes(cases: list[ModeCase], frames: int) -> Iterator[np.ndarray]:
    """Release each isolated mode from rest in synchronized physical time."""
    fig, axes = _new_figure(cases, 0.0, 0.98, 0.04)
    fig.set_dpi(67)
    updates = [_draw_mode(ax, case, mode)
               for case, row in zip(cases, axes) for mode, ax in enumerate(row)]
    frequencies = np.concatenate([case.frequencies.to_array() for case in cases])
    duration = 2 / np.min(frequencies[frequencies > 0])

    for time in np.linspace(0, duration, frames):
        for update, phase in zip(updates, np.cos(2 * np.pi * frequencies * time)):
            update(phase)
        yield capture(fig)
    plt.close(fig)


def render_setup(
    body: Point,
    anchors: Point,
    attachments: Point,
) -> plt.Figure:
    """The suspension at equilibrium: the plate, its springs and their anchors."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120, facecolor="white")

    plate = Polygon(coordinates(body), closed=True, facecolor="#d4e3ed", edgecolor=BODY, lw=1.6, zorder=3)
    ax.add_patch(plate)

    for anchor, attachment in zip(anchors, attachments):
        fixed_support(ax, anchor)
        ax.plot(*spring_path(anchor, attachment).T, lw=1.3, color=SPRING, zorder=5)

    att_coords = coordinates(attachments)
    ax.scatter(att_coords[:, 0], att_coords[:, 1], s=14, color=BODY, zorder=6)

    all_pts = np.concatenate([coordinates(body), coordinates(anchors)])
    lo, hi = all_pts.min(axis=0) - [0.35, 0.35], all_pts.max(axis=0) + [0.35, 0.35]
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    return fig
