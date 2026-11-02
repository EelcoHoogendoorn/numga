"""Views of rigid-body normal modes and spring suspensions, drawn on the GPU in the plate's own algebra.

The mechanics supplies infinitesimal mode displacements. Outlines and springs
are displaced linearly and exaggerated for readability.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera, panel
from numga import concatenate, stack
from examples.mechanics.modes.core import ModeCase, Point, mv


BODY, GHOST, SUPPORT, SPRING, PLATE = (to_rgb(colour) for colour in ("#254e70", "#929ba7", "#3a414b", "#5b6470", "#d4e3ed"))
# Widths and radii in pixels; the reference outline dotted, dots of no length a few pixels apart.
BODY_WIDTH, GHOST_WIDTH, SPRING_WIDTH = 2.0, 1.5, 1.5
GHOST_DOTS = (0.0, 3.5)
SUPPORT_RADIUS, ATTACHMENT_RADIUS = 2.6, 2.0
# A spring's coil: its places along the spring as shares of its length, straight leads at both ends, and how far
# each zigzags across it, to either side in turn.
n_coil_turns = 15
coil_along = np.r_[0.0, 0.16, np.linspace(0.20, 0.80, n_coil_turns), 0.84, 1.0]
coil_across = np.r_[0.0, 0.0, 0.035 * (-1.0) ** np.arange(n_coil_turns), 0.0, 0.0]


def coordinates(points: Point) -> np.ndarray:
    """Coordinates of points of unit weight, and of directions, `[..., 2]`: their pairings with the planes x = 0 and
    y = 0."""
    return (stack((mv.x, mv.y)) & points[..., None]).to_array()


def extent(points: list[Point]) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and the highest readings across and up of batches of points of unit weight, `[2]` each: their
    pairings with the planes x = 0 and y = 0."""
    readings = concatenate([stack((mv.x, mv.y))[:, None] & batch.reshape(-1) for batch in points], axis=-1)
    return readings.min(axis=-1).to_array(), readings.max(axis=-1).to_array()


def spring_paths(starts: Point, ends: Point) -> Point:
    """Coils with straight leads between pairs of endpoints `[springs]`, their places `[springs, coil]`."""
    along = ends - starts                                                    # [springs] Point-
    # Across each spring: its step turned a quarter turn in the plane, over its length, the size of the line through
    # its ends.
    across = along.commutator(mv.xy) / (starts & ends).norm()                # [springs] Point-
    return starts[:, None] + along[:, None] * coil_along + across[:, None] * coil_across


def plate(n_corners: int) -> tuple[simplices.Topology, simplices.Topology]:
    """A convex plate of `n_corners` corners as triangles fanning out from its first corner, and its rim, the fan's
    boundary edges."""
    corners = np.arange(n_corners)
    surface = simplices.Topology(n_corners, np.stack([np.zeros(n_corners - 2, dtype=int), corners[1:-1], corners[2:]], -1))
    return surface, surface.border


def suspension(camera: Camera, body: Point, anchors: Point, attachments: Point) -> list[np.ndarray]:
    """The pictures of a plate on its springs: the plate filled and outlined, the springs, their anchors and the
    attachments on the plate."""
    surface, rim = plate(len(body))
    paths = spring_paths(anchors, attachments).reshape(-1)                   # [springs * coil] Point-
    coils = simplices.Topology.chains([len(coil_along)] * len(anchors))
    return [contours.picture(body, surface, np.zeros(len(body)), PLATE, camera),
            simplices.strokes(body, rim, BODY, BODY_WIDTH, camera),
            simplices.markers(anchors, SUPPORT, SUPPORT_RADIUS, camera),
            simplices.strokes(paths, coils, SPRING, SPRING_WIDTH, camera),
            simplices.markers(attachments, BODY, ATTACHMENT_RADIUS, camera)]


def _new_figure(cases: list[ModeCase], header: float, top: float, hspace: float) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """A row of three mode panels per case, with `header` inches above them."""
    n = len(cases)
    fig, axes_grid = plt.subplots(n, 3, figsize=(9, 2.7 * n + header), dpi=75, facecolor="white", squeeze=False)
    axes = axes_grid.tolist()

    low, high = extent([points for case in cases for points in (case.body, case.anchors)])
    lo, hi = low - 0.35, high + 0.25
    for row in axes:
        for ax in row:
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
            ax.set_aspect("equal")
            ax.set_axis_off()

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=top, wspace=0.04, hspace=hspace)
    return fig, axes


def _draw_mode(ax: plt.Axes, case: ModeCase, mode: int):
    """Draw one mode panel and return its phase-update function."""
    camera = panel(ax, mv.x, mv.y)
    _, rim = plate(len(case.body))
    reference = simplices.strokes(case.body, rim, GHOST, GHOST_WIDTH, camera, dashes=GHOST_DOTS)

    # The largest offset of a corner drawn a fifth of a unit long; a direction's length is the size of the line
    # through the origin square to it, its dual.
    lengths = case.body_offsets[mode].dual().norm().reshape(-1)
    scale = 0.20 / lengths.max()

    def picture(phase: float) -> np.ndarray:
        s = phase * scale
        current_body = case.body + case.body_offsets[mode] * s
        current_attachments = case.attachments + case.attachment_offsets[mode] * s
        return simplices.composited([reference, *suspension(camera, current_body, case.anchors, current_attachments)])

    image = simplices.overlay(ax, [picture(1.0)])

    def update(phase: float):
        image.set_data(picture(phase))
        return [image]

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

    low, high = extent([body, anchors])
    lo, hi = low - 0.35, high + 0.35
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    simplices.overlay(ax, suspension(panel(ax, mv.x, mv.y), body, anchors, attachments))
    return fig
