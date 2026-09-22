"""Matplotlib views of planar rigid-body normal modes and spring suspensions.

The mechanics supplies infinitesimal mode displacements. Outlines and springs
are displaced linearly and exaggerated for readability; spring colour records
the linear change in length, rather than the length of the exaggerated drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
import numpy as np

from examples.mechanics.modes.modes import Point, Scalar
from numga.algebras import PGA2D


BODY, GHOST, SUPPORT = "#254e70", "#929ba7", "#3a414b"
STRETCH, COMPRESS, UNCHANGED = "#d67721", "#2783b7", "#87909a"


@dataclass
class PlotCase:
    body: Point                     # [4] Point
    attachments: Point              # [n_springs] Point
    anchors: Point                  # [n_springs] Point
    frequencies: np.ndarray         # [3] float
    body_offsets: Point             # [3, 4] Point
    attachment_offsets: Point       # [3, n_springs] Point
    extensions: Scalar              # [3, n_springs] Scalar


def coordinates(points: Point) -> np.ndarray:
    """Read Cartesian coordinates from PGA2D Point antivectors for plotting."""
    return points.select_subspace(PGA2D.subspace("yw wx")).kernel


def spring_path(start: Point, end: Point) -> np.ndarray:
    """A coil with straight leads, oriented between two endpoints."""
    p0 = coordinates(start)
    p1 = coordinates(end)
    axis = p1 - p0
    normal = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
    along = np.r_[0.0, 0.16, np.linspace(0.20, 0.80, 15), 0.84, 1.0]
    across = np.zeros_like(along)
    across[2:-2] = 0.048 * (-1.0) ** np.arange(15)
    return p0 + along[:, None] * axis + across[:, None] * normal


def spring_colour(extension: float) -> str:
    return STRETCH if extension > 1e-9 else COMPRESS if extension < -1e-9 else UNCHANGED


def fixed_support(ax: plt.Axes, anchor: Point, attachment: Point) -> None:
    """Draw a wall hatch fixture at an anchor point perpendicular to the spring axis."""
    anc = coordinates(anchor)
    att = coordinates(attachment)
    direction = (anc - att) / np.linalg.norm(anc - att)
    tangent = np.array([-direction[1], direction[0]])
    ends = anc + np.array([-0.14, 0.14])[:, None] * tangent
    ax.plot(*ends.T, color=SUPPORT, lw=2, zorder=4)
    for offset in np.linspace(-0.12, 0.12, 5):
        start = anc + offset * tangent
        end = start + 0.075 * (direction + tangent)
        ax.plot(*np.array([start, end]).T, color=SUPPORT, lw=1)


def _new_figure(cases: list[PlotCase], title: str = "") -> tuple[plt.Figure, list[list[plt.Axes]]]:
    n = len(cases)
    fig, axes_grid = plt.subplots(n, 3, figsize=(9, 2.7 * n + (0.5 if title else 0)), dpi=150, facecolor="white")
    axes = axes_grid.tolist() if n > 1 else [axes_grid.tolist()]

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

    if title:
        fig.suptitle(title, fontsize=12, color=SUPPORT, y=0.98)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.91, wspace=0.04, hspace=0.08)
    else:
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.04, hspace=0.04)
    return fig, axes


def _draw_mode(ax: plt.Axes, case: PlotCase, mode: int):
    """Draw one mode panel and return its phase-update function."""
    reference = Polygon(coordinates(case.body), closed=True, facecolor="#eef2f6", edgecolor=GHOST,
                        lw=1.2, linestyle=":", zorder=1)
    ax.add_patch(reference)

    offsets = coordinates(case.body_offsets[mode])
    scale = 0.20 / np.linalg.norm(offsets, axis=-1).max()

    displaced_body = case.body + case.body_offsets[mode] * scale
    body = Polygon(coordinates(displaced_body), closed=True,
                   facecolor="#d4e3ed", edgecolor=BODY, lw=2, alpha=0.9, zorder=3)
    ax.add_patch(body)
    springs = []
    for anchor, attachment in zip(case.anchors, case.attachments):
        fixed_support(ax, anchor, attachment)
        spring, = ax.plot([], [], lw=2.2, zorder=5)
        springs.append(spring)
    markers = ax.scatter([], [], s=25, facecolor="white", edgecolor=BODY,
                         linewidth=1.2, zorder=6)

    def update(phase: float):
        s = phase * scale
        current_body = case.body + case.body_offsets[mode] * s
        current_attachments = case.attachments + case.attachment_offsets[mode] * s
        current_extensions = case.extensions[mode] * s

        body.set_xy(coordinates(current_body))
        markers.set_offsets(coordinates(current_attachments))
        for spring, anchor, attachment, ext in zip(
            springs, case.anchors, current_attachments, current_extensions
        ):
            spring.set_data(*spring_path(anchor, attachment).T)
            ext_val = float(ext.kernel[0])
            spring.set_color(spring_colour(ext_val))
        return [body, markers, *springs]

    update(1.0)
    return update


def draw_modes(cases: list[PlotCase], title: str = "", plot_path: str = "") -> plt.Figure:
    """Render mode plots for one or two suspension cases."""
    fig, axes = _new_figure(cases, title=title)
    for case, row in zip(cases, axes):
        for mode, ax in enumerate(row):
            _draw_mode(ax, case, mode)
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, facecolor="white")
    return fig


def save_animation(cases: list[PlotCase], animation_path: str) -> None:
    """Release each isolated mode from rest in synchronized physical time."""
    fig, axes = _new_figure(cases)
    fig.set_dpi(80)
    updates = [_draw_mode(ax, case, mode)
               for case, row in zip(cases, axes) for mode, ax in enumerate(row)]
    frequencies = np.concatenate([case.frequencies for case in cases])
    duration = 2 / np.min(frequencies[frequencies > 0])

    def frame(time: float):
        artists = []
        for update, phase in zip(updates, np.cos(2 * np.pi * frequencies * time)):
            artists.extend(update(phase))
        return artists

    animation = FuncAnimation(fig, frame, frames=np.linspace(0, duration, 120),
                              interval=50, blit=False)
    Path(animation_path).parent.mkdir(parents=True, exist_ok=True)
    animation.save(animation_path, writer=PillowWriter(fps=20), dpi=80)
    plt.close(fig)


def render_setup(
    body: Point,
    anchors: Point,
    attachments: Point,
    title: str = "Baseline Suspension: Two Vertical Springs",
    plot_path: str = "",
) -> plt.Figure:
    """Render the physical suspension layout at equilibrium."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120, facecolor="white")

    plate = Polygon(coordinates(body), closed=True, facecolor="#d4e3ed", edgecolor=BODY, lw=2, zorder=3)
    ax.add_patch(plate)

    for anchor, attachment in zip(anchors, attachments):
        fixed_support(ax, anchor, attachment)
        path = spring_path(anchor, attachment)
        ax.plot(*path.T, lw=2.2, color=UNCHANGED, zorder=5)

    att_coords = coordinates(attachments)
    ax.scatter(att_coords[:, 0], att_coords[:, 1], s=30, facecolor="white", edgecolor=BODY, linewidth=1.2, zorder=6)

    all_pts = np.concatenate([coordinates(body), coordinates(anchors)])
    lo, hi = all_pts.min(axis=0) - [0.35, 0.35], all_pts.max(axis=0) + [0.35, 0.35]
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, color=SUPPORT, pad=6)

    fig.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, facecolor="white")
    return fig


def render_mass_distribution(
    body: Point,
    title: str = "Plate Corner Masses",
    plot_path: str = "",
) -> plt.Figure:
    """Plot the rigid plate and its 4 corner mass points at rest."""
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120, facecolor="white")
    b = coordinates(body)
    plate = Polygon(b, closed=True, facecolor="#d4e3ed", edgecolor=BODY, lw=2, zorder=2)
    ax.add_patch(plate)
    ax.scatter(b[:, 0], b[:, 1], s=60, color=STRETCH, edgecolor=BODY, linewidth=1.2, zorder=5)
    ax.set_aspect("equal")
    lo, hi = b.min(axis=0) - [0.3, 0.3], b.max(axis=0) + [0.3, 0.3]
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, color=SUPPORT, pad=6)
    fig.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, facecolor="white")
    return fig
