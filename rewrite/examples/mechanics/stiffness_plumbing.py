"""Matplotlib views of the planar spring stiffness modes.

The mechanics supplies infinitesimal mode displacements. Outlines and springs
are displaced linearly and exaggerated for readability; spring colour records
the linear change in length, rather than the length of the exaggerated drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np


BODY, GHOST, SUPPORT = "#254e70", "#929ba7", "#3a414b"
STRETCH, COMPRESS, UNCHANGED = "#d67721", "#2783b7", "#87909a"


@dataclass
class PlotCase:
    title: str
    description: str
    body: np.ndarray
    attachments: np.ndarray
    anchors: np.ndarray
    frequencies: np.ndarray
    body_offsets: np.ndarray
    attachment_offsets: np.ndarray
    extensions: np.ndarray
    labels: tuple[str, str, str]


def spring_path(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """A coil with straight leads, oriented between two endpoints."""
    axis = end - start
    normal = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
    along = np.r_[0.0, 0.16, np.linspace(0.20, 0.80, 15), 0.84, 1.0]
    across = np.zeros_like(along)
    across[2:-2] = 0.048 * (-1.0) ** np.arange(15)
    return start + along[:, None] * axis + across[:, None] * normal


def spring_colour(extension: float) -> str:
    return STRETCH if extension > 1e-9 else COMPRESS if extension < -1e-9 else UNCHANGED


def fixed_support(ax: plt.Axes, anchor: np.ndarray, attachment: np.ndarray) -> None:
    direction = (anchor - attachment) / np.linalg.norm(anchor - attachment)
    tangent = np.array([-direction[1], direction[0]])
    ends = anchor + np.array([-0.14, 0.14])[:, None] * tangent
    ax.plot(*ends.T, color=SUPPORT, lw=2, zorder=4)
    for offset in np.linspace(-0.12, 0.12, 5):
        start = anchor + offset * tangent
        end = start + 0.075 * (direction + tangent)
        ax.plot(*np.array([start, end]).T, color=SUPPORT, lw=1)


def _new_figure(cases: list[PlotCase]) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    if len(cases) != 2:
        raise ValueError("The mode comparison expects two spring layouts.")
    fig = plt.figure(figsize=(15, 9), dpi=150, facecolor="white")
    fig.suptitle("Spring geometry selects the motions of a rigid body", y=0.968,
                 fontsize=20, fontweight="medium", color=SUPPORT)
    fig.text(0.5, 0.925, "Each panel isolates one natural mode of the same planar body.",
             ha="center", fontsize=12, color=SUPPORT)
    all_points = np.concatenate([np.concatenate((case.body, case.anchors)) for case in cases])
    lo, hi = all_points.min(axis=0) - [0.36, 0.33], all_points.max(axis=0) + [0.27, 0.25]
    axes = []
    for case, bottom, heading in zip(cases, (0.535, 0.16), (0.87, 0.495)):
        fig.text(0.055, heading, case.title, fontsize=14, fontweight="bold", color=SUPPORT)
        fig.text(0.055, heading - 0.027, case.description, fontsize=11, color=SUPPORT)
        row = []
        for mode in range(3):
            ax = fig.add_axes([0.038 + 0.322 * mode, bottom, 0.286, 0.265])
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
            ax.set_aspect("equal")
            ax.set_axis_off()
            frequency = case.frequencies[mode]
            detail = "free · 0 Hz" if frequency == 0 else f"{frequency:.2f} Hz"
            ax.set_title(f"{case.labels[mode]}   /   {detail}", fontsize=12, pad=7,
                         color=BODY)
            row.append(ax)
        axes.append(row)
    handles = [
        Line2D([], [], color=STRETCH, lw=2.5, label="Spring lengthens"),
        Line2D([], [], color=COMPRESS, lw=2.5, label="Spring shortens"),
        Line2D([], [], color=UNCHANGED, lw=2.5, label="No first-order change"),
        Line2D([], [], color=GHOST, lw=1.5, ls=":", label="Reference body"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.088),
               ncol=4, frameon=False, fontsize=10, handlelength=2.5, columnspacing=2.7)
    fig.text(0.5, 0.062, "Small-motion modes, with displacements amplified. "
             "Dashed outline: the opposite displacement.", ha="center", fontsize=10, color=SUPPORT)
    fig.text(0.5, 0.035, "Springs begin relaxed; gravity is omitted. "
             "A free mode has no linear restoring force and stays displaced when released at rest.",
             ha="center", fontsize=10, color=SUPPORT)
    return fig, axes


def _draw_mode(ax: plt.Axes, case: PlotCase, mode: int, arrows: bool):
    """Draw one panel and return its phase-update function."""
    reference = Polygon(case.body, closed=True, facecolor="#e8edf2", edgecolor=GHOST,
                        lw=1.5, linestyle=":", zorder=1)
    ax.add_patch(reference)
    opposite = Polygon(case.body - case.body_offsets[mode], closed=True,
                       facecolor="none", edgecolor=BODY, lw=1.1, linestyle="--", alpha=0.36)
    ax.add_patch(opposite)
    body = Polygon(case.body + case.body_offsets[mode], closed=True,
                   facecolor="#d4e3ed", edgecolor=BODY, lw=2, alpha=0.88, zorder=3)
    ax.add_patch(body)
    springs = []
    for anchor, attachment in zip(case.anchors, case.attachments):
        fixed_support(ax, anchor, attachment)
        spring, = ax.plot([], [], lw=2.2, zorder=5)
        springs.append(spring)
    markers = ax.scatter([], [], s=23, facecolor="white", edgecolor=BODY,
                         linewidth=1.2, zorder=6)
    if arrows:
        for point, offset in zip(case.body, case.body_offsets[mode]):
            if np.linalg.norm(offset) > 0.015:
                ax.annotate("", xy=point + offset, xytext=point,
                            arrowprops={"arrowstyle": "->", "color": BODY,
                                        "lw": 1.6, "shrinkA": 0, "shrinkB": 0}, zorder=7)

    def update(phase: float):
        body.set_xy(case.body + phase * case.body_offsets[mode])
        points = case.attachments + phase * case.attachment_offsets[mode]
        markers.set_offsets(points)
        for spring, anchor, point, extension in zip(
                springs, case.anchors, points, case.extensions[mode]):
            spring.set_data(*spring_path(anchor, point).T)
            spring.set_color(spring_colour(phase * extension))
        return [body, markers, *springs]

    update(1.0)
    return update


def draw_modes(cases: list[PlotCase], plot_path: str) -> plt.Figure:
    """Save the two layouts and their three independent small-motion modes."""
    fig, axes = _new_figure(cases)
    for case, row in zip(cases, axes):
        for mode, ax in enumerate(row):
            _draw_mode(ax, case, mode, arrows=True)
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, facecolor="white")
    return fig


def save_animation(cases: list[PlotCase], animation_path: str) -> None:
    """Release each isolated mode from rest; all panels share physical time.

    Zero-frequency modes remain at the released displacement. The GIF shows
    two periods of the slowest supported mode, then restarts the release.
    """
    fig, axes = _new_figure(cases)
    fig.set_dpi(80)
    fig.texts[1].set_text("Released from rest in each mode; the same elapsed time in every panel.")
    fig.texts[-2].set_text("Small-motion modes, with displacements amplified. "
                           "Dashed outline: opposite initial displacement.")
    updates = [_draw_mode(ax, case, mode, arrows=False)
               for case, row in zip(cases, axes) for mode, ax in enumerate(row)]
    frequencies = np.concatenate([case.frequencies for case in cases])
    duration = 2 / np.min(frequencies[frequencies > 0])
    clock = fig.text(0.94, 0.925, "", ha="right", fontsize=10, color=SUPPORT)

    def frame(time: float):
        clock.set_text(f"t = {time:.2f} s")
        artists = [clock]
        for update, phase in zip(updates, np.cos(2 * np.pi * frequencies * time)):
            artists.extend(update(phase))
        return artists

    animation = FuncAnimation(fig, frame, frames=np.linspace(0, duration, 120),
                              interval=50, blit=False)
    Path(animation_path).parent.mkdir(parents=True, exist_ok=True)
    animation.save(animation_path, writer=PillowWriter(fps=20), dpi=80)
    plt.close(fig)
