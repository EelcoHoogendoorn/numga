"""Drawing for the earthquake focal mechanisms: beachballs in the equal-area projection of the lower
hemisphere, and the fault in space with the lobes of its P waves."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.mechanics.earthquake import core

# Pushes shaded, pulls left white; the nodal lines, the pressure and tension axes, the slip.
PUSH, PULL = "#1e293b", "#f8fafc"
NODAL, PRESSURE, TENSION, SLIP = "#0f172a", "#0ea5e9", "#e11d48", "#f59e0b"
# The half width of the fault patch, and of the view in space.
PATCH = 0.55
LIMIT = 0.75


def coordinates(vectors: core.Vector) -> np.ndarray:
    """The north, east and down coordinates of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def lambert(vectors: core.Vector) -> np.ndarray:
    """Directions in the equal-area projection of the lower hemisphere, east to the right and north
    up: a direction on the upper hemisphere is drawn as its opposite."""
    ned = coordinates(vectors)
    ned = np.where(ned[..., 2:] < 0, -ned, ned)
    return ned[..., [1, 0]] / np.sqrt(1 + ned[..., 2:])


def upright(vectors: core.Vector) -> np.ndarray:
    """Vectors in space for drawing: east, north and up."""
    ned = coordinates(vectors)
    return np.stack([ned[..., 1], ned[..., 0], -ned[..., 2]], axis=-1)


def draw_ball(ax, faults: core.Mechanism, index: int, title: str) -> None:
    """The beachball of one fault: the lower hemisphere with the pushes shaded, its nodal lines, and
    its pressure and tension axes."""
    grid, push = lambert(faults.directions), faults.push[index].to_array()
    bound = 2 * np.abs(push).max()
    ax.contourf(*np.moveaxis(grid, -1, 0), push, levels=[-bound, 0, bound], colors=[PULL, PUSH])
    ax.contour(*np.moveaxis(grid, -1, 0), push, levels=[0], colors=NODAL, linewidths=1.4)
    angle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(angle), np.sin(angle), color=NODAL, linewidth=1.6)
    axes = lambert(faults.axes[index])
    for point, label, colour in [(axes[0], "P", PRESSURE), (axes[2], "T", TENSION)]:
        ax.scatter(*point, s=110, color=colour, edgecolor="white", linewidth=1.2, zorder=5)
        ax.text(*point, label, ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=6)
    ax.text(0, 1.06, "N", ha="center", va="bottom", fontsize=8, color="#64748b")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.18)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=6)


def draw_mechanisms(faults: core.Mechanism, names: list[str]) -> plt.Figure:
    """The beachball of each fault, side by side."""
    figure, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), dpi=120)
    for index, (ax, name) in enumerate(zip(axes, names)):
        draw_ball(ax, faults, index, name)
    figure.tight_layout()
    return figure


def draw_fault(ax, lobes: core.Mechanism, index: int) -> None:
    """One fault in space: the patch of the fault, the slip of each side across it, and the lobes of
    the P waves, each direction drawn as far as its first motion, pushes dark and pulls pale."""
    push = lobes.push[index].to_array()
    surface = upright(lobes.directions) * np.abs(push)[..., None]
    shade = np.where(push[..., None] > 0, to_rgba(PUSH, 0.9), to_rgba("#cbd5e1", 0.6))
    ax.computed_zorder = False
    ax.plot_surface(*np.moveaxis(surface, -1, 0), facecolors=shade, linewidth=0, antialiased=True, shade=False, zorder=1)
    normal, slip, null = upright(lobes.normals[index]), upright(lobes.slips[index]), upright(lobes.axes[index, 1])
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)[:, None]
    rim = PATCH * (slip * np.cos(angle) + null * np.sin(angle))
    ax.add_collection3d(Poly3DCollection([rim], alpha=0.3, facecolor="#fde68a", edgecolor="#b45309", linewidth=1.2, zorder=2))
    for side in (1, -1):
        base = side * 0.15 * normal - 0.45 * side * slip
        ax.quiver(*base, *(0.9 * side * slip), color=SLIP, linewidth=2.6, arrow_length_ratio=0.25, zorder=3)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_zlim(-LIMIT, LIMIT)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=24, azim=-60)
    ax.set_axis_off()


def animate_sweep(balls: core.Mechanism, lobes: core.Mechanism) -> list[np.ndarray]:
    """Each frame of the rake's turn: the fault in space with its P lobes, and its beachball."""
    images = []
    for index in range(balls.push.shape[0]):
        figure = plt.figure(figsize=(8.4, 4.2), dpi=100)
        draw_fault(figure.add_subplot(1, 2, 1, projection="3d"), lobes, index)
        draw_ball(figure.add_subplot(1, 2, 2), balls, index, "beachball")
        figure.tight_layout()
        images.append(capture(figure))
        plt.close(figure)
    return images
