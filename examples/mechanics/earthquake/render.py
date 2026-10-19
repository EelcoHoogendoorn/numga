"""Drawing for the earthquake focal mechanisms: beachballs in the equal-area projection of the lower
hemisphere, and the fault in space with the lobes of its P waves."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.mechanics.earthquake import core

# Pushes shaded, pulls left white; the nodal lines, the pressure and tension axes, the slip.
PUSH, PULL = "#1e293b", "#f8fafc"
NODAL, PRESSURE, TENSION, SLIP = "#0f172a", "#0ea5e9", "#e11d48", "#f59e0b"
# The lobes in space, pushing and pulling, and the light that shades them.
PUSH_LOBE, PULL_LOBE = "#475569", "#e2e8f0"
LIGHT = LightSource(azdeg=315, altdeg=45)
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


def draw_ball(ax, directions: core.Vector, push: core.Scalar, axes: core.Vector) -> None:
    """The beachball of one fault: the lower hemisphere `[rings, spokes]` with the pushes shaded, its
    nodal lines, and its pressure and tension axes, north up."""
    grid, push = lambert(directions), push.to_array()
    bound = 2 * np.abs(push).max()
    ax.contourf(*np.moveaxis(grid, -1, 0), push, levels=[-bound, 0, bound], colors=[PULL, PUSH])
    ax.contour(*np.moveaxis(grid, -1, 0), push, levels=[0], colors=NODAL, linewidths=1.4)
    angle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(angle), np.sin(angle), color=NODAL, linewidth=1.6)
    principal = lambert(axes)
    for point, colour in [(principal[0], PRESSURE), (principal[2], TENSION)]:
        ax.scatter(*point, s=110, color=colour, edgecolor="white", linewidth=1.2, zorder=5)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_mechanisms(directions: core.Vector, push: core.Scalar, axes: core.Vector) -> plt.Figure:
    """The beachball of each fault `[faults]`, side by side."""
    count = push.shape[0]
    figure, panels = plt.subplots(1, count, figsize=(3.2 * count, 3.2), dpi=120)
    for index, ax in enumerate(panels):
        draw_ball(ax, directions, push[index], axes[index])
    figure.tight_layout()
    return figure


def draw_fault(ax, directions: core.Vector, push: core.Scalar, normal: core.Vector, slip: core.Vector,
               null: core.Vector) -> None:
    """One fault in space: the patch of the fault, the slip of each side across it, and the lobes of
    the P waves, each direction drawn as far as its first motion, pushes dark and pulls pale."""
    # The rings in reverse, so that the cells of the surface face outwards, towards the light.
    push = push.to_array()[::-1]
    surface = upright(directions)[::-1] * np.abs(push)[..., None]
    # The colour of each cell by the sign at its middle: the nodal planes run along the grid lines.
    middle = (push[1:, 1:] + push[:-1, :-1] + push[1:, :-1] + push[:-1, 1:])[..., None]
    colours = np.where(middle > 0, to_rgba(PUSH_LOBE), to_rgba(PULL_LOBE))
    ax.computed_zorder = False
    ax.plot_surface(*np.moveaxis(surface, -1, 0), facecolors=colours, linewidth=0, antialiased=False, shade=True,
                    lightsource=LIGHT, rstride=1, cstride=1, zorder=1)
    normal, slip, null = upright(normal), upright(slip), upright(null)
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


def animate_sweep(hemisphere: core.Vector, frames: Iterable[tuple[core.Vector, core.Vector, core.Vector, core.Scalar,
                                                                     core.Vector, core.Scalar]]) -> list[np.ndarray]:
    """Each frame of the rake's turn: the fault in space with its P lobes, and its beachball over the
    lower hemisphere."""
    images = []
    for normal, slip, axes, ball, sphere, lobes in frames:
        figure = plt.figure(figsize=(8.4, 4.2), dpi=100)
        draw_fault(figure.add_subplot(1, 2, 1, projection="3d"), sphere, lobes, normal, slip, axes[1])
        draw_ball(figure.add_subplot(1, 2, 2), hemisphere, ball, axes)
        figure.tight_layout()
        images.append(capture(figure))
        plt.close(figure)
    return images
