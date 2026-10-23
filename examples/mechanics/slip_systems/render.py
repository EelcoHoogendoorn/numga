"""The stereographic chart of pull directions, coloured by the slip system that yields first, and the paths a pull's
axis takes through the crystal."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from examples.util.animation import capture
from examples.mechanics.slip_systems import core

SYSTEM_MAP = "tab20"
# The Schmid factor at which a region is drawn white, and the one at which it is drawn in full colour.
FAINT, FULL = 0.25, 0.5
PATH_COLOUR = "0.1"
NORMAL_COLOUR = "0.15"
EDGE_COLOUR = "0.3"
# The corners of the standard triangle, the samples of its edges, and the window that frames it.
CORNERS = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
NAMES = ("[001]", "[011]", "[111]")
# The directions a pull's axis heads for: the leading slip direction, and the corner where two systems share.
GOALS = np.array([[1, 0, 1], [1, 1, 2]])
GOAL_NAMES = ("[101]", "[112]")
# How much darker than its systems' colours a path is drawn.
PATH_SHADE = 0.55
# How far each corner's name sits above its corner, in points: [001] below, the others above.
LIFTS = (-10, 8, 8)
EDGE_SAMPLES = 60
WINDOW = (-0.04, 0.46)
# The animation's side in inches, and in pixels.
SIDE = 4.6
PIXELS = 460


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def coordinates(directions: core.Direction) -> np.ndarray:
    """x, y and z of each direction, on the last axis."""
    return directions.cast(core.ga.subspace("yz zx xy")).kernel


def along(lines: core.Line) -> np.ndarray:
    """x, y and z along each line, on the last axis."""
    return lines.cast(core.ga.subspace("yz zx xy")).kernel


def normals(planes: core.Plane) -> np.ndarray:
    """x, y and z of each plane's normal, on the last axis."""
    return planes.cast(core.ga.subspace("x y z")).kernel


def projected(xyz: np.ndarray) -> np.ndarray:
    """Directions on the stereographic disk of the upper half of the sphere: a direction and its opposite are
    the same pull, so each is taken on its upper side."""
    unit = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    upper = unit * np.where(unit[..., 2:] < 0, -1, 1)
    return upper[..., :2] / (1 + upper[..., 2:])


def system_colours(count: int) -> np.ndarray:
    """The colour of each slip system, `[systems, 3]`."""
    return plt.get_cmap(SYSTEM_MAP)(np.arange(count) % 20)[:, :3]


def shades(factors: core.Scalar) -> np.ndarray:
    """A colour per pull direction: the hue of the system that yields first, lightened where its Schmid factor
    is small; and white outside the disk."""
    values = np.abs(scalars(factors)).reshape(factors.shape[:2] + (-1,))      # [rows, columns, systems]
    hue = system_colours(values.shape[-1])[values.argmax(axis=-1)]
    strength = np.clip((values.max(axis=-1) - FAINT) / (FULL - FAINT), 0, 1)[..., None]
    colour = 1 - strength * (1 - hue)
    across = np.linspace(-1, 1, factors.shape[0])
    inside = (across[None, :] ** 2 + across[:, None] ** 2 <= 1)[..., None]
    return np.where(inside, colour, 1.0)


def draw_triangle(ax: plt.Axes) -> None:
    """The standard triangle, between [001], [011] and [111], its edges arcs of great circles."""
    share = np.linspace(0, 1, EDGE_SAMPLES)[:, None]
    for start, end in zip(CORNERS, CORNERS[[1, 2, 0]]):
        ax.plot(*projected(start * (1 - share) + end * share).T, color=EDGE_COLOUR, linewidth=0.9)


def draw_disk(ax: plt.Axes, factors: core.Scalar, planes: core.Plane, slips: core.Line) -> None:
    """The whole disk: each region in the colour of its first system, the slip planes' normals as triangles and
    the slip directions as dots, and the standard triangle."""
    ax.imshow(shades(factors), extent=(-1, 1, -1, 1), origin="lower", interpolation="bilinear")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=EDGE_COLOUR, linewidth=0.9))
    ax.scatter(*projected(normals(planes)).T, marker="^", s=40, color=NORMAL_COLOUR, zorder=3)
    ax.scatter(*projected(along(slips).reshape(-1, 3)).T, marker="o", s=16, color=NORMAL_COLOUR, zorder=3)
    draw_triangle(ax)
    ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
    ax.set_axis_off()


def draw_close(ax: plt.Axes, factors: core.Scalar) -> None:
    """The standard triangle close up, in the colours of its systems, with the directions the axes head for."""
    ax.imshow(shades(factors), extent=(-1, 1, -1, 1), origin="lower", interpolation="bilinear")
    draw_triangle(ax)
    for corner, name, lift in zip(projected(CORNERS), NAMES, LIFTS):
        ax.annotate(name, corner, xytext=(0, lift), textcoords="offset points", ha="center", va="center",
                    fontsize=9, color=EDGE_COLOUR)
    for goal, name in zip(projected(GOALS), GOAL_NAMES):
        ax.scatter(*goal, marker="D", s=22, color=EDGE_COLOUR, zorder=3)
        ax.annotate(name, goal, xytext=(0, 9), textcoords="offset points", ha="center", va="center", fontsize=9,
                    color=EDGE_COLOUR)
    ax.set(xlim=WINDOW, ylim=WINDOW, aspect="equal")
    ax.set_axis_off()


def steps(axes: core.Direction, amounts: core.Scalar) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each pull's axis's places on the disk `[steps + 1, starts, 2]`, the steps between them
    `[starts, steps, ends, 2]`, and each step's colour `[starts, steps, 3]`: the colours of the systems that slipped
    in it, mixed by how much, darkened."""
    tracks = projected(coordinates(axes))                                     # [steps + 1, starts, 2]
    weights = np.abs(scalars(amounts)).reshape(amounts.shape[:2] + (-1,))      # [steps, starts, systems]
    mixed = weights @ system_colours(weights.shape[-1]) / weights.sum(axis=-1, keepdims=True)   # [steps, starts, 3]
    segments = np.stack([tracks[:-1], tracks[1:]], axis=2)                    # [steps, starts, ends, 2]
    return tracks, np.moveaxis(segments, 1, 0), np.moveaxis(mixed, 1, 0) * PATH_SHADE


def draw_paths(ax: plt.Axes, factors: core.Scalar, axes: core.Direction, amounts: core.Scalar) -> None:
    """The standard triangle close up, in the colours of its systems, with each pull's axis's path so far and
    its current place, each step in the colours of the systems that slipped in it, mixed by how much, and the
    directions the axes head for."""
    draw_close(ax, factors)
    tracks, segments, colours = steps(axes, amounts)
    for track, shade in zip(segments, colours):
        ax.add_collection(LineCollection(track, colors=shade, linewidths=1.6))
    ax.scatter(*tracks[-1].T, s=14, color=PATH_COLOUR, zorder=3)


def draw_chart(directions: core.Direction, factors: core.Scalar, planes: core.Plane, slips: core.Line,
               axes: core.Direction, amounts: core.Scalar) -> plt.Figure:
    """The whole disk, and the standard triangle with the paths."""
    figure, (disk, close) = plt.subplots(1, 2, figsize=(9.0, 4.5), layout="constrained")
    draw_disk(disk, factors, planes, slips)
    draw_paths(close, factors, axes, amounts)
    return figure


def animate_paths(directions: core.Direction, factors: core.Scalar, axes: core.Direction, amounts: core.Scalar,
                  frames: int) -> Iterator[np.ndarray]:
    """The paths through the standard triangle growing, over so many frames, the triangle drawn once."""
    return animate_paths_sized(directions, factors, axes, amounts, frames, PIXELS)


def animate_paths_sized(directions: core.Direction, factors: core.Scalar, axes: core.Direction, amounts: core.Scalar,
                        frames: int, pixels: int) -> Iterator[np.ndarray]:
    """The paths through the standard triangle growing, over so many frames, the triangle drawn once, the figure
    so many pixels across."""
    reached = np.linspace(2, axes.shape[0], frames).round().astype(int)
    tracks, segments, colours = steps(axes, amounts)
    figure, ax = plt.subplots(figsize=(SIDE, SIDE), layout="constrained")
    figure.set_dpi(pixels / SIDE)
    draw_close(ax, factors)
    paths = [ax.add_collection(LineCollection([], linewidths=1.6)) for _ in segments]
    places = ax.scatter(*tracks[0].T, s=14, color=PATH_COLOUR, zorder=3)
    # Lay the figure out once; only the paths and the places move between frames.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    for count in reached:
        for path, track, shade in zip(paths, segments, colours):
            path.set_segments(track[:count - 1])
            path.set_color(shade[:count - 1])
        places.set_offsets(tracks[count - 1])
        yield capture(figure)
    plt.close(figure)
