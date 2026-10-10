"""Drawing for the reflection notebooks: a kaleidoscope painted from its folded points, a sphere and
disks of tiles coloured by how many reflections folded each pixel, and the 600-cell projected from
the three-sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from examples.animation import capture

BACKGROUND = np.array([0.06, 0.07, 0.09])
# The two tiles of a reflection: one colour for an even number of reflections, one for an odd number.
EVEN = np.array([0.93, 0.83, 0.62])
ODD = np.array([0.35, 0.55, 0.72])
LINE = np.array([0.12, 0.12, 0.16])
# Beads tumbling in the kaleidoscope, scattered over the disk: centre radius, angle, speed and size of
# each, and its colour, drawn from a short palette.
_beads = np.random.default_rng(7)
BEADS = np.stack([
    np.sqrt(_beads.uniform(0.0, 1.0, 40)), _beads.uniform(0.0, 2 * np.pi, 40),
    _beads.choice([-2, -1, 1, 2], 40), _beads.uniform(0.04, 0.13, 40),
], axis=-1)
PALETTE = np.array([
    [0.91, 0.30, 0.24], [0.98, 0.77, 0.19], [0.18, 0.62, 0.48], [0.56, 0.27, 0.68],
    [0.20, 0.47, 0.85], [0.95, 0.55, 0.65], [0.99, 0.97, 0.90],
])
BEAD_COLOURS = PALETTE[np.arange(len(BEADS)) % len(PALETTE)]


def disk(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres of a square view of the unit disk: the two coordinates, each [rows, columns]."""
    centres = (np.arange(resolution) + 0.5) / resolution * 2 - 1
    u, v = np.meshgrid(centres, -centres)
    return u, v


def coordinates(points, blades: str) -> np.ndarray:
    """The coefficients of points on the named blades, in that order along the last axis."""
    return points.cast(points.algebra.subspace(blades)).kernel


def to_image(colours: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] to 8-bit pixels."""
    return (np.clip(colours, 0.0, 1.0) * 255).round().astype(np.uint8)


def side_by_side(images: np.ndarray, gap: int = 6) -> np.ndarray:
    """Panels along the leading axis joined left to right, with a gap of background between them."""
    spacer = np.broadcast_to(to_image(BACKGROUND), (images.shape[1], gap, 3))
    return np.concatenate([part for image in images for part in (image, spacer)][:-1], axis=1)


# --- kaleidoscope -----------------------------------------------------------------------
def kaleidoscope(folded, time: float) -> np.ndarray:
    """The beads at the given time, painted at every folded point: [..., rows, columns, 3] pixels."""
    xy = coordinates(folded, "x y").astype(np.float32)                         # [..., rows, columns, 2]
    # The beads from the top down: each is painted over the ones listed before it.
    radius, angle, speed, size = BEADS[::-1].T.astype(np.float32)
    centres = radius[:, None] * np.stack([np.cos(angle + speed * time), np.sin(angle + speed * time)], axis=-1)
    # Squared distances to the bead centres through one matrix product over all pixels.
    squared = (xy ** 2).sum(axis=-1, keepdims=True) - 2 * xy @ centres.T + (centres ** 2).sum(axis=-1)
    distance = np.sqrt(np.maximum(squared, 0.0))                               # [..., rows, columns, beads]
    cover = np.clip((size - distance) / 0.02, 0.0, 1.0)
    # What shows of a bead is its cover times what the beads above it leave uncovered, and the background
    # shows where all of them leave it uncovered.
    uncovered = np.cumprod(1 - cover, axis=-1)                                 # [..., rows, columns, beads]
    shown = cover * np.concatenate([np.ones_like(uncovered[..., :1]), uncovered[..., :-1]], axis=-1)
    return to_image(uncovered[..., -1:] * BACKGROUND + shown @ BEAD_COLOURS[::-1])


def animate_kaleidoscope(folded, frames: int) -> list[np.ndarray]:
    """The beads tumbling once around, each frame the kaleidoscopes side by side."""
    return [side_by_side(kaleidoscope(folded, time)) for time in np.linspace(0, 2 * np.pi, frames, endpoint=False)]


# --- tiles ------------------------------------------------------------------------------
def tiles(flips, nearness: np.ndarray) -> np.ndarray:
    """Colours of tiles by the parity of their reflections, darkened along the mirrors, where the
    nearness to the closest mirror runs from one on it to zero away from it."""
    odd = (np.round(flips.to_array()) % 2)[..., None]
    colours = EVEN + odd * (ODD - EVEN)
    return colours + nearness[..., None] * (LINE - colours)


def sphere(screen, folded, flips, mirrors, outline) -> np.ndarray:
    """The visible half of a sphere tiled by reflections, lit from the viewer's upper left, with the
    tiles' sides drawn thin and the sides on the outlining mirror, one of the mirrors, drawn bold."""
    view = coordinates(screen, "x y z")
    lit = 0.35 + 0.65 * np.clip(view @ np.array([-0.4, 0.5, 0.77]), 0.0, 1.0)
    sides = np.abs((mirrors | folded[..., None]).to_array())                  # [rows, columns, mirrors]
    thin = np.clip(1 - sides.min(axis=-1) / 0.012, 0.0, 1.0)
    bold = np.clip(1 - np.abs((outline | folded).to_array()) / 0.03, 0.0, 1.0)
    colours = tiles(flips, np.maximum(0.6 * thin, bold)) * lit[..., None]
    inside = (view[..., :2] ** 2).sum(axis=-1) < 1
    return to_image(np.where(inside[..., None], colours, BACKGROUND))


def animate_sphere(screen, turning, mirrors, outline) -> list[np.ndarray]:
    """One frame per folded view of the turning sphere."""
    return [sphere(screen, folded, flips, mirrors, outline) for folded, flips in turning]


def tiling(folded, flips, mirrors) -> np.ndarray:
    """A disk of tiles, the sides drawn where a folded point pairs with a mirror close to zero, relative
    to its pairings with all three."""
    sides = np.abs((mirrors & folded[..., None]).to_array())                  # [rows, columns, mirrors]
    nearness = np.clip(1 - sides.min(axis=-1) / sides.sum(axis=-1) / 0.02, 0.0, 1.0)
    u, v = disk(folded.shape[-1])
    inside = (u ** 2 + v ** 2 < 1)[..., None] & np.isfinite(nearness)[..., None]
    return to_image(np.where(inside, tiles(flips, np.nan_to_num(nearness)), BACKGROUND))


# --- the 600-cell -----------------------------------------------------------------------
def distinct(points: np.ndarray) -> np.ndarray:
    """The distinct rows of an array of points, up to round-off: one of each set of rows equal to six
    decimals, in sorted order."""
    rounded = np.round(points * 1e6).astype(np.int64)                          # [points, coordinates]
    # Equal rows sort next to each other, the first of them first; a row starts a new set where it differs
    # from the one before.
    order = np.lexsort(rounded.T[::-1])
    first = order[np.concatenate([[True], (np.diff(rounded[order], axis=0) != 0).any(axis=-1)])]
    return points[np.sort(first)]


def polytope(spinors, tilt: float = 0.45) -> plt.Figure:
    """The distinct unit spinors as points of the three-sphere, in perspective from beyond the scalar
    axis into three dimensions and from there seen from slightly above; edges join nearest neighbours,
    drawn lighter the nearer they are."""
    points = distinct(coordinates(spinors, "1 yz zx xy"))                      # [vertices, 4]
    gaps = np.linalg.norm(points[:, None] - points[None], axis=-1)
    nearest = gaps[gaps > 1e-9].min()
    first, second = np.nonzero(np.triu(np.abs(gaps - nearest) < 1e-6))
    space = points[:, 1:] / (2.4 - points[:, :1])                              # [vertices, 3]
    turn = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    seen = space @ turn.T
    depth = (seen[first, 2] + seen[second, 2]) / 2
    shade = (depth - depth.min()) / np.ptp(depth)
    order = np.argsort(depth)
    shade = shade[order, None]                                                 # [edges, 1]
    colours = np.concatenate([EVEN + (1 - shade) * (ODD - EVEN), 0.35 + 0.65 * shade], axis=-1)   # [edges, 4]
    segments = np.stack([seen[first[order], :2], seen[second[order], :2]], axis=1)   # [edges, 2, 2]
    figure, ax = plt.subplots(figsize=(5, 5), facecolor=BACKGROUND)
    ax.add_collection(LineCollection(segments, colors=colours, linewidths=0.6 + 1.2 * shade[:, 0], capstyle="projecting"))
    ax.set(xlim=(-0.75, 0.75), ylim=(-0.75, 0.75), aspect="equal")
    ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    return figure


def animate_polytope(turning) -> list[np.ndarray]:
    """One frame per batch of turned spinors."""
    frames = []
    for spinors in turning:
        figure = polytope(spinors)
        frames.append(capture(figure))
        plt.close(figure)
    return frames


def animate_tilings(moving) -> list[np.ndarray]:
    """One frame per step of the motion: the disks of tiles of every geometry side by side."""
    return [side_by_side(np.stack([tiling(*geometry) for geometry in step])) for step in moving]
