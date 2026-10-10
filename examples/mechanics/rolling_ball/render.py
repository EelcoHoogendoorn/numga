"""Drawing for the rolling ball: the table seen from above, with the loop and the ball on it; the ball
painted with a pattern fixed to it, and with the trail its contact point has left on it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture

TABLE = np.array([0.93, 0.91, 0.86])
LOOP = np.array([0.55, 0.53, 0.50])
# The ball's two pattern colours, one for each sign of the product of its three coordinates, the dark
# great circles where a coordinate vanishes, and the trail of its contact point.
LIGHT = np.array([0.96, 0.80, 0.30])
DARK = np.array([0.25, 0.45, 0.70])
CIRCLE = np.array([0.15, 0.15, 0.18])
TRAIL = np.array([0.85, 0.20, 0.20])
# At most this many trail points are painted, evenly spaced along it.
TRAIL_POINTS = 240


def disk(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres of a square view of the unit disk, seen from above: the two coordinates, each
    [rows, columns], the top row first."""
    centres = (np.arange(resolution) + 0.5) / resolution * 2 - 1
    u, v = np.meshgrid(centres, -centres)
    return u, v


def coordinates(points) -> np.ndarray:
    """The x, y and z coefficients of vectors, along the last axis."""
    return points.cast(points.algebra.subspace("x y z")).kernel


def ball(screen, body, trail) -> np.ndarray:
    """The visible half of the ball as RGBA pixels: each pixel coloured by the point of the ball under
    it, in the ball's own frame, lit from the upper left; transparent outside the ball."""
    view = coordinates(screen)
    fixed = coordinates(body)
    colours = np.where((np.prod(fixed, axis=-1) > 0)[..., None], LIGHT, DARK)
    circles = np.clip(1 - np.abs(fixed).min(axis=-1) / 0.04, 0.0, 1.0)
    colours = colours + circles[..., None] * (CIRCLE - colours)
    marks = coordinates(trail)
    marks = marks[np.linspace(0, len(marks) - 1, min(len(marks), TRAIL_POINTS)).astype(int)]
    # Squared distance to a mark is |p|^2 + |m|^2 - 2 p . m; the nearest mark minimizes the last two terms, the
    # cross term one matrix product over all pixels and marks.
    closest = ((marks ** 2).sum(axis=-1) + fixed @ (-2 * marks).T).min(axis=-1)
    nearest = np.sqrt(np.clip((fixed ** 2).sum(axis=-1) + closest, 0.0, None))
    colours = colours + np.clip(1 - nearest / 0.05, 0.0, 1.0)[..., None] * (TRAIL - colours)
    lit = 0.4 + 0.6 * np.clip(view @ np.array([-0.4, 0.4, 0.82]), 0.0, 1.0)
    inside = (view[..., :2] ** 2).sum(axis=-1) < 1
    return np.concatenate([np.clip(colours * lit[..., None], 0, 1), inside[..., None].astype(float)], axis=-1)


def scene(path, position, image: np.ndarray, radius: float) -> plt.Figure:
    """The table from above: the loop, and the ball's image at its position."""
    xy = coordinates(path)[:, :2]
    centre = coordinates(position)[:2]
    reach = np.abs(xy).max() + 1.3 * radius
    figure, ax = plt.subplots(figsize=(5, 5), facecolor=TABLE)
    ax.set_facecolor(TABLE)
    ax.plot(*xy.T, color=LOOP, linewidth=1.0)
    ax.imshow(image, extent=(centre[0] - radius, centre[0] + radius, centre[1] - radius, centre[1] + radius),
              interpolation="bilinear", zorder=3)
    ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal")
    ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    return figure


def animate(path, screen, rolling, radius: float) -> list[np.ndarray]:
    """One frame per state of the ball: its position, the ball under each pixel and its trail."""
    frames = []
    for position, body, trail in rolling:
        figure = scene(path, position, ball(screen, body, trail), radius)
        frames.append(capture(figure))
        plt.close(figure)
    return frames
