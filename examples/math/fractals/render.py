"""Drawing for the fractals notebook: escape counts coloured smoothly, the points that stayed bounded
black, as still images and as an animation of slices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

BACKGROUND = np.array([0.02, 0.02, 0.04])
# The colours escaping points run through as they take longer to escape, repeating.
BANDS = np.array([
    [0.05, 0.07, 0.25], [0.12, 0.35, 0.65], [0.55, 0.80, 0.90], [0.98, 0.93, 0.70],
    [0.95, 0.60, 0.20], [0.55, 0.15, 0.25], [0.05, 0.07, 0.25],
])
# Escape counts per pass through the colours.
PERIOD = 24


def grid(lower: tuple[float, float], upper: tuple[float, float], rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres of a view: the horizontal and vertical coordinates, each [rows, columns], the top
    row first."""
    across = lower[0] + (np.arange(columns) + 0.5) / columns * (upper[0] - lower[0])
    down = upper[1] - (np.arange(rows) + 0.5) / rows * (upper[1] - lower[1])
    return np.meshgrid(across, down)


def escape_image(points, count, iterations: int) -> np.ndarray:
    """Pixels coloured by how long their points took to escape, smoothed by how far past the circle of
    radius two the last step took them; black where they never escaped."""
    steps = count.to_array()
    size = np.sqrt(np.maximum(points.scalar_norm_squared().to_array(), 4.0))
    smooth = steps + 1 - np.log2(np.log2(size))
    position = (smooth / PERIOD) % 1 * (len(BANDS) - 1)
    lower = np.floor(position).astype(int)
    fraction = (position - lower)[..., None]
    colours = BANDS[lower] + fraction * (BANDS[lower + 1] - BANDS[lower])
    bounded = steps > iterations - 0.5
    return (np.clip(np.where(bounded[..., None], BACKGROUND, colours), 0, 1) * 255).round().astype(np.uint8)


def show(image: np.ndarray, lower: tuple[float, float], upper: tuple[float, float]) -> plt.Figure:
    """An image over its view, without axes."""
    height, width = image.shape[:2]
    figure, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image, extent=(lower[0], upper[0], lower[1], upper[1]), interpolation="nearest")
    ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    return figure


def animate(slices, iterations: int) -> list[np.ndarray]:
    """One frame per slice of points and escape counts."""
    return [escape_image(points, count, iterations) for points, count in slices]
