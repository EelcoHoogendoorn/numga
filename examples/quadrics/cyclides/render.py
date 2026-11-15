"""Headlight shading and panel tiling for ray-traced Dupin cyclides."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.util.panels import image_figure

background_rgb = np.array([0.013, 0.02, 0.036])
teal_rgb = np.array([0.055, 0.42, 0.39])
gold_rgb = np.array([0.6, 0.42, 0.08])
panel_gap = 4
body_palette = np.array([teal_rgb, [0.65, 0.38, 0.08], [0.35, 0.13, 0.40], [0.12, 0.35, 0.60]])
linked_palette = np.stack([teal_rgb, gold_rgb])


def headlight_shading(facing: np.ndarray, visible: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Headlight cosines `facing` shade visible pixels in `color` over `background_rgb`."""
    clamped = np.clip(facing, 0, 1)[..., None]
    lit = color * (0.2 + 0.8 * clamped) + 0.12 * clamped ** 16
    return np.where(visible[..., None], np.clip(lit, 0, 1) ** (1 / 2.2), background_rgb ** (1 / 2.2))


def facing_animation(image_shape: tuple[int, int], states: Iterable) -> Iterator[np.ndarray]:
    """Yields `uint8` RGB frames of shape `[*image_shape, 3]` from single-surface `(facing, angle)`
    states."""
    for facing, angle in states:
        rgb = headlight_shading(facing.to_array(), angle.isfinite(), teal_rgb).reshape(*image_shape, 3)
        yield (rgb * 255).round().astype(np.uint8)


def scene_animation(image_shape: tuple[int, int], colors: np.ndarray, states: Iterable) -> Iterator[np.ndarray]:
    """Yields `uint8` RGB frames of shape `[*image_shape, 3]` keeping the nearest hit across bodies
    in `colors`."""
    for facing, angle in states:
        nearest = angle.argmin(axis=0)
        pixel_indices = np.arange(angle.shape[1])
        chosen_facing = facing[nearest, pixel_indices]                         # [...] Scalar+
        chosen_angle = angle[nearest, pixel_indices]                           # [...] Scalar+
        rgb = headlight_shading(chosen_facing.to_array(), chosen_angle.isfinite(),
                                colors[nearest % len(colors)]).reshape(*image_shape, 3)
        yield (rgb * 255).round().astype(np.uint8)


def linked_animation(image_shape: tuple[int, int], states: Iterable) -> Iterator[np.ndarray]:
    """Yields `uint8` RGB frames of a teal cyclide linked with a gold ring."""
    return scene_animation(image_shape, linked_palette, states)


def tiled_grid(images: list[np.ndarray], n_columns: int) -> np.ndarray:
    """Same-sized RGB `images` arrange into a grid of `n_columns` columns separated by `panel_gap`
    white pixels."""
    n_rows = -(-len(images) // n_columns)
    height, width = images[0].shape[:2]
    grid = np.full((n_rows * (height + panel_gap) - panel_gap,
                    n_columns * (width + panel_gap) - panel_gap, 3), 255, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, n_columns)
        top, left = row * (height + panel_gap), column * (width + panel_gap)
        grid[top:top + height, left:left + width] = image
    return grid


def facing_figure(facing, angle, image_shape: tuple[int, int]) -> plt.Figure:
    """Each surface along the leading axis of `facing` renders side by side at native pixels."""
    rgb = headlight_shading(facing.to_array(), angle.isfinite(), teal_rgb).reshape(-1, *image_shape, 3)
    images = list((rgb * 255).round().astype(np.uint8))
    return image_figure(tiled_grid(images, len(images)))


def gallery_figure(images: list[np.ndarray], n_columns: int) -> plt.Figure:
    """RGB `images` display in a native-pixel grid of `n_columns` columns."""
    return image_figure(tiled_grid(images, n_columns))
