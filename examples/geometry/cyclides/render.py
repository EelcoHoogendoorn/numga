"""Headlight images of the traced surfaces: drawing only."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

BACKGROUND = np.array([0.013, 0.02, 0.036])
TEAL = np.array([0.055, 0.42, 0.39])
GOLD = np.array([0.6, 0.42, 0.08])
PALETTE = np.array([TEAL, [0.65, 0.38, 0.08], [0.35, 0.13, 0.40], [0.12, 0.35, 0.60]])   # per body, in turn


def headlight(facing: np.ndarray, visible: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Surface colour lit from the eye: facing is the cosine between the ray and the normal."""
    lit = color * (0.2 + 0.8 * np.clip(facing, 0, 1))[..., None] + 0.12 * np.clip(facing, 0, 1)[..., None] ** 16
    return np.where(visible[..., None], np.clip(lit, 0, 1) ** (1 / 2.2), BACKGROUND ** (1 / 2.2))


def facing_frames(traced: Iterable, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """Headlight images as RGB frames, one per traced (facing, angle) pair of one surface, as they are consumed."""
    for facing, angle in traced:
        rgb = headlight(facing.to_array(), np.isfinite(angle), TEAL).reshape(*shape, 3)
        yield (rgb * 255).round().astype(np.uint8)


def scene_frames(traced: Iterable, colors: np.ndarray, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """RGB frames of several surfaces, one per traced (facing, angle) pair over the bodies: each pixel shows the
    nearest hit among them, lit in that body's colour."""
    for facing, angle in traced:
        nearest = angle.argmin(axis=0)
        pixels = np.arange(angle.shape[1])
        facing = facing.to_array()[nearest, pixels]
        rgb = headlight(facing, np.isfinite(angle[nearest, pixels]), colors[nearest % len(colors)]).reshape(*shape, 3)
        yield (rgb * 255).round().astype(np.uint8)


def draw_facing(facing, angle: np.ndarray, shape: tuple[int, int]) -> plt.Figure:
    """Headlight images, one panel per surface along the leading axis of a Scalar facing; angle is inf on a miss."""
    rgb = headlight(facing.to_array(), np.isfinite(angle), TEAL).reshape(-1, *shape, 3)
    figure, axes = plt.subplots(1, len(rgb), figsize=(4.5 * len(rgb), 3.6), squeeze=False)
    for ax, image in zip(axes[0], rgb):
        ax.imshow(image)
        ax.axis("off")
    figure.tight_layout()
    return figure


def draw_images(images: list[np.ndarray], columns: int) -> plt.Figure:
    """RGB images in a grid of the given number of columns."""
    rows = -(-len(images) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image in zip(axes.flat, images):
        ax.imshow(image)
    figure.tight_layout()
    return figure
