"""Headlight images of the traced surfaces: drawing only."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

BACKGROUND = np.array([0.013, 0.02, 0.036])


def headlight(facing: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Surface colour lit from the eye: facing is the cosine between the ray and the normal."""
    color = np.array([0.055, 0.42, 0.39]) * (0.2 + 0.8 * np.clip(facing, 0, 1))[..., None] + 0.12 * np.clip(facing, 0, 1)[..., None] ** 16
    return np.where(visible[..., None], np.clip(color, 0, 1) ** (1 / 2.2), BACKGROUND ** (1 / 2.2))


def facing_frames(traced: Iterable, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """Headlight images as RGB frames, one per traced (facing, visible) pair, as they are consumed."""
    for facing, visible in traced:
        rgb = headlight(facing.to_array(), visible).reshape(*shape, 3)
        yield (rgb * 255).round().astype(np.uint8)


def draw_facing(facing, visible: np.ndarray, shape: tuple[int, int]) -> plt.Figure:
    """Headlight images, one panel per surface along the leading axis of a Scalar facing."""
    rgb = headlight(facing.to_array(), visible).reshape(-1, *shape, 3)
    figure, axes = plt.subplots(1, len(rgb), figsize=(4.5 * len(rgb), 3.6), squeeze=False)
    for ax, image in zip(axes[0], rgb):
        ax.imshow(image)
        ax.axis("off")
    figure.tight_layout()
    return figure
