"""Frame capture and GIF export shared by the animated examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def capture(fig: plt.Figure) -> np.ndarray:
    """Render a figure and return its RGB pixels."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames: list[np.ndarray], path: str, duration_ms: int, scale: float = 1.0, colors: int = 64, verbose: bool = False) -> str:
    """Write RGB frames as a looping palette GIF, optionally box-filtered down by scale."""
    images = [Image.fromarray(frame) for frame in frames]
    if scale != 1.0:
        size = (round(images[0].width * scale), round(images[0].height * scale))
        images = [image.resize(size, resample=Image.Resampling.BOX) for image in images]
    palette = Image.fromarray(np.concatenate([np.asarray(image) for image in images], axis=0)).quantize(colors=colors)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], palette=palette.getpalette(),
        duration=duration_ms, loop=0, optimize=True,
    )
    if verbose:
        print(f"Animated GIF exported to {path}")
    return path
