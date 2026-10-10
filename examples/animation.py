"""Figure and animation output shared by the examples: the one place they write files.

Figures and animations go to PLOT_DIR under a new, incrementing name, so a run never
overwrites an earlier result. Each save prints the path it wrote and returns it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from examples import PLOT_DIR, auto_increment_path

# How many times the figure's resolution a frame is drawn at, per axis, before each block is averaged.
SUPERSAMPLE = 4
PALETTE_FRAMES = 128
PALETTE_SIZE = 96


def capture(fig: plt.Figure) -> np.ndarray:
    """Render a figure at SUPERSAMPLE times its resolution and return its RGB pixels, each the average of
    its block: antialiasing that holds for every artist."""
    dpi = fig.dpi
    fig.set_dpi(dpi * SUPERSAMPLE)
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)
    fig.set_dpi(dpi)
    rows, columns = pixels.shape[0] // SUPERSAMPLE, pixels.shape[1] // SUPERSAMPLE
    blocks = pixels[:rows * SUPERSAMPLE, :columns * SUPERSAMPLE].reshape(rows, SUPERSAMPLE, columns, SUPERSAMPLE, 3)
    return blocks.mean(axis=(1, 3)).round().astype(np.uint8)


def save_gif(frames: list[np.ndarray], path: str, duration_ms: int, scale: float, colors: int) -> str:
    """Write RGB frames as a looping palette GIF, box-filtered by scale."""
    images = [Image.fromarray(frame) for frame in frames]
    size = (round(images[0].width * scale), round(images[0].height * scale))
    images = [image.resize(size, resample=Image.Resampling.BOX) for image in images]
    # Bound the palette population: very large repeated colour counts overflow Pillow's
    # quantizer accumulators. Sample the whole timeline while retaining source colours.
    indices = np.linspace(0, len(images) - 1, min(len(images), PALETTE_FRAMES), dtype=int)
    samples = [images[index].resize((PALETTE_SIZE, PALETTE_SIZE), Image.Resampling.NEAREST)
               for index in indices]
    palette = Image.fromarray(np.concatenate([np.asarray(sample) for sample in samples], axis=0)).quantize(colors=colors)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], palette=palette.getpalette(),
        duration=duration_ms, loop=0, optimize=True,
    )
    return path


def save_figure(figure: plt.Figure, name: str) -> Path:
    """Save a figure as PLOT_DIR/<name>.png, or the next free <name>_NN.png."""
    path = auto_increment_path(PLOT_DIR / f"{name}.png")
    figure.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    return path


def save_animation(frames: list[np.ndarray], name: str, duration_ms: int, colors: int = 256, scale: float = 1.0) -> Path:
    """Save RGB frames as PLOT_DIR/<name>.gif, or the next free <name>_NN.gif, in a palette of `colors`,
    box-filtered to `scale` of their size."""
    path = auto_increment_path(PLOT_DIR / f"{name}.gif")
    save_gif(frames, str(path), duration_ms, scale, colors)
    print(f"Saved {path}")
    return path
