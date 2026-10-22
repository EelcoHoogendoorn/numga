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
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    fig.set_dpi(dpi)
    return np.asarray(image.reduce(SUPERSAMPLE))


def indexed(image: Image.Image, entries: np.ndarray, palette: Image.Image) -> Image.Image:
    """An RGB image in a palette `[entries, 3]`, each colour its nearest entry, exactly: Pillow's own mapping looks
    entries up at reduced precision, and moves a white background onto a near-white entry."""
    # Each colour as one number, red, green and blue a byte each, for a fast count of the distinct ones.
    codes, where = np.unique(np.asarray(image).reshape(-1, 3).astype(np.int32) @ np.array([1 << 16, 1 << 8, 1]),
                             return_inverse=True)
    colours = np.stack([codes >> 16, (codes >> 8) & 255, codes & 255], axis=-1)            # [colours, 3]
    distances = ((colours[:, None, :] - entries[None, :, :].astype(np.int32)) ** 2).sum(axis=-1)
    shown = Image.fromarray(distances.argmin(axis=1).astype(np.uint8)[where.reshape(-1)].reshape(image.height, image.width), "P")
    shown.putpalette(palette.getpalette())
    return shown


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
    sampled = np.concatenate([np.asarray(sample) for sample in samples], axis=0)
    palette = Image.fromarray(sampled).quantize(colors=colors)
    # Each entry snapped to the commonest colour among the pixels it stands for, rather than their mean, so that flat
    # colours, a white background above all, stay exact.
    entries = np.array(palette.getpalette(), dtype=np.uint8).reshape(-1, 3)
    pixels, chosen = sampled.reshape(-1, 3), np.asarray(palette).reshape(-1)
    for entry in np.unique(chosen):
        members, counts = np.unique(pixels[chosen == entry], axis=0, return_counts=True)
        entries[entry] = members[counts.argmax()]
    palette.putpalette(entries.reshape(-1).tolist())
    images = [indexed(image, entries, palette) for image in images]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], palette=palette.getpalette(),
        duration=duration_ms, loop=0, optimize=True,
    )
    return path


def save_figure(figure: plt.Figure, name: str) -> Path:
    """Save a figure as PLOT_DIR/<name>.png, or the next free <name>_NN.png."""
    path = auto_increment_path(PLOT_DIR / f"{name}.png")
    figure.savefig(path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {path}")
    return path


def save_animation(frames: list[np.ndarray], name: str, duration_ms: int, colors: int = 256, scale: float = 1.0) -> Path:
    """Save RGB frames as PLOT_DIR/<name>.gif, or the next free <name>_NN.gif, in a palette of `colors`,
    box-filtered to `scale` of their size."""
    path = auto_increment_path(PLOT_DIR / f"{name}.gif")
    save_gif(frames, str(path), duration_ms, scale, colors)
    print(f"Saved {path}")
    return path


def show_animation(frames: list[np.ndarray], name: str, duration_ms: int, colors: int = 256, scale: float = 1.0):
    """Save frames as an animation through `save_animation`, and return it for a notebook to show."""
    from IPython.display import Image as Shown

    return Shown(filename=str(save_animation(frames, name, duration_ms, colors, scale)))
