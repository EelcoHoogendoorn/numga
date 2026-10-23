"""Figure and animation output shared by the examples: the one place they write files.

Figures and animations go to PLOT_DIR under a new, incrementing name, so a run never
overwrites an earlier result. Each save prints the path it wrote and returns it.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from examples import PLOT_DIR, auto_increment_path

# How many times the figure's resolution a frame is drawn at, per axis, before each block is averaged.
SUPERSAMPLE = 2
PALETTE_FRAMES = 128
PALETTE_SIZE = 96
# The shortest time in seconds between the frames an animation shows as they come.
UPDATE_INTERVAL = 0.1


def capture(fig: plt.Figure) -> np.ndarray:
    """Render a figure at SUPERSAMPLE times its resolution and return its RGB pixels, each the average of
    its block: antialiasing that holds for every artist."""
    # A closed figure holds its pixels in reference cycles until the collector runs, which a frame's few objects and
    # large buffers rarely trigger: collected here, an animation keeps one frame's figure alive at a time.
    gc.collect()
    dpi = fig.dpi
    fig.set_dpi(dpi * SUPERSAMPLE)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    fig.set_dpi(dpi)
    return np.asarray(image.reduce(SUPERSAMPLE))


def codes(colours: np.ndarray) -> np.ndarray:
    """Each RGB colour `[..., 3]` bytes as one number, red, green and blue a byte each."""
    colours = colours.astype(np.uint32)
    return (colours[..., 0] << 16) | (colours[..., 1] << 8) | colours[..., 2]


def indexed(image: Image.Image, lookup: np.ndarray, palette: Image.Image) -> Image.Image:
    """An RGB image in a palette: Pillow's own mapping, which looks entries up at reduced precision, with every colour
    that is an entry exactly put on that entry, `lookup` giving each colour's entry or -1, so that a white
    background stays white."""
    shown = np.asarray(image.quantize(palette=palette, dither=Image.Dither.NONE))
    exact = lookup[codes(np.asarray(image))]                                  # [rows, columns] entry, or -1
    fixed = Image.fromarray(np.where(exact >= 0, exact, shown).astype(np.uint8), "P")
    fixed.putpalette(palette.getpalette())
    return fixed


def save_gif(frames: Iterable[np.ndarray], path: str, duration_ms: int, scale: float, colors: int) -> str:
    """Write RGB frames as a looping palette GIF, box-filtered by scale."""
    frames = list(frames)
    height, width = frames[0].shape[:2]
    size = (round(width * scale), round(height * scale))

    def resized(frame: np.ndarray) -> Image.Image:
        return Image.fromarray(frame).resize(size, resample=Image.Resampling.BOX)

    # Bound the palette population: very large repeated colour counts overflow Pillow's
    # quantizer accumulators. Sample the whole timeline while retaining source colours.
    indices = np.linspace(0, len(frames) - 1, min(len(frames), PALETTE_FRAMES), dtype=int)
    samples = [resized(frames[index]).resize((PALETTE_SIZE, PALETTE_SIZE), Image.Resampling.NEAREST)
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
    # Every colour's entry where it is one exactly, the first of equal entries, and -1 elsewhere.
    lookup = np.full(1 << 24, -1, dtype=np.int16)
    lookup[codes(entries)[::-1]] = np.arange(len(entries))[::-1]
    # Each frame resized and put in the palette in turn, so that only the indexed frames, a byte a pixel, are held.
    images = [indexed(resized(frame), lookup, palette) for frame in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], palette=palette.getpalette(),
        duration=duration_ms, loop=0,
    )
    return path


def save_figure(figure: plt.Figure, name: str) -> Path:
    """Save a figure as PLOT_DIR/<name>.png, or the next free <name>_NN.png."""
    path = auto_increment_path(PLOT_DIR / f"{name}.png")
    figure.savefig(path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {path}")
    return path


def forward_and_back(frames: list) -> list:
    """Frames played forward and then back, without repeating either end, so that the loop turns smoothly."""
    return frames + frames[-2:0:-1]


def save_animation(frames: Iterable[np.ndarray], name: str, duration_ms: int, colors: int = 256, scale: float = 1.0,
                   forward_then_back: bool = False) -> Path:
    """Save RGB frames as PLOT_DIR/<name>.gif, or the next free <name>_NN.gif, in a palette of `colors`,
    box-filtered to `scale` of their size; with `forward_then_back`, looping forward and then back."""
    path = auto_increment_path(PLOT_DIR / f"{name}.gif")
    frames = list(frames)
    save_gif(forward_and_back(frames) if forward_then_back else frames, str(path), duration_ms, scale, colors)
    print(f"Saved {path}")
    return path


def show_animation(frames: Iterable[np.ndarray], name: str, duration_ms: int, scale: float = 1.0, n_frames: int = 0,
                   forward_then_back: bool = False) -> None:
    """Show RGB frames in a notebook as they come, each in place of the one before with how many have come, then the
    whole animation looping, as the GIF it is saved as: PLOT_DIR/<name>.gif, or the next free <name>_NN.gif, as
    `save_animation` writes it, box-filtered to `scale` of their size. The count under each frame says how many are
    expected where `n_frames` is given. With `forward_then_back`, the loop plays the frames forward and then back."""
    import base64
    import io

    from IPython.display import HTML, display

    def embedded(image: Image.Image, kind: str, caption: str, **options) -> HTML:
        """An image as an image element, which every notebook front end shows, whatever image formats its kernel
        knows, with a line of text under it."""
        encoded = io.BytesIO()
        image.save(encoded, kind, **options)
        source = base64.b64encode(encoded.getvalue()).decode()
        return HTML(f'<img src="data:image/{kind.lower()};base64,{source}"><div>{caption}</div>')

    shown = display(HTML(""), display_id=True)
    images = []
    updated = -np.inf
    for frame in frames:
        image = Image.fromarray(frame)
        images.append(image.resize((round(image.width * scale), round(image.height * scale)), resample=Image.Resampling.BOX))
        # Frames that come faster than the front end redraws smoothly are kept but not shown.
        if time.monotonic() - updated < UPDATE_INTERVAL:
            continue
        updated = time.monotonic()
        count = f"frame {len(images)} of {n_frames}" if n_frames else f"frame {len(images)}"
        shown.update(embedded(images[-1], "JPEG", count, quality=85))
    if forward_then_back:
        images = forward_and_back(images)
    path = auto_increment_path(PLOT_DIR / f"{name}.gif")
    save_gif([np.asarray(image) for image in images], str(path), duration_ms, 1.0, 256)
    shown.update(HTML(f'<img src="data:image/gif;base64,{base64.b64encode(path.read_bytes()).decode()}">'))
    print(f"Saved {path}")
