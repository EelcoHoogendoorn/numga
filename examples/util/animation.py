"""Figure and animation output shared by the examples: the one place they write files.

Figures and animations go to PLOT_DIR under a new, incrementing name, so a run never
overwrites an earlier result. Each save prints the path it wrote and returns it.

Frames are written at exactly their own pixels, never enlarged and never shrunk (hard rule): a smaller picture is a
picture rendered with fewer pixels. The one averaging is `capture`'s, supersampling within a frame's own pixels.

In a notebook, every picture, figure or animation, is shown at its pixel width's share of a column DESIGN_WIDTH
pixels wide, so that pictures keep their sizes relative to one another and to the column on any screen. Importing
this module in a notebook kernel shows matplotlib figures so; anywhere else it changes nothing.
"""

from __future__ import annotations

import base64
import gc
import io
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from examples import PLOT_DIR, auto_increment_path
from examples.util.charts import DESIGN_WIDTH

# How a figure is cropped to what it draws, saved and shown alike.
CROP = dict(bbox_inches="tight", pad_inches=0.02)
# How many times the figure's resolution a frame is drawn at, per axis, before each block is averaged.
SUPERSAMPLE = 2
PALETTE_FRAMES = 128
PALETTE_SIZE = 96
# The shortest time in seconds between the frames an animation shows as they come.
UPDATE_INTERVAL = 0.1
# The shortest frame duration GIF players keep: delays are stored in centiseconds and browsers draw shorter ones at
# about a tenth of a second. A smooth process is shown at 25 frames a second.
SHORTEST_DURATION_MS = 20
SMOOTH_DURATION_MS = 40


def check_frame_duration(duration_ms: int, stepped: bool) -> None:
    """Refuse a frame duration that does not fit its animation: a smooth process runs at SMOOTH_DURATION_MS, and a
    sequence of discrete steps, declared by `stepped`, at that or slower."""
    if duration_ms < SHORTEST_DURATION_MS:
        raise ValueError(f"a frame of {duration_ms} ms is shown at about 100 ms; {SHORTEST_DURATION_MS} ms is the shortest")
    if stepped and duration_ms < SMOOTH_DURATION_MS:
        raise ValueError(f"discrete steps are shown at {SMOOTH_DURATION_MS} ms a frame or slower, not {duration_ms} ms")
    if not stepped and duration_ms != SMOOTH_DURATION_MS:
        raise ValueError(f"a smooth process runs at {SMOOTH_DURATION_MS} ms a frame, not {duration_ms} ms: sample its "
                         f"frames over the same span at that rate, or pass stepped=True for a sequence of discrete steps")


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


def save_gif(frames: Iterable[np.ndarray], path: str, duration_ms: int, colors: int) -> str:
    """Write RGB frames as a looping palette GIF, at their own pixels."""
    frames = list(frames)

    # Bound the palette population: very large repeated colour counts overflow Pillow's
    # quantizer accumulators. Sample the whole timeline while retaining source colours.
    indices = np.linspace(0, len(frames) - 1, min(len(frames), PALETTE_FRAMES), dtype=int)
    samples = [Image.fromarray(frames[index]).resize((PALETTE_SIZE, PALETTE_SIZE), Image.Resampling.NEAREST)
               for index in indices]
    sampled = np.concatenate([np.asarray(sample) for sample in samples], axis=0)
    # One entry of the 256 left free, which Pillow takes for the pixels a frame leaves as they were, drawn transparent
    # over the frame before, so that what does not move costs next to nothing.
    palette = Image.fromarray(sampled).quantize(colors=min(colors, 255))
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
    # Each frame put in the palette in turn, so that only the indexed frames, a byte a pixel, are held.
    images = [indexed(Image.fromarray(frame), lookup, palette) for frame in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], palette=palette.getpalette(),
        duration=duration_ms, loop=0, optimize=True,
    )
    return path


def save_figure(figure: plt.Figure, name: str) -> Path:
    """Save a figure as PLOT_DIR/<name>.png, or the next free <name>_NN.png."""
    path = auto_increment_path(PLOT_DIR / f"{name}.png")
    figure.savefig(path, **CROP)
    print(f"Saved {path}")
    return path


def scaled_image(encoded: bytes, kind: str, width_px: int) -> str:
    """An image element for an encoded image `width_px` wide, shown at that width's share of a DESIGN_WIDTH column
    and never wider than the column."""
    source = base64.b64encode(encoded).decode()
    return (f'<img src="data:image/{kind};base64,{source}" '
            f'style="width: {100 * width_px / DESIGN_WIDTH:.1f}%; max-width: 100%">')


def forward_and_back(frames: list) -> list:
    """Frames played forward and then back, without repeating either end, so that the loop turns smoothly."""
    return frames + frames[-2:0:-1]


def save_animation(frames: Iterable[np.ndarray], name: str, duration_ms: int, colors: int = 256,
                   forward_then_back: bool = False, stepped: bool = False) -> Path:
    """Save RGB frames as PLOT_DIR/<name>.gif, or the next free <name>_NN.gif, in a palette of `colors`, at their own
    pixels; with `forward_then_back`, looping forward and then back. A smooth process is shown at
    SMOOTH_DURATION_MS a frame; `stepped` declares the frames a sequence of discrete steps (solver iterations, edge
    collapses, accumulating measurements), which may each be shown longer."""
    check_frame_duration(duration_ms, stepped)
    path = auto_increment_path(PLOT_DIR / f"{name}.gif")
    frames = list(frames)
    save_gif(forward_and_back(frames) if forward_then_back else frames, str(path), duration_ms, colors)
    print(f"Saved {path}")
    return path


def show_animation(frames: Iterable[np.ndarray], name: str, duration_ms: int, n_frames: int = 0,
                   forward_then_back: bool = False, stepped: bool = False) -> None:
    """Show RGB frames in a notebook as they come, each in place of the one before with how many have come, then the
    whole animation looping, as the GIF it is saved as: PLOT_DIR/<name>.gif, or the next free <name>_NN.gif, as
    `save_animation` writes it, at their own pixels. The count under each frame says how many are
    expected where `n_frames` is given. With `forward_then_back`, the loop plays the frames forward and then back.
    A smooth process is shown at SMOOTH_DURATION_MS a frame; `stepped` declares the frames a sequence of discrete
    steps, which may each be shown longer."""
    check_frame_duration(duration_ms, stepped)
    from IPython.display import HTML, display

    def embedded(image: Image.Image, kind: str, caption: str, **options) -> HTML:
        """An image as an image element, which every notebook front end shows, whatever image formats its kernel
        knows, with a line of text under it."""
        encoded = io.BytesIO()
        image.save(encoded, kind, **options)
        return HTML(f'{scaled_image(encoded.getvalue(), kind.lower(), image.width)}<div>{caption}</div>')

    shown = display(HTML(""), display_id=True)
    images = []
    updated = -np.inf
    for frame in frames:
        images.append(Image.fromarray(frame))
        # Frames that come faster than the front end redraws smoothly are kept but not shown.
        if time.monotonic() - updated < UPDATE_INTERVAL:
            continue
        updated = time.monotonic()
        count = f"frame {len(images)} of {n_frames}" if n_frames else f"frame {len(images)}"
        shown.update(embedded(images[-1], "PNG", count, compress_level=1))
    if forward_then_back:
        images = forward_and_back(images)
    path = auto_increment_path(PLOT_DIR / f"{name}.gif")
    save_gif([np.asarray(image) for image in images], str(path), duration_ms, 256)
    shown.update(HTML(scaled_image(path.read_bytes(), "gif", images[0].width)))
    print(f"Saved {path}")


def show_figure(figure: plt.Figure) -> None:
    """Show a figure in a notebook as the PNG `save_figure` writes, at its width's share of the column."""
    from IPython.display import HTML, display

    encoded = io.BytesIO()
    figure.savefig(encoded, format="png", **CROP)
    display(HTML(scaled_image(encoded.getvalue(), "png", Image.open(encoded).width)))


def show_figures_scaled() -> None:
    """In a notebook kernel, show every matplotlib figure with `show_figure`, in place of the PNG at its own pixels
    that the kernel shows otherwise: the figures the inline backend shows after a cell, and a figure that is a cell's
    value. Outside a kernel, in a script, a test or a terminal, nothing changes."""
    shell = sys.modules["IPython"].get_ipython() if "IPython" in sys.modules else None
    # Only a kernel's shell has a kernel. The formatter that displays an object itself is the one that the inline
    # backend leaves alone when it sets up its own image formatters for figures, whenever that happens.
    if getattr(shell, "kernel", None) is not None:
        shell.display_formatter.ipython_display_formatter.for_type(plt.Figure, show_figure)


show_figures_scaled()
