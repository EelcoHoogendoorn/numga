"""Shared figure and GIF output writes every example image and animation to `PLOT_DIR` while keeping
notebook display scaled to the shared column width.

`save_figure`, `save_animation`, `show_animation`, and `show_states` write the latest PNG or GIF to
`PLOT_DIR/<name>`, overwriting any earlier file of that name, and append a numbered copy to every
directory in `DUMP_DIRS` so development runs keep a chronological record. `pin_figure` and
`pin_animation` save in the same way and also overwrite the repository-tracked file under
`PINNED_DIR/<name>`, which `python -m examples.pinned` regenerates across the gallery.

Frames keep their rendered `[rows, columns, 3]` pixel dimensions without post-hoc resizing. `capture`
renders a matplotlib figure at `SUPERSAMPLE` times its `dpi` and box-averages back to the native
pixel grid while locking the figure's layout engine after the first draw so axes do not shift across
frames. `save_gif` builds a global palette from sampled per-frame palettes, assigns modal colours to
preserve flat backgrounds, and leaves one of the 256 GIF indices free for inter-frame transparency.

Smooth processes run at `SMOOTH_DURATION_MS` (`40` ms per frame, `25` frames per second), whereas
discrete-step sequences (`stepped=True`) run at `SMOOTH_DURATION_MS` or slower. Importing this module
inside an IPython kernel registers `show_figure` on `matplotlib.figure.Figure` so static figures and
looping GIFs alike scale to `width_px / DESIGN_WIDTH` of the notebook column.
"""

from __future__ import annotations

import base64
import gc
import io
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from examples import DUMP_DIRS, PINNED_DIR, PLOT_DIR, auto_increment_path
from examples.util.charts import DESIGN_WIDTH

# Tight bounding-box crop shared between saved PNGs and inline notebook figures.
CROP = dict(bbox_inches="tight", pad_inches=0.02)
# Per-axis oversampling factor drawn by `capture` before box-averaging back to native pixels.
SUPERSAMPLE = 2
PALETTE_FRAMES = 128
# Per-frame palette size sampled when building a shared GIF palette.
FRAME_COLOURS = 64
# Minimum wall-clock interval in seconds between progressive frame previews in a notebook.
UPDATE_INTERVAL = 0.1
# GIF delays store whole centiseconds and browsers clamp delays below 20 ms to roughly 100 ms;
# smooth motion runs at 40 ms per frame (25 frames per second).
SHORTEST_DURATION_MS = 20
SMOOTH_DURATION_MS = 40


def check_frame_duration(duration_ms: int, stepped: bool) -> None:
    """Smooth animations require `SMOOTH_DURATION_MS` per frame, while discrete-step sequences
    (`stepped=True`) accept `SMOOTH_DURATION_MS` or slower; sub-`SHORTEST_DURATION_MS` delays raise
    `ValueError`."""
    if duration_ms < SHORTEST_DURATION_MS:
        raise ValueError(f"a frame of {duration_ms} ms is shown at about 100 ms; {SHORTEST_DURATION_MS} ms is the shortest")
    if stepped and duration_ms < SMOOTH_DURATION_MS:
        raise ValueError(f"discrete steps are shown at {SMOOTH_DURATION_MS} ms a frame or slower, not {duration_ms} ms")
    if not stepped and duration_ms != SMOOTH_DURATION_MS:
        raise ValueError(f"a smooth process runs at {SMOOTH_DURATION_MS} ms a frame, not {duration_ms} ms: sample its "
                         f"frames over the same span at that rate, or pass stepped=True for a sequence of discrete steps")


# Identity of the figure most recently rendered by `capture`.
captured = {}


def capture(fig: plt.Figure) -> np.ndarray:
    """Drawing `fig` at `SUPERSAMPLE` times its `dpi` and box-averaging each `SUPERSAMPLE` by
    `SUPERSAMPLE` block produces an antialiased `[rows, columns, 3]` `uint8` RGB frame. Locking the
    layout engine after the first draw keeps axes stationary across animation frames."""
    # Closed figures retain large pixel buffers in reference cycles until garbage collection runs;
    # collecting whenever `fig` changes bounds memory for per-frame figures without taxing single-figure loops.
    if captured.get("figure") != id(fig):
        gc.collect()
        captured["figure"] = id(fig)
    dpi = fig.dpi
    fig.set_dpi(dpi * SUPERSAMPLE)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    fig.set_dpi(dpi)
    return np.asarray(image.reduce(SUPERSAMPLE))


def codes(colours: np.ndarray) -> np.ndarray:
    """Three `uint8` RGB channels of shape `[..., 3]` pack into 24-bit integer colour codes `[...]`."""
    colours = colours.astype(np.uint32)
    return (colours[..., 0] << 16) | (colours[..., 1] << 8) | colours[..., 2]


def indexed(image: Image.Image, lookup: np.ndarray, palette: Image.Image) -> Image.Image:
    """Exact 24-bit palette entries recorded in `lookup` (`>= 0`) override Pillow's reduced-precision
    quantization so colours present verbatim in `palette`, such as a white background, stay exact."""
    shown = np.asarray(image.quantize(palette=palette, dither=Image.Dither.NONE))
    exact = lookup[codes(np.asarray(image))]                                    # [..., ...]
    fixed = Image.fromarray(np.where(exact >= 0, exact, shown).astype(np.uint8), "P")
    fixed.putpalette(palette.getpalette())
    return fixed


def commonest(pixels: np.ndarray, chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Within each quantized cluster assigned in `chosen` (`[n_pixels]`), the modal RGB colour from
    `pixels` (`[n_pixels, 3]`) and the cluster's pixel count form arrays of shape `[n_entries, 3]`
    and `[n_entries]`."""
    keys = (chosen.astype(np.int64) << 24) | codes(pixels)
    unique, counts = np.unique(keys, return_counts=True)
    entries = unique >> 24
    # Sort each palette entry's candidate colours by descending pixel count to pick its modal RGB colour.
    first = np.lexsort((-counts, entries))
    leading = first[np.r_[True, entries[first][1:] != entries[first][:-1]]]
    code = unique[leading] & 0xFFFFFF
    colours = np.stack(((code >> 16) & 255, (code >> 8) & 255, code & 255), axis=-1).astype(np.uint8)
    return colours, np.bincount(np.searchsorted(entries[leading], entries), weights=counts)


def save_gif(frames: Iterable[np.ndarray], path: str, duration_ms: int, colors: int) -> str:
    """Quantizing `[rows, columns, 3]` RGB `frames` to a shared modal palette writes a looping GIF at
    `path` at the frames' native pixel dimensions."""
    frames = list(frames)
    if os.environ.get("NUMGA_NO_PLOT_WRITES") == "1":
        return path

    # Sampling per-frame palettes gives short transient flashes weight alongside colours present in
    # every frame, and picking each cluster's modal colour rather than its mean keeps flat backgrounds exact.
    indices = np.linspace(0, len(frames) - 1, min(len(frames), PALETTE_FRAMES), dtype=int)
    colours, weights = zip(*(commonest(frames[index].reshape(-1, 3),
                                       np.asarray(Image.fromarray(frames[index]).quantize(colors=FRAME_COLOURS)).reshape(-1))
                             for index in indices))
    colours, merged = np.unique(np.concatenate(colours), axis=0, return_inverse=True)
    weights = np.bincount(merged.reshape(-1), weights=np.concatenate(weights))
    # Leave one of the 256 GIF palette slots free so Pillow can encode unchanged pixels transparently across frames.
    palette = Image.fromarray(colours[None]).quantize(colors=min(colors, 255))
    chosen = np.asarray(palette).reshape(-1)
    entries = np.array(palette.getpalette(), dtype=np.uint8).reshape(-1, 3)
    for entry in np.unique(chosen):
        members = np.flatnonzero(chosen == entry)
        entries[entry] = colours[members[weights[members].argmax()]]
    palette.putpalette(entries.reshape(-1).tolist())
    # Map every exact 24-bit palette colour to its first matching entry, and all other colours to -1.
    lookup = np.full(1 << 24, -1, dtype=np.int16)
    lookup[codes(entries)[::-1]] = np.arange(len(entries))[::-1]
    # Convert frames one at a time so only 1-byte-per-pixel indexed images remain in memory together.
    images = [indexed(Image.fromarray(frame), lookup, palette) for frame in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], palette=palette.getpalette(),
        duration=duration_ms, loop=0, optimize=True,
    )
    return path


def kept(path: Path) -> Path:
    """Each directory in `DUMP_DIRS` receives an auto-numbered copy of `path` before `path` is
    printed and returned."""
    if os.environ.get("NUMGA_NO_PLOT_WRITES") == "1":
        return path
    for dump in DUMP_DIRS:
        dump.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, auto_increment_path(dump / path.name))
    print(f"Saved {path}")
    return path


def save_figure(figure: plt.Figure, name: str) -> Path:
    """Saves `figure` to `PLOT_DIR/<name>.png` and records a numbered copy in each `DUMP_DIRS`
    folder."""
    path = PLOT_DIR / f"{name}.png"
    if os.environ.get("NUMGA_NO_PLOT_WRITES") == "1":
        plt.close(figure)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, **CROP)
    return kept(path)


def pin_figure(figure: plt.Figure, name: str) -> Path:
    """Saves `figure` via `save_figure` and copies the PNG over the tracked `PINNED_DIR/<name>.png`."""
    pinned = PINNED_DIR / f"{name}.png"
    if os.environ.get("NUMGA_NO_PLOT_WRITES") == "1":
        plt.close(figure)
        return pinned
    pinned.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(save_figure(figure, name), pinned)
    print(f"Pinned {pinned}")
    return pinned


def scaled_image(encoded: bytes, kind: str, width_px: int) -> str:
    """Base64 image data of format `kind` and pixel width `width_px` formats as an HTML `<img>` tag
    sized to `width_px / DESIGN_WIDTH` of the notebook column (clamped at `100%`)."""
    source = base64.b64encode(encoded).decode()
    return (f'<img src="data:image/{kind};base64,{source}" '
            f'style="width: {100 * width_px / DESIGN_WIDTH:.1f}%; max-width: 100%">')


def forward_and_back(frames: list) -> list:
    """Appending the interior frames in reverse order plays `frames` forward and back without
    doubling either endpoint."""
    return frames + frames[-2:0:-1]


def save_animation(frames: Iterable[np.ndarray], name: str, duration_ms: int, colors: int = 256,
                   forward_then_back: bool = False, stepped: bool = False) -> Path:
    """Saves `[rows, columns, 3]` RGB `frames` to `PLOT_DIR/<name>.gif` in a palette of `colors` and
    records a numbered copy in `DUMP_DIRS`.

    Smooth processes require `SMOOTH_DURATION_MS` per frame, while `stepped=True` permits longer
    per-frame durations for discrete solver steps or edge collapses; `forward_then_back` bounces the
    loop smoothly without repeating endpoints.
    """
    check_frame_duration(duration_ms, stepped)
    path = PLOT_DIR / f"{name}.gif"
    frames = list(frames)
    save_gif(forward_and_back(frames) if forward_then_back else frames, str(path), duration_ms, colors)
    return kept(path)


def pin_animation(frames: Iterable[np.ndarray], name: str, duration_ms: int, colors: int = 256,
                  forward_then_back: bool = False, stepped: bool = False) -> Path:
    """Saves `frames` via `save_animation` and copies the GIF over the tracked
    `PINNED_DIR/<name>.gif`."""
    pinned = PINNED_DIR / f"{name}.gif"
    saved = save_animation(frames, name, duration_ms, colors, forward_then_back, stepped)
    if os.environ.get("NUMGA_NO_PLOT_WRITES") == "1":
        return pinned
    pinned.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(saved, pinned)
    print(f"Pinned {pinned}")
    return pinned


def show_animation(frames: Iterable[np.ndarray], name: str, duration_ms: int, n_frames: int = 0,
                   forward_then_back: bool = False, stepped: bool = False, colors: int = 256) -> None:
    """Streams `[rows, columns, 3]` RGB `frames` progressively into a notebook cell with a live frame
    counter, writes the finished looping GIF to `PLOT_DIR/<name>.gif`, and replaces the preview with
    the saved GIF scaled to the notebook column.

    Passing `n_frames` displays `frame k of n_frames` during generation, `forward_then_back` bounces
    the loop without repeating endpoints, and `stepped=True` allows frame durations above
    `SMOOTH_DURATION_MS` for discrete-step sequences.
    """
    check_frame_duration(duration_ms, stepped)
    from IPython.display import HTML, display

    def embedded(image: Image.Image, kind: str, caption: str, **options) -> HTML:
        encoded = io.BytesIO()
        image.save(encoded, kind, **options)
        return HTML(f'{scaled_image(encoded.getvalue(), kind.lower(), image.width)}<div>{caption}</div>')

    shown = display(HTML(""), display_id=True)
    images = []
    updated = -np.inf
    for frame in frames:
        images.append(Image.fromarray(frame))
        # Throttle live notebook previews to `UPDATE_INTERVAL` while still recording every frame for the GIF.
        if time.monotonic() - updated < UPDATE_INTERVAL:
            continue
        updated = time.monotonic()
        count = f"frame {len(images)} of {n_frames}" if n_frames else f"frame {len(images)}"
        shown.update(embedded(images[-1], "PNG", count, compress_level=1))
    if forward_then_back:
        images = forward_and_back(images)
    if os.environ.get("NUMGA_NO_PLOT_WRITES") == "1":
        return
    path = PLOT_DIR / f"{name}.gif"
    save_gif([np.asarray(image) for image in images], str(path), duration_ms, colors)
    shown.update(HTML(scaled_image(kept(path).read_bytes(), "gif", images[0].width)))


def show_states(states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments, name: str, duration_ms: int,
                n_frames: int = 0, stepped: bool = False, colors: int = 256) -> list:
    """Streams `states` once through `animate(*arguments, states)` into `show_animation` while
    collecting and returning the list of states for downstream readouts or checks."""
    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    show_animation(animate(*arguments, passing()), name, duration_ms, n_frames=n_frames, stepped=stepped,
                   colors=colors)
    return kept


def show_figure(figure: plt.Figure) -> None:
    """Displays `figure` in a notebook as a cropped PNG scaled to `width_px / DESIGN_WIDTH` of the
    column."""
    from IPython.display import HTML, display

    encoded = io.BytesIO()
    figure.savefig(encoded, format="png", **CROP)
    display(HTML(scaled_image(encoded.getvalue(), "png", Image.open(encoded).width)))


def show_figures_scaled() -> None:
    """Registers `show_figure` on `matplotlib.figure.Figure` when running inside an IPython kernel,
    leaving script, test, and terminal sessions unchanged."""
    shell = sys.modules["IPython"].get_ipython() if "IPython" in sys.modules else None
    # Only an active kernel exposes `shell.kernel`; registering on `ipython_display_formatter` takes
    # precedence over the inline backend's raw-pixel PNG formatter.
    if getattr(shell, "kernel", None) is not None:
        shell.display_formatter.ipython_display_formatter.for_type(plt.Figure, show_figure)


show_figures_scaled()
