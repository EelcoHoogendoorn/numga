"""Patterns shaders in the numga dialect share, as WGSL to append to a shader's own.

`supersampled` makes the image: a shader that defines what one sample sees, `fn sample(place:
vec2<f32>) -> Sample`, with `Sample` any type that adds and scales (a colour, `alias Sample =
vec3<f32>;`, or a struct of several), gets an `image` of the average over a grid of samples in each pixel,
and `screen(place)`, the place's direction across the view.
"""

from __future__ import annotations

import numpy as np


def software() -> bool:
    """Whether the adapter is the CPU itself, a software driver as in a notebook's runtime without a GPU."""
    from examples.wgsl.runner import device
    return device().adapter.info["adapter_type"] == "CPU"


# The samples a side of every pixel, for every picture: sixteen to a pixel on a GPU, four on the CPU, where sixteen
# cost too much; and the share of a traced picture's pixels drawn across, all of them on a GPU, half on the CPU.
samples_per_side = 2 if software() else 4
pixel_share = 0.5 if software() else 1.0


def drawn(pixels: int) -> int:
    """The pixels a traced picture draws across where `pixels` are asked for: all of them on a GPU, `pixel_share` of
    them where the CPU draws."""
    return round(pixels * pixel_share)


def supersampled() -> str:
    """WGSL for an image of `samples_per_side` squared samples a pixel, on a grid turned so that no two
    share a row or a column, which resolves near-horizontal and near-vertical edges as finely as a grid
    of that many to a side."""
    count = samples_per_side**2
    offsets = [((across * samples_per_side + down + 0.5) / count - 0.5,
                (down * samples_per_side + samples_per_side - 1 - across + 0.5) / count - 0.5)
               for across in range(samples_per_side) for down in range(samples_per_side)]
    entries = ", ".join(f"vec2<f32>({x!r}, {y!r})" for x, y in offsets)
    return f"""
// The image's size, columns then rows, and each pixel's average of what its samples see.
var<uniform> pixels: vec2<u32>;
var<storage, read_write> image: array<Sample>;

// Where a pixel's samples lie, in pixel widths from its centre: one in each cell of a grid `samples_per_side` across.
const samples_per_side = {samples_per_side}u;
const n_samples = {count}u;
const samples = array<vec2<f32>, {count}>({entries});

// A place on the image, in pixels from its top left, as a direction across the view: right and up, in
// half heights of the image from its centre.
fn screen(place: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>((place.x * 2 - f32(pixels.x)) / f32(pixels.y), 1 - place.y * 2 / f32(pixels.y));
}}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= pixels.x || id.y >= pixels.y) {{
        return;
    }}
    var total = sample(vec2<f32>(id.xy) + 0.5 + samples[0]) / f32(n_samples);
    for (var index = 1u; index < n_samples; index++) {{
        total += sample(vec2<f32>(id.xy) + 0.5 + samples[index]) / f32(n_samples);
    }}
    image[id.y * pixels.x + id.x] = total;
}}
"""


def compose(layers: list[tuple[np.ndarray, np.ndarray, float]], background: np.ndarray) -> np.ndarray:
    """Layers blended over a background, nearest first at each pixel, `[rows, columns, 3]` bytes: each layer its
    colour and its depth along the view, `[rows, columns, 4]` and `[rows, columns]`, both weighted by how much of
    the pixel it covers, and its opacity."""
    colours = np.stack([colour * opacity for colour, _, opacity in layers])          # [layers, rows, columns, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        depths = np.stack([np.where(colour[..., 3] > 0, depth / colour[..., 3], np.inf) for colour, depth, _ in layers])
    order = np.argsort(depths, axis=0)
    shown = np.zeros(colours.shape[1:3] + (3,))
    passing = np.ones(colours.shape[1:3] + (1,))
    for nearest in np.take_along_axis(colours, order[..., None], axis=0):
        shown += passing * nearest[..., :3]
        passing *= 1 - nearest[..., 3:]
    return np.round(np.clip(shown + passing * background, 0, 1) * 255).astype(np.uint8)


def vec3(rgb: np.ndarray) -> str:
    """A colour as a WGSL constant."""
    return f"vec3<f32>({', '.join(f'{value:.6f}' for value in rgb)})"


def show(ax, image: np.ndarray, view):
    """An image of a view in matplotlib axes, its middle at the origin, in the scene's units, so that places from
    `view.across` land where they are seen; without axes. The image, for an animation to set its data."""
    shown = ax.imshow(image, extent=view.extent, interpolation="bilinear")
    ax.set_axis_off()
    return shown


def labels(ax, view, places, texts: list[str], **style) -> None:
    """Texts at points, of any weight, as a view sees them, over its image shown by `show`."""
    for (x, y), text in zip(view.across(places), texts):
        ax.text(x, y, text, **style)


def image_figure(image: np.ndarray):
    """A figure of an image at its pixel size, without axes or margins."""
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image)
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure
