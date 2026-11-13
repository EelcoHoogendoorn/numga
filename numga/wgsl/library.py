"""Patterns shaders in the numga dialect share, as WGSL to append to a shader's own.

`supersampled` makes the image: a shader that defines what one sample sees, `fn sample(place:
vec2<f32>) -> Sample`, with `Sample` any type that adds and scales (a colour, `alias Sample =
vec3<f32>;`, or a struct of several), gets an `image` of the average over a grid of samples in each pixel,
and `screen(place)`, the place's direction across the view.
"""

from __future__ import annotations

from functools import cache


def software() -> bool:
    """Whether the adapter is the CPU itself, a software driver as in a runtime without a GPU."""
    from .runner import device
    return device().adapter.info["adapter_type"] == "CPU"


@cache
def samples_per_side() -> int:
    """The samples a side of every pixel, for every picture: sixteen to a pixel on a GPU, one on the CPU, where more
    cost too much. Asked of the device when a picture is first drawn, so that importing the drawing code needs none."""
    return 1 if software() else 4


def supersampled() -> str:
    """WGSL for a picture of `samples_per_side` squared samples a pixel: `sampled` at the pictures' default."""
    return sampled(samples_per_side())


def level_lines() -> str:
    """WGSL for lines where a field crosses its levels, each `width` samples wide however steep the field: its distance
    to a level over `rate`, its change across one sample, `fwidth` of it taken where the control flow is uniform.
    `level_line(level, rate, width)` covers the whole numbers of a field scaled so that its levels fall on them;
    `level_mark(offset, rate, width)` covers the one level where an offset from it vanishes. A line is drawn only where
    the field changes: where it is flat, even at a level, there is no line to draw."""
    return """
fn level_line(level: f32, rate: f32, width: f32) -> f32 {
    // The distance to the nearest level, in samples: how far the field lies from it over its change across one sample.
    let distance = abs(fract(level + 0.5) - 0.5) / max(rate, 1e-6);
    return select(0.0, 1.0 - smoothstep(width / 2.0 - 0.5, width / 2.0 + 0.5, distance), rate > 0.0);
}

fn level_mark(offset: f32, rate: f32, width: f32) -> f32 {
    let distance = abs(offset) / max(rate, 1e-30);
    return select(0.0, 1.0 - smoothstep(width / 2.0 - 0.5, width / 2.0 + 0.5, distance), rate > 0.0);
}
"""


def sampled(side: int) -> str:
    """WGSL for an image of `side` squared samples a pixel, on a grid turned so that no two share a row or a
    column, which resolves near-horizontal and near-vertical edges as finely as a grid of that many to a side;
    one, at each pixel's centre, for an image traced live."""
    count = side**2
    offsets = [((across * side + down + 0.5) / count - 0.5,
                (down * side + side - 1 - across + 0.5) / count - 0.5)
               for across in range(side) for down in range(side)]
    entries = ", ".join(f"vec2<f32>({x!r}, {y!r})" for x, y in offsets)
    return f"""
// The image's size, columns then rows, and each pixel's average of what its samples see.
var<uniform> pixels: vec2<u32>;
var<storage, read_write> image: array<Sample>;

// Where a pixel's samples lie, in pixel widths from its centre: one in each cell of a grid `samples_per_side` across.
const samples_per_side = {side}u;
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
