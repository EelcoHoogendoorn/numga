"""Patterns shaders in the numga dialect share, as WGSL to append to a shader's own.

`supersampled` makes the image: a shader that defines what one sample sees, `fn sample(place:
vec2<f32>) -> Sample`, with `Sample` any type that adds and scales (a colour, `alias Sample =
vec3<f32>;`, or a struct of several), gets an `image` of the average over a grid of samples in each pixel,
and `screen(place)`, the place's direction across the view.
"""

from __future__ import annotations


def supersampled(samples_per_side: int) -> str:
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

// Where a pixel's samples lie, in pixel widths from its centre.
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
