"""Reusable WGSL snippets supply subpixel image sampling and screen-space contour evaluation for
shaders written in the `numga` dialect. `sampled` and `supersampled` provide the `@compute` entry
point, uniform `pixels` dimensions, output `image` buffer, and `screen(place)` view coordinates for
any shader defining `fn sample(place: vec2<f32>) -> Sample` on an additive, scalable `Sample` type.
Within each pixel, sample offsets sit on a rotated `n`-rooks grid so no two samples share a row or
column, resolving near-horizontal and near-vertical edges at `n**2` abscissas while `samples_per_side`
queries the WebGPU adapter lazily on first draw (`4` on hardware GPUs, `1` on CPU software drivers).

`level_lines` provides screen-space anti-aliased contour helpers `level_line` and `level_mark`,
dividing a scalar field's distance to a target level by its per-sample screen-space variation `rate`
(such as `fwidth` evaluated in uniform control flow) so lines stay `width` samples wide regardless of
field steepness and vanish where the field is flat.
"""

from __future__ import annotations

from functools import cache


def software() -> bool:
    """A `"CPU"` adapter type indicates a software WebGPU driver in an environment without a hardware GPU."""
    from .runner import device
    return device().adapter.info["adapter_type"] == "CPU"


@cache
def samples_per_side() -> int:
    """Hardware GPUs evaluate four samples per pixel side (sixteen per pixel), whereas CPU software
    adapters fall back to one sample per pixel. Querying the adapter on first call avoids opening a
    WebGPU device at import time."""
    return 1 if software() else 4


def supersampled() -> str:
    """Default supersampling expands `sampled` at the device's `samples_per_side()` resolution."""
    return sampled(samples_per_side())


def level_lines() -> str:
    """Screen-space contour helpers divide a field's distance to a target level by its per-sample
    variation `rate` so lines maintain a constant thickness of `width` samples and vanish where
    `rate <= 0`. `level_line` marks every integer value of `level`, while `level_mark` marks the zero
    crossing of `offset`."""
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


def sampled(n_samples_per_side: int) -> str:
    """Rotated `n`-rooks subpixel offsets place `n_samples_per_side**2` samples per pixel with distinct
    horizontal and vertical coordinates, reducing to a single pixel-centre sample when
    `n_samples_per_side == 1`."""
    count = n_samples_per_side**2
    offsets = [((across * n_samples_per_side + down + 0.5) / count - 0.5,
                (down * n_samples_per_side + n_samples_per_side - 1 - across + 0.5) / count - 0.5)
               for across in range(n_samples_per_side) for down in range(n_samples_per_side)]
    entries = ", ".join(f"vec2<f32>({x!r}, {y!r})" for x, y in offsets)
    return f"""
// The image's size, columns then rows, and each pixel's average of what its samples see.
var<uniform> pixels: vec2<u32>;
var<storage, read_write> image: array<Sample>;

// Where a pixel's samples lie, in pixel widths from its centre: one in each cell of a grid `samples_per_side` across.
const samples_per_side = {n_samples_per_side}u;
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
