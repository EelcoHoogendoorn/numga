"""Smooth escape-count colouring in WGSL and float-to-byte image conversion for the fractals notebook."""

from __future__ import annotations

import numpy as np


def colouring() -> str:
    """Repeating colour bands in WGSL map smoothed escape counts through `banded` while assigning `bounded` to non-escaping starts."""
    return '''
// Repeating palette stops across `period` escape steps per pass.
const bands = array<vec3<f32>, 7>(
    vec3<f32>(0.05, 0.07, 0.25), vec3<f32>(0.12, 0.35, 0.65), vec3<f32>(0.55, 0.80, 0.90), vec3<f32>(0.98, 0.93, 0.70),
    vec3<f32>(0.95, 0.60, 0.20), vec3<f32>(0.55, 0.15, 0.25), vec3<f32>(0.05, 0.07, 0.25));
const n_bands = 6u;
const period = 24.0;
const bounded = vec3<f32>(0.02, 0.02, 0.04);

// Smooth linear interpolation across the repeating colour stops.
fn banded(count: f32) -> vec3<f32> {
    let position = fract(count / period) * f32(n_bands);
    let lower = u32(position);
    return mix(bands[lower], bands[lower + 1], fract(position));
}
'''


def picture(image: np.ndarray) -> np.ndarray:
    """Clamping `[rows, columns, 3]` float RGB values to `[0, 1]` and rounding to `uint8` produces a displayable byte image."""
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
