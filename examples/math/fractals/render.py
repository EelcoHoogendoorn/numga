"""Colouring for the fractals notebook, as WGSL for its shaders: escape counts coloured smoothly through repeating
bands, the starts that stayed bounded dark; and the shaders' images as pictures."""

from __future__ import annotations

import numpy as np


def colouring() -> str:
    """WGSL for the colouring: `banded`, the colour of a start that escaped after a smoothed count of steps, and
    `bounded`, the colour of the starts that stay bounded."""
    return """
// The colours escaping starts run through as they take longer to escape, repeating, how many steps lie between the
// first and the last, and the escape counts per pass through them.
const bands = array<vec3<f32>, 7>(
    vec3<f32>(0.05, 0.07, 0.25), vec3<f32>(0.12, 0.35, 0.65), vec3<f32>(0.55, 0.80, 0.90), vec3<f32>(0.98, 0.93, 0.70),
    vec3<f32>(0.95, 0.60, 0.20), vec3<f32>(0.55, 0.15, 0.25), vec3<f32>(0.05, 0.07, 0.25));
const n_bands = 6u;
const period = 24.0;
// The colour of the starts that stay bounded.
const bounded = vec3<f32>(0.02, 0.02, 0.04);

// The colour of a start that escaped after a smoothed count of steps: through the bands, repeating.
fn banded(count: f32) -> vec3<f32> {
    let position = fract(count / period) * f32(n_bands);
    let lower = u32(position);
    return mix(bands[lower], bands[lower + 1], fract(position));
}
"""


def picture(image: np.ndarray) -> np.ndarray:
    """A shader's image `[rows, columns, 3]` as bytes."""
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
