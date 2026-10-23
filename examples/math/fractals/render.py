"""Colouring for the fractals notebook, as WGSL for its shaders: escape counts coloured smoothly through repeating
bands, the starts that stayed bounded dark; and the shaders' images as pictures."""

from __future__ import annotations

import numpy as np

from examples.util.wgsl import vec3

BACKGROUND = np.array([0.02, 0.02, 0.04])
# The colours escaping points run through as they take longer to escape, repeating.
BANDS = np.array([
    [0.05, 0.07, 0.25], [0.12, 0.35, 0.65], [0.55, 0.80, 0.90], [0.98, 0.93, 0.70],
    [0.95, 0.60, 0.20], [0.55, 0.15, 0.25], [0.05, 0.07, 0.25],
])
# Escape counts per pass through the colours.
PERIOD = 24


def colouring() -> str:
    """WGSL for the colouring: `banded`, the colour of a start that escaped after a smoothed count of steps, and
    `bounded`, the colour of the starts that stay bounded."""
    entries = ", ".join(vec3(colour) for colour in BANDS)
    return f"""
// The bands escaping starts run through, how many steps between the first and the last, and the escape counts per
// pass through them.
const bands = array<vec3<f32>, {len(BANDS)}>({entries});
const n_bands = {len(BANDS) - 1}u;
const period = {float(PERIOD)};
const bounded = {vec3(BACKGROUND)};

// The colour of a start that escaped after a smoothed count of steps: through the bands, repeating.
fn banded(count: f32) -> vec3<f32> {{
    let position = fract(count / period) * f32(n_bands);
    let lower = u32(position);
    return mix(bands[lower], bands[lower + 1], fract(position));
}}
"""


def picture(image: np.ndarray) -> np.ndarray:
    """A shader's image `[rows, columns, 3]` as bytes."""
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
