"""Rasterise the great circles of a set of planes on the front hemisphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.conformal_elliptical.core import Point, Vector, mv

# Pixels across the hemisphere before downsampling.
RESOLUTION = 512
# The tanh gain, which sets the antialiased width of a circle.
SHARPNESS = 512.0
# The downsampling factor.
BIN = 2


# --- plumbing -------------------------------------------------------------------------
def hemisphere(n: int) -> tuple[Point, np.ndarray]:
    """Points of the front hemisphere seen along z, and the mask of pixels outside the disk."""
    x = np.linspace(-1.0, +1.0, n)[:, None] * np.ones((n, n))
    y = np.linspace(-1.0, +1.0, n)[None, :] * np.ones((n, n))
    z = np.sqrt(np.maximum(1.0 - x**2 - y**2, 0.0))
    return mv.vector(np.stack([x, y, z], axis=-1)).dual(), (x**2 + y**2) > 1.0


def draw_planes(planes: Vector) -> plt.Figure:
    """The planes' great circles on the front hemisphere, the cells between them in two shades by the
    parity of the planes a point lies above."""
    points, outside = hemisphere(RESOLUTION)
    # Each point's height above each plane, by their pairing.
    heights = (planes & points[:, :, None]).to_array()                                 # [n, n, planes]
    shade = (np.prod(np.tanh(heights * SHARPNESS), axis=-1) + 1.0) / 2.0
    image = (shade + 1.0) * (1.0 - outside)
    size = RESOLUTION // BIN
    frame = image.reshape(size, BIN, size, BIN).mean(axis=(1, 3))
    fig, ax = plt.subplots(dpi=150)
    ax.imshow(frame, cmap="gray")
    ax.axis("off")
    return fig
