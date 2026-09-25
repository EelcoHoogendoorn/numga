"""Rasterise the great circles of a set of planes on the front hemisphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.conformal_elliptical.core import Vector, mv

# Pixels across the hemisphere before downsampling.
RESOLUTION = 512
# The tanh gain, which sets the antialiased width of a circle.
SHARPNESS = 512.0
# The downsampling factor.
BIN = 2


def hemisphere(n: int) -> tuple[Vector, np.ndarray]:
    """Points of the front hemisphere seen along z, and the mask of pixels outside the disk."""
    x = np.linspace(-1.0, +1.0, n)[:, None] * np.ones((n, n))
    y = np.linspace(-1.0, +1.0, n)[None, :] * np.ones((n, n))
    z = np.sqrt(np.maximum(1.0 - x**2 - y**2, 0.0))
    return mv.x * x + mv.y * y + mv.z * z, (x**2 + y**2) > 1.0


def draw_octahedral_sphere(planes: Vector) -> plt.Figure:
    rays, outside = hemisphere(RESOLUTION)
    side = (rays[:, :, None] | planes).to_array()
    shade = (np.prod(np.tanh(side * SHARPNESS), axis=-1) + 1.0) / 2.0
    image = (shade + 1.0) * (1.0 - outside)
    size = RESOLUTION // BIN
    frame = image.reshape(size, BIN, size, BIN).mean(axis=(1, 3))
    fig, ax = plt.subplots(dpi=150)
    ax.imshow(frame, cmap="gray")
    ax.axis("off")
    ax.set_title("Octahedral Planes on Unit 2-Sphere")
    return fig
