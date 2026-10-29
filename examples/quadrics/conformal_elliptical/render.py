"""Rasterise the great circles of a set of planes on the front hemisphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.conformal_elliptical.core import Plane, Point, mv
from examples.util.wgsl import image_figure

# Samples across the hemisphere, `samples_per_side` by `samples_per_side` of them averaged into each pixel of the
# picture.
n_samples = 512
# The tanh gain, which sets the antialiased width of a circle.
sharpness = 512.0
samples_per_side = 2
# The direction toward the viewer, who looks along -z.
toward_viewer = mv.z.dual()                                                  # [] Point-


# --- plumbing -------------------------------------------------------------------------
def hemisphere(n: int) -> tuple[Point, np.ndarray]:
    """The points of the front hemisphere under an n by n grid of samples across the unit disk seen along z, and which
    of the samples lie inside the disk: each sample's place across the disk lifted toward the viewer onto the
    sphere."""
    steps = np.linspace(-1.0, +1.0, n)
    across = mv.x.dual() * steps[:, None] + mv.y.dual() * steps[None, :]     # [rows, columns] Point-
    inside = across.norm() <= 1.0                                            # [rows, columns]
    disk = across[inside]                                                    # [samples] Point-
    return disk + toward_viewer * (1 - disk.scalar_norm_squared()).square_root(), inside


def draw_planes(planes: Plane) -> plt.Figure:
    """The planes' great circles on the front hemisphere, the cells between them in two shades by the
    parity of the planes a point lies above."""
    points, inside = hemisphere(n_samples)
    # Each point's height above each plane, by their pairing.
    heights = (planes & points[:, None]).to_array()                          # [samples, planes]
    image = np.zeros(inside.shape)
    image[inside] = (np.prod(np.tanh(heights * sharpness), axis=-1) + 1.0) / 2.0 + 1.0
    size = n_samples // samples_per_side
    frame = image.reshape(size, samples_per_side, size, samples_per_side).mean(axis=(1, 3))
    grey = plt.get_cmap("gray")((frame - frame.min()) / (frame.max() - frame.min()))
    return image_figure(np.round(grey[..., :3] * 255).astype(np.uint8))
