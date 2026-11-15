"""Orthographic hemisphere samples pair with central planes to shade the spherical regions between their great circles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.conformal_elliptical.core import Plane, Point, mv
from examples.util.panels import image_figure

n_samples = 512
n_subsamples = 2
sharpness = 512.0
# Viewing axis pointing from the sphere centre toward the camera.
toward_viewer = mv.z.dual()                                                     # [] Point-


# --- plumbing -------------------------------------------------------------------------
def hemisphere_samples(n_samples: int) -> tuple[Point, np.ndarray]:
    """Orthographic grid samples inside the unit disc lift along the viewing axis onto the front hemisphere (`[...] Point-`), paired with the `[n_samples, n_samples]` disc mask."""
    coordinates = np.linspace(-1, 1, n_samples)                                 # [n_samples]
    disc_grid = (
        mv.x.dual() * coordinates[:, None] + mv.y.dual() * coordinates[None, :]
    )                                                                           # [n_samples, n_samples] Point-
    disc_mask = disc_grid.antinorm() <= 1                                       # [n_samples, n_samples]
    disc_points = disc_grid[disc_mask]                                          # [...] Point-
    front_depths = (1 - disc_points.scalar_norm_squared()).square_root()        # [...] Scalar+
    return disc_points + toward_viewer * front_depths, disc_mask


def planes_figure(planes: Plane) -> plt.Figure:
    """Pairings between `planes` (`[n_planes] Plane+`) and front-hemisphere points shade the spherical regions between great circles in two alternating tones."""
    n_planes = len(planes)
    points, disc_mask = hemisphere_samples(n_samples)
    # Signed height of each hemisphere point above each central plane.
    heights = (planes & points[:, None]).to_array()                             # [..., n_planes]
    image = np.zeros(disc_mask.shape)
    image[disc_mask] = (np.prod(np.tanh(heights * sharpness), axis=-1) + 1) / 2 + 1
    n_pixels = n_samples // n_subsamples
    frame = image.reshape(n_pixels, n_subsamples, n_pixels, n_subsamples).mean(axis=(1, 3))
    grey = plt.get_cmap("gray")((frame - frame.min()) / (frame.max() - frame.min()))
    return image_figure(np.round(grey[..., :3] * 255).astype(np.uint8))
