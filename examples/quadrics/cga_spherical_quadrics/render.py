"""Rasterise spherical quadrics on the front hemisphere of S²."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb

from examples.quadrics.cga_spherical_quadrics.core import Quadric, point
from examples.quadrics.elliptic_physics.render import hemisphere, paint


def vortex_frames(world: Quadric, colors: list[str], resolution: int, supersample: int) -> list[np.ndarray]:
    """One frame per row of world quadrics: a pixel takes a quadric's colour where quadrics(pixels) & pixels < 0."""
    coordinates, inside, rim = hemisphere(resolution * supersample)
    pixels = point(coordinates)
    rgb = np.array([to_rgb(color) for color in colors])
    return [paint(((quadrics[:, None](pixels) & pixels) < 0.0), rgb, inside, rim, supersample) for quadrics in world]
