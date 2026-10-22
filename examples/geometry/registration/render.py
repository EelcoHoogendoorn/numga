"""Drawing for the registration example, ray traced on the GPU: the clouds as small balls, the targets as
translucent shells around where the aligned points should land, and each aligned point tied to its target."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.geometry.registration.core import Point, mv
from examples.wgsl import compose, image_figure, solids

source_colour, target_colour, aligned_colour = (np.array(to_rgb(colour)) for colour in ("#94a3b8", "#0284c7", "#f43f5e"))
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the points, as a
# share of the image's larger side; and the pixels across the image.
elevation, azimuth = np.radians([30.0, 30.0])
margin = 0.12
n_pixels = 900
# Radii in pixels: of a point, of the shell around a target, and of the tie from an aligned point to its target;
# and how much of what lies behind a shell shows through it.
point_radius, target_radius, tie_radius = 5.0, 8.5, 1.5
target_opacity = 0.3


def coordinates(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def draw_registration(source: Point, target: Point, aligned: Point) -> plt.Figure:
    """The source, grey, the target, blue shells, and the aligned source, red, each aligned point tied to
    its target."""
    view = solids.view([source, target, aligned], 0.0, elevation, azimuth, margin, n_pixels)
    pixel = view.pixel
    opaque = solids.layer([(solids.ball(source, point_radius * pixel)[:, None], source_colour),
                           (solids.ball(aligned, point_radius * pixel)[:, None], aligned_colour),
                           (solids.rods(aligned, target, tie_radius * pixel), aligned_colour)], view)
    shells = solids.layer([(solids.ball(target, target_radius * pixel)[:, None], target_colour)], view)
    return image_figure(compose([(*opaque, 1.0), (*shells, target_opacity)], background))
