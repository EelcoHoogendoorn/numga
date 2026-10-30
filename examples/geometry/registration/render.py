"""Drawing for the registration example, ray traced on the GPU: the clouds as small balls, the targets as translucent shells around where the aligned points should land, and each aligned point tied to its target."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, stack
from examples.util.wgsl import image_figure, solids

source_colour, target_colour, aligned_colour = (np.array(to_rgb(colour)) for colour in ("#94a3b8", "#0284c7", "#f43f5e"))
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the points, as a
# share of the image's larger side; and the pixels across the image.
elevation, azimuth = np.radians([30.0, 30.0])
margin = 0.12
n_pixels = 300
# Radii in pixels: of a point, of the shell around a target, and of the tie from an aligned point to its target;
# and how much of what lies behind them each surface of a shell hides.
point_radius, target_radius, tie_radius = 2.5, 4.5, 1.0
target_opacity = 0.16


# --- plumbing -------------------------------------------------------------------------
def gpu_points(points: Extensor) -> solids.Point:
    """View the source along its remaining directions, with its x, y and z slices along the scene's axes."""
    mv = points.context.multivector
    readings = stack((mv.x, mv.y, mv.z)) & points.normalized()[..., None]
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)



def draw_registration(source: Extensor, target: Extensor, aligned: Extensor) -> plt.Figure:
    """The source, grey, the target, blue shells, and the aligned source, red, each aligned point tied to its target."""
    source, target, aligned = (gpu_points(points) for points in (source, target, aligned))
    view = solids.view([source, target, aligned], 0.0, elevation, azimuth, margin, n_pixels)
    pixel = view.pixel
    opaque = [(solids.ball(source, point_radius * pixel)[:, None], source_colour),
              (solids.ball(aligned, point_radius * pixel)[:, None], aligned_colour),
              (solids.rods(aligned, target, tie_radius * pixel), aligned_colour)]
    shells = [(solids.ball(target, target_radius * pixel)[:, None], target_colour, target_opacity)]
    return image_figure(solids.draw(opaque, view, background, inks=[], translucent=shells, meshes=[]))
