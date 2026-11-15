"""GPU ray-traced rendering of point-cloud registration draws the source and aligned clouds as opaque balls, surrounds each target with a translucent shell, and links each aligned point to its target with a rod."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor
from numga.render import solids
from examples.util.panels import image_figure

source_colour, target_colour, aligned_colour = (np.array(to_rgb(colour)) for colour in ("#94a3b8", "#0284c7", "#f43f5e"))
background = np.ones(3)
# Camera elevation and azimuth in radians (azimuth zero looks along +y), fractional margin around the clouds,
# and image side length in pixels.
elevation, azimuth = np.radians([30.0, 30.0])
margin = 0.12
n_pixels = 300
# Screen-space radii in pixels for cloud points, target shells, and residual rods, plus shell surface opacity.
point_radius, target_radius, tie_radius = 2.5, 4.5, 1.0
target_opacity = 0.16


# --- plumbing -------------------------------------------------------------------------
def registration_figure(source: Extensor, target: Extensor, aligned: Extensor) -> plt.Figure:
    """Ray-traced figure showing the grey `source` cloud, translucent blue `target` shells, and red `aligned` points tied to their targets (`[...] Point-`)."""
    # Slicing along w extracts the x, y, z offset from the origin in flat space and the orthographic
    # equatorial projection on the three-sphere, which the projective scene places around its origin.
    weight_plane = source.context.multivector.w                                # Plane+
    source_points, target_points, aligned_points = (
        solids.origin + solids.context.include_from_algebra(weight_plane ^ (points.normalized() & weight_plane.dual()))
        for points in (source, target, aligned)
    )                                                                          # [...] Point-, [...] Point-, [...] Point-
    view = solids.view([source_points, target_points, aligned_points], 0.0, elevation, azimuth, margin, n_pixels)
    pixel = view.pixel
    opaque = [
        (solids.ball(source_points, point_radius * pixel)[:, None], source_colour),
        (solids.ball(aligned_points, point_radius * pixel)[:, None], aligned_colour),
        (solids.rods(aligned_points, target_points, tie_radius * pixel), aligned_colour),
    ]
    shells = [(solids.ball(target_points, target_radius * pixel)[:, None], target_colour, target_opacity)]
    return image_figure(solids.draw(opaque, view, background, inks=[], translucent=shells, meshes=[]))
