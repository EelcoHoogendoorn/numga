"""Rendering for a rotating Bose-Einstein condensate draws the atom density across the triangulated disk in shades of
blue so vortex cores appear as white holes, alongside a radial staircase chart comparing the enclosed vortex count
against Feynman's areal density rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.quantum.condensate import core
from examples.util.charts import style, subplots
from numga.render import Camera, contours, drawing
from numga.topology import Topology

mv = core.mv
density_colormap = colormaps["Blues"]
count_colour = np.array(to_rgb("#1d4ed8"))                                    # [3]
feynman_colour = np.array(to_rgb("#c2410c"))                                  # [3]
# Pixel width, half-width in oscillator lengths, and peak colourmap fraction for the density animation.
n_pixels, half_width = 300, 7.5
peak_shade = 0.95
chart_width, chart_height = 300, 220


def white_composite(picture: np.ndarray) -> np.ndarray:
    """Alpha blending over white converts an RGBA image (`[n_pixels, n_pixels, 4]`) into `uint8` RGB bytes
    (`[n_pixels, n_pixels, 3]`)."""
    blended = picture[..., :3] + (1 - picture[..., 3:])                       # [n_pixels, n_pixels, 3]
    return np.round(np.clip(blended, 0, 1) * 255).astype(np.uint8)            # [n_pixels, n_pixels, 3]


def density_animation(topology: Topology, vertices: core.Point, states: Iterable[core.Wave]) -> Iterator[np.ndarray]:
    """Yields RGB frames (`[n_pixels, n_pixels, 3]`) of the condensate atom density across the trap, normalized to each
    frame's peak vertex density."""
    camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), np.array([half_width, half_width]), n_pixels)
    points = vertices.batch()                                                 # [V] Point-
    for wave in states:
        values = wave.scalar_norm_squared().to_array()                        # [V]
        colours = density_colormap(peak_shade * values / values.max())[:, :3] # [V, 3]
        yield white_composite(drawing.composited([contours.picture(points, topology, values, colours, camera)]))


@style()
def count_figure(radii: np.ndarray, counts: np.ndarray, feynman_counts: np.ndarray) -> Figure:
    """The radial chart compares the enclosed vortex staircase (`[n_radii]`) against Feynman's uniform-vorticity curve
    (`[n_radii]`)."""
    figure, ax = subplots(chart_width, chart_height)
    ax.plot(radii, feynman_counts, color=feynman_colour)
    ax.plot(radii, counts, color=count_colour, drawstyle="steps-post")
    ax.set_xlim(0, radii.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("radius / oscillator length")
    ax.set_ylabel("vortices within")
    figure.tight_layout(pad=0.4)
    return figure
