"""Rendering for a rotating Bose-Einstein condensate draws the wave across the triangulated disk over black, its phase as
hue and its density as brightness, so each vortex core shows as a dark disc around which the colour wheel winds once,
alongside a radial staircase chart comparing the enclosed vortex count against Feynman's areal density rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.quantum.condensate import core
from examples.util.charts import style, subplots
from numga.render import Camera, contours, drawing
from numga.topology import Topology

mv = core.mv
count_colour = np.array(to_rgb("#1d4ed8"))                                    # [3]
feynman_colour = np.array(to_rgb("#c2410c"))                                  # [3]
# Pixel width and half-width in oscillator lengths for the wave animation.
n_pixels, half_width = 300, 7.5
# The colour wheel's swing about grey, short of clipping into bands of pure primaries.
swing = 0.75
# The colour's strength: a curve on the density over its local background, the density smoothed over about
# `smoothing_length` oscillator lengths, which darkens each core to a dark disc, times a gentle curve on that background
# over its peak, which fades the cloud's rim.
core_curve = (0.0, 0.6)
rim_curve = (0.0, 0.4)
smoothing_length = 1.2
chart_width, chart_height = 300, 220


def smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    """Hermite step from 0 at `low` to 1 at `high`."""
    shares = np.clip((values - low) / (high - low), 0, 1)
    return shares * shares * (3 - 2 * shares)


def black_composite(picture: np.ndarray) -> np.ndarray:
    """Alpha blending over black converts an RGBA image (`[n_pixels, n_pixels, 4]`) into `uint8` RGB bytes
    (`[n_pixels, n_pixels, 3]`)."""
    blended = picture[..., :3] * picture[..., 3:]                             # [n_pixels, n_pixels, 3]
    return np.round(np.clip(blended, 0, 1) * 255).astype(np.uint8)            # [n_pixels, n_pixels, 3]


def wave_animation(topology: Topology, vertices: core.Point, states: Iterable[core.Wave]) -> Iterator[np.ndarray]:
    """Yields RGB frames (`[n_pixels, n_pixels, 3]`) of the condensate wave across the trap over black: its phase as
    hue, and its density over the local background as brightness, fading to black at the cores and at the cloud's
    rim."""
    camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), np.array([half_width, half_width]), n_pixels)
    points = vertices.batch()                                                 # [V] Point-
    # Three directions of the wave's plane a third of a turn apart; the unit wave read against each is a cosine of its
    # phase, so the three channels run once through the colour wheel as the wave turns once, with no angle.
    turns = (mv.ab * (-np.pi / 3 * np.arange(3))).exp()                       # [3] Even+
    channels = turns >> mv.a                                                  # [3] Wave+
    # Averaging each vertex with the mean of its neighbours along the mesh's edges, repeated enough times to spread the
    # density over about `smoothing_length`, from the mean edge length.
    ends = topology.incidence[1, 0].sum(axis=0)                               # Scalar+[E <- V]
    gathering = ends.adjugate() * ends                                        # Scalar+[V <- V]
    neighbourhoods = gathering(np.ones(topology.n_vertices))                  # [V]
    spacing = (topology.d[0] * vertices).antinorm().to_array().mean()
    n_smoothing_steps = round(2 * (smoothing_length / spacing) ** 2)
    for wave in states:
        density = wave.scalar_norm_squared()                                  # Scalar+[V]
        amplitude = density.square_root()                                     # Scalar+[V]
        readings = wave.batch()[:, None].scalar_product(channels) / amplitude.batch()[:, None]   # [V, 3] Scalar+
        hues = np.clip(0.5 + swing * readings.to_array(), 0, 1)               # [V, 3]
        densities = backgrounds = density.to_array()                          # [V]
        for _ in range(n_smoothing_steps):
            backgrounds = gathering(backgrounds) / neighbourhoods             # [V]
        # A core reads against the cloud around it, the cloud's rim against the peak of the background, which the
        # cores do not dent.
        cores = smoothstep(*core_curve, densities / backgrounds)              # [V]
        rims = smoothstep(*rim_curve, backgrounds / backgrounds.max())        # [V]
        strengths = cores * rims                                              # [V]
        colours = strengths[:, None] * hues                                   # [V, 3]
        yield black_composite(drawing.composited([contours.picture(points, topology, strengths, colours, camera)]))


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
