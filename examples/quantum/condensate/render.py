"""Rendering for a rotating Bose-Einstein condensate draws the wave across the triangulated disk inside a physically
rotating octagonal vacuum-chamber viewport, with phase as hue and local density as brightness so each vortex core shows
as a dark disc around which the colour wheel winds once, alongside a radial staircase chart comparing the enclosed
vortex count against Feynman's areal density rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.quantum.condensate import core
from examples.util.charts import style, subplots
from numga import concatenate
from numga.render import Camera, contours, drawing
from numga.topology import Topology

mv = core.mv
count_colour = np.array(to_rgb("#1d4ed8"))                                    # [3]
feynman_colour = np.array(to_rgb("#c2410c"))                                  # [3]
# Pixel width and half-width in oscillator lengths for the wave animation.
n_pixels, half_width = 300, 7.5
# Regular octagonal viewport mask: inner corner radius share of `half_width` and outer skirt past the frame corners.
n_viewport_sides = 8
viewport_share, viewport_outer_scale = 0.98, 2.0
# The colour wheel's swing about grey, short of clipping into bands of pure primaries.
swing = 0.75
# The colour's strength: a soft core taper on the density over its local background, smoothed over `smoothing_length`
# oscillator lengths, times a rim curve on that background over its peak.
core_curve = (0.0, 0.8)
rim_curve = (0.0, 0.4)
smoothing_length = 1.2
chart_width, chart_height = 300, 220


def smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    """A cubic Hermite step rises smoothly from 0 at `low` to 1 at `high`."""
    shares = np.clip((values - low) / (high - low), 0, 1)
    return shares * shares * (3 - 2 * shares)


def black_composite(picture: np.ndarray) -> np.ndarray:
    """Alpha blending over black converts an RGBA image (`[n_pixels, n_pixels, 4]`) into `uint8` RGB bytes
    (`[n_pixels, n_pixels, 3]`)."""
    blended = picture[..., :3] * picture[..., 3:]                             # [n_pixels, n_pixels, 3]
    return np.round(np.clip(blended, 0, 1) * 255).astype(np.uint8)            # [n_pixels, n_pixels, 3]


def viewport_annulus() -> tuple[Topology, core.Point]:
    """An annular topology of `2 * n_viewport_sides` triangles and its reference vertices (`[2 * n_viewport_sides]
    Point-`) frame a regular octagonal aperture whose corners stay inside the square picture at every rotation."""
    angles = (np.arange(n_viewport_sides) + 0.5) * (2 * np.pi / n_viewport_sides)   # [n_viewport_sides]
    corner_rotors = (mv.xy * (-angles / 2)).exp()                             # [n_viewport_sides] Even+
    inner_corners = corner_rotors >> (viewport_share * half_width * core.horizontal)   # [n_viewport_sides] Point-
    outer_corners = corner_rotors >> (viewport_outer_scale * half_width * core.horizontal)   # [n_viewport_sides] Point-
    side = np.arange(n_viewport_sides)
    next_side = (side + 1) % n_viewport_sides
    faces = np.concatenate((
        np.stack((side, n_viewport_sides + side, n_viewport_sides + next_side), axis=-1),
        np.stack((side, n_viewport_sides + next_side, next_side), axis=-1),
    ))
    return Topology(2 * n_viewport_sides, faces), concatenate((inner_corners, outer_corners))


def wave_animation(topology: Topology, vertices: core.Point, rotation: float, frame_time: float,
                   states: Iterable[core.Wave]) -> Iterator[np.ndarray]:
    """Yields RGB frames (`[n_pixels, n_pixels, 3]`) of the condensate wave in the rotating trap frame inside a
    physically counter-rotating octagonal laboratory viewport."""
    camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), np.array([half_width, half_width]), n_pixels)
    points = vertices.batch()                                                 # [V] Point-
    viewport_topology, viewport_points = viewport_annulus()
    viewport_values = np.ones(viewport_topology.n_vertices, dtype=np.float32)   # [2 * n_viewport_sides]
    viewport_colours = np.ones((viewport_topology.n_vertices, 3), dtype=np.float32)   # [2 * n_viewport_sides, 3]
    # Three directions of the wave's plane a third of a turn apart; the unit wave read against each is a cosine of its
    # phase, so the three channels run once through the colour wheel as the wave turns once, with no angle.
    turns = (mv.ab * (-np.pi / 3 * np.arange(3))).exp()                       # [3] Even+
    channels = turns >> mv.a                                                  # [3] Wave+
    # Averaging each vertex with the mean of its neighbours along the topology's edges spreads the density over about
    # `smoothing_length` oscillator lengths.
    ends = topology.incidence[1, 0].sum(axis=0)                               # Scalar+[E <- V]
    gathering = ends.adjugate() * ends                                        # Scalar+[V <- V]
    neighbourhoods = gathering(np.ones(topology.n_vertices))                  # [V]
    spacing = (topology.d[0] * vertices).antinorm().to_array().mean()
    n_smoothing_steps = round(2 * (smoothing_length / spacing) ** 2)
    for frame_index, wave in enumerate(states):
        density = wave.scalar_norm_squared()                                  # Scalar+[V]
        densities = backgrounds = density.to_array()                          # [V]
        amplitudes = np.sqrt(np.maximum(densities, 1e-30))                    # [V]
        projections = wave.batch()[:, None].scalar_product(channels).to_array()   # [V, 3]
        hues = np.clip(0.5 + swing * (projections / amplitudes[:, None]), 0, 1)   # [V, 3]
        for _ in range(n_smoothing_steps):
            backgrounds = gathering(backgrounds) / neighbourhoods             # [V]
        # A core reads against the cloud around it with a soft taper, and the rim against the background peak.
        ratios = densities / np.maximum(backgrounds, 1e-12 * backgrounds.max())   # [V]
        cores = (smoothstep(*core_curve, ratios) + np.clip(ratios, 0, 1) ** 0.5) / 2   # [V]
        rims = smoothstep(*rim_curve, backgrounds / backgrounds.max())        # [V]
        strengths = cores * rims                                              # [V]
        colours = strengths[:, None] * hues                                   # [V, 3]
        # In the trap's rotating frame, the stationary laboratory chamber turns opposite to the trap's spin.
        lab_angle = -rotation * frame_index * frame_time
        viewport_rotor = (mv.xy * (-lab_angle / 2)).exp()                     # [] Even+
        lab_viewport = viewport_rotor >> viewport_points                      # [2 * n_viewport_sides] Point-
        cloud_draw = contours.picture(points, topology, strengths, colours, camera)
        viewport_draw = contours.picture(lab_viewport, viewport_topology, viewport_values, viewport_colours, camera)
        yield black_composite(drawing.composited([cloud_draw, viewport_draw]))


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
