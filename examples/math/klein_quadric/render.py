"""The four-line problem draws each real line as its chord across a ball about the origin: the hyperboloid
rulings stay faint, the three lines fixing the surface are dark blue, the fourth line is orange, and the
transversals and crossings are red wherever they are real."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.math.klein_quadric import core
from numga.topology import Topology
from numga.render import simplices, solids
from examples.util.panels import image_figure

# Lines are clipped to a ball about the origin at fixed elevation and initial azimuth (radians, zero looking along +y).
# Rulings on the hyperboloid are clipped to a tighter ball around its waist so a sparse set outlines the surface cleanly.
ball_radius = 3.0
hyperboloid_radius = 1.8
elevation, azimuth = np.radians([18.0, 30.0])
margin = 0.06
n_pixels = 300
# Pixel radii of hyperboloid rulings, scene lines, and crossing balls, together with ruling opacity.
ruling_radius, line_radius, crossing_radius = 0.8, 1.5, 3.2
ruling_opacity = 0.3
ruling_colours = [np.array(to_rgb(colour)) for colour in ("#9ecae1", "#c7c7c7")]
line_colour, fourth_colour, transversal_colour = (np.array(to_rgb(colour)) for colour in ("#08306b", "#e67e22", "#c0392b"))
background = np.ones(3)
# Animation frames quantize pixels onto flat tones (background, half-blends, full hues) so a turning camera keeps a compact palette.
hues = np.stack([*ruling_colours, line_colour, fourth_colour, transversal_colour])
tones = np.concatenate([background[None], (background + hues) / 2, hues]) * 255
darkest = np.concatenate([np.ones(1 + len(hues)), np.full(len(hues), 0.3)])


# --- plumbing -------------------------------------------------------------------------
def real_mask(points: core.Point) -> np.ndarray:
    """Homogeneous coordinate pairing identifies points (`[...]`) whose imaginary part is negligible."""
    mv = points.context.multivector
    coordinates = (stack((mv.x, mv.y, mv.z, mv.w)) & points[..., None]).to_array()   # [..., 4]
    return np.abs(coordinates.imag).max(axis=-1) <= 1e-6 * np.abs(coordinates).max(axis=-1)


def ball_chords(lines: core.Bivector, radius: float) -> core.Point:
    """Real lines crossing the ball of `radius` about the origin meet its boundary in real chord
    endpoints (`[..., 2] Antivector-`), omitting complex lines and lines that miss the ball."""
    ends = core.crossings_of(solids.ball(solids.origin, radius), lines)       # [..., 2] Antivector-
    return ends[real_mask(ends).all(axis=-1)].real()                          # [..., 2] Antivector-


def ball_rods(lines: core.Bivector, radius: float, rod_radius: float) -> solids.Quadric:
    """Real chords of `lines` across the ball of `radius` form cylinders of `rod_radius`."""
    ends = ball_chords(lines, radius)                                         # [..., 2] Antivector-
    return solids.rods(ends[:, 0], ends[:, 1], rod_radius)


def picture(lines: core.Bivector, rulings: core.Bivector, transversals: core.Bivector,
            crossings: core.Point, view_azimuth: float) -> np.ndarray:
    """Ray-tracing from `view_azimuth` renders the four lines, hyperboloid rulings, real
    transversals, and real crossings into an `[n_pixels, n_pixels, 3]` RGB image."""
    view = solids.View(solids.aim(solids.origin, elevation, view_azimuth), np.full(2, ball_radius * (1 + margin)),
                       n_pixels)
    points = crossings[real_mask(crossings)].real()                           # [...] Antivector-
    # Tube meshes composite sample by sample without ray-tracing through every translucent cylinder layer.
    ruling_ends = [ball_chords(family, hyperboloid_radius) for family in rulings]
    hyperboloid = [(simplices.tubes(ends.reshape(-1), Topology.chains([2] * len(ends)), colour, 2 * ruling_radius,
                                    view, gloss=solids.gloss),
                    # Matching two cylinder wall crossings with a single tube pass squares the transmission factor.
                    1 - (1 - ruling_opacity) ** 2) for ends, colour in zip(ruling_ends, ruling_colours)]
    four = [(ball_rods(lines[:3], ball_radius, line_radius * view.pixel), line_colour),
            (ball_rods(lines[3:], ball_radius, line_radius * view.pixel), fourth_colour),
            (ball_rods(transversals, ball_radius, line_radius * view.pixel), transversal_colour),
            (solids.ball(points, crossing_radius * view.pixel)[:, None], transversal_colour)]
    return solids.draw(four, view, background, inks=[], translucent=[], meshes=hyperboloid)


def tone_quantized(image: np.ndarray) -> np.ndarray:
    """Each RGB pixel of `image` snaps to its nearest lit palette tone as `uint8` RGB bytes."""
    # Clamp best-fit illumination per tone within its allowed shadow range before choosing the closest hue.
    light = np.clip((image @ tones.T) / (tones**2).sum(axis=-1), darkest, 1.0)   # [..., 11]
    nearest = ((image[..., None, :] - light[..., None] * tones) ** 2).sum(axis=-1).argmin(axis=-1)
    return np.round(tones[nearest]).astype(np.uint8)


def scene_figure(lines: core.Bivector, rulings: core.Bivector, transversals: core.Bivector,
                 crossings: core.Point) -> plt.Figure:
    """A still figure shows the four lines, both ruling families of the hyperboloid, and any real
    transversals and crossings."""
    return image_figure(picture(lines, rulings, transversals, crossings, azimuth))


def swing_frames(n_frames: int,
                 scenes: Iterator[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]]) -> Iterator[np.ndarray]:
    """Yields flat-toned RGB frames across `scenes` as the camera completes one turn over `n_frames`."""
    for frame, scene in enumerate(scenes):
        yield tone_quantized(picture(*scene, azimuth + 2 * np.pi * frame / n_frames))
