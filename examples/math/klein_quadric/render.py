"""The four-line problem is drawn with every line as its chord of a ball about the origin: the lines
on the hyperboloid faint, the three lines that fix it dark, the fourth orange, and the transversals
red where they are real."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, stack
from examples.math.klein_quadric import core
from examples.util.wgsl import image_figure, solids, strokes

# Lines are drawn inside this ball about the origin, seen from this elevation and, at first, this azimuth, in
# radians, azimuth zero looking along +y; the margin around the ball, as a share of its radius; the pixels across.
# The lines on the hyperboloid are drawn inside a smaller ball, about its waist, so that a few of them show the surface.
ball_radius = 3.0
hyperboloid_radius = 1.8
elevation, azimuth = np.radians([18.0, 30.0])
margin = 0.06
n_pixels = 300
# Radii in pixels: of the lines on the hyperboloid, of the four lines and the transversals, and of the crossings;
# and how much of what lies behind them each surface of the lines on the hyperboloid hides, so that they stay faint.
ruling_radius, line_radius, crossing_radius = 0.8, 1.5, 3.2
ruling_opacity = 0.3
ruling_colours = [np.array(to_rgb(colour)) for colour in ("#9ecae1", "#c7c7c7")]
line_colour, fourth_colour, transversal_colour = (np.array(to_rgb(colour)) for colour in ("#08306b", "#e67e22", "#c0392b"))
background = np.ones(3)
# The flat tones an animation's frames are put in: the background, and every colour laid over it by half and whole;
# and how deep a shadow the lamp casts on a whole colour, which still counts as that colour. At these pixels shading
# and soft edges hardly show, and with the view turning, every tone is paid for in every frame.
hues = np.stack([*ruling_colours, line_colour, fourth_colour, transversal_colour])
tones = np.concatenate([background[None], (background + hues) / 2, hues]) * 255
darkest = np.concatenate([np.ones(1 + len(hues)), np.full(len(hues), 0.3)])


# --- plumbing -------------------------------------------------------------------------
def real(extensors: Extensor) -> np.ndarray:
    """Which extensors are real up to round-off, by their coefficients."""
    kernel = extensors.kernel
    return np.abs(kernel.imag).max(axis=-1) <= 1e-6 * np.abs(kernel).max(axis=-1)


def coordinates(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of finite points: their pairings with the coordinate planes, at unit
    weight."""
    mv = points.context.multivector
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points[..., None])).to_array().real


def chords(lines: core.Bivector, radius: float) -> np.ndarray:
    """The two ends of each line's chord of a ball about the origin, NaN for a line that misses it."""
    mv = lines.context.multivector
    unit = lines.normalized()
    nearest = coordinates((core.origin | unit) ^ unit)                        # [..., 3]
    heading = (stack((mv.x, mv.y, mv.z)) & (unit ^ core.horizon)[..., None]).to_array().real   # [..., 3]
    heading = heading / np.linalg.norm(heading, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        half = np.sqrt(radius**2 - (nearest**2).sum(axis=-1, keepdims=True))
    return np.stack([nearest - half * heading, nearest + half * heading], axis=-2)


def finite(ends: np.ndarray) -> np.ndarray:
    """Chords, `[..., 2, 3]`, flattened to `[chords, 2, 3]`, the chords of lines that miss the ball left out."""
    ends = ends.reshape(-1, 2, 3)
    return ends[np.isfinite(ends).all(axis=(-1, -2))]


def rods(ends: np.ndarray, radius: float) -> solids.Quadric:
    """Chords, `[..., 2, 3]`, as rods of a radius, `[chords, 3]`, the chords of lines that miss the ball left out."""
    ends = finite(ends)
    return solids.rods(solids.mv.point(ends[:, 0]), solids.mv.point(ends[:, 1]), radius)


def picture(lines: core.Bivector, rulings: core.Bivector, transversals: core.Bivector,
            crossings: core.Point, view_azimuth: float) -> np.ndarray:
    """The scene ray traced from an azimuth, framed on the ball: `[pixels, pixels, 3]` bytes."""
    view = solids.View(solids.aim(solids.origin, elevation, view_azimuth), np.full(2, ball_radius * (1 + margin)),
                       n_pixels)
    points = solids.mv.point(coordinates(crossings)[real(crossings)])         # [crossings] Point-
    # The hyperboloid's two families of lines, faint: strokes, which the tracer composites sample by sample, where
    # traced rods would each be searched through every layer of the translucent crowd.
    ruling_ends = [solids.mv.point(finite(chords(family, hyperboloid_radius))) for family in rulings]
    hyperboloid = [(strokes.segments(ends[:, 0], ends[:, 1], colour, 2 * ruling_radius, view),
                    # A traced rod is crossed twice, a stroke once: its nearest surface hides what both would.
                    1 - (1 - ruling_opacity) ** 2) for ends, colour in zip(ruling_ends, ruling_colours)]
    four = [(rods(chords(lines[:3], ball_radius), line_radius * view.pixel), line_colour),
            (rods(chords(lines[3:], ball_radius), line_radius * view.pixel), fourth_colour),
            (rods(chords(transversals, ball_radius)[real(transversals)], line_radius * view.pixel), transversal_colour),
            (solids.ball(points, crossing_radius * view.pixel)[:, None], transversal_colour)]
    return solids.draw(four, view, background, inks=[], translucent=[], meshes=hyperboloid)


def flattened(image: np.ndarray) -> np.ndarray:
    """An image, `[rows, columns, 3]` bytes, with every pixel put on the flat tone it is nearest in some light."""
    # How brightly each tone would be lit to match each pixel best, within the shadows it may fall into.
    light = np.clip((image @ tones.T) / (tones**2).sum(axis=-1), darkest, 1.0)   # [rows, columns, tones]
    nearest = ((image[..., None, :] - light[..., None] * tones) ** 2).sum(axis=-1).argmin(axis=-1)
    return np.round(tones[nearest]).astype(np.uint8)


def draw(lines: core.Bivector, rulings: core.Bivector, transversals: core.Bivector,
         crossings: core.Point) -> plt.Figure:
    """The four lines, the hyperboloid of the first three as its two families of lines, and the
    transversals through the points where the fourth line crosses it."""
    return image_figure(picture(lines, rulings, transversals, crossings, azimuth))


def animate(scenes: Iterator[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]],
            n_frames: int) -> Iterator[np.ndarray]:
    """One frame per scene in flat tones, the view turning once around the hyperboloid over the loop of `n_frames`
    scenes."""
    for frame, scene in enumerate(scenes):
        yield flattened(picture(*scene, azimuth + 2 * np.pi * frame / n_frames))
