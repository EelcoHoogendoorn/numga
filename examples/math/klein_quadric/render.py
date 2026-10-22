"""Drawing for the four-line problem: every line as its chord of a ball about the origin; the lines
on the hyperboloid faint, the three lines that fix it dark, the fourth orange, and the transversals
red where they are real."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor
from examples.math.klein_quadric import core
from examples.wgsl import compose, image_figure, solids

mv = core.mv
# Lines are drawn inside this ball about the origin, seen from this elevation and, at first, this azimuth, in
# radians, azimuth zero looking along +y; the margin around the ball, as a share of its radius; the pixels across.
RADIUS = 3.0
ELEVATION, AZIMUTH = np.radians([18.0, 30.0])
MARGIN = 0.06
N_PIXELS = 650
# Radii in pixels: of the lines on the hyperboloid, of the four lines and the transversals, and of the crossings;
# and how much of what lies behind the lines on the hyperboloid they hide, so that they stay faint.
RULING_RADIUS, LINE_RADIUS, CROSSING_RADIUS = 1.1, 2.4, 6.0
RULING_OPACITY = 0.5
RULING_COLOURS = [np.array(to_rgb(colour)) for colour in ("#9ecae1", "#c7c7c7")]
LINE_COLOUR, FOURTH_COLOUR, TRANSVERSAL_COLOUR = (np.array(to_rgb(colour)) for colour in ("#08306b", "#e67e22", "#c0392b"))
BACKGROUND = np.ones(3)
AXES = mv(core.Plane.derive.nondegenerate, np.eye(3))                          # [3] Plane


# --- plumbing -------------------------------------------------------------------------
def real(extensors: Extensor) -> np.ndarray:
    """Which extensors are real up to round-off, by their coefficients."""
    kernel = extensors.kernel
    return np.abs(kernel.imag).max(axis=-1) <= 1e-6 * np.abs(kernel).max(axis=-1)


def coordinates(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of finite points: their pairings with the coordinate planes, at unit
    weight."""
    return ((AXES & points[..., None]) / (mv.w & points[..., None])).to_array().real


def chords(lines: core.Bivector) -> np.ndarray:
    """The two ends of each line's chord of the ball, NaN for a line that misses it."""
    unit = lines.normalized()
    nearest = coordinates((core.ORIGIN | unit) ^ unit)                        # [..., 3]
    heading = (AXES & (unit ^ core.HORIZON)[..., None]).to_array().real       # [..., 3]
    heading = heading / np.linalg.norm(heading, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        half = np.sqrt(RADIUS**2 - (nearest**2).sum(axis=-1, keepdims=True))
    return np.stack([nearest - half * heading, nearest + half * heading], axis=-2)


def rods(ends: np.ndarray, radius: float) -> solids.Quadric:
    """Chords, `[..., 2, 3]`, as rods of a radius, `[chords, 3]`, the chords of lines that miss the ball left out."""
    ends = ends.reshape(-1, 2, 3)
    ends = ends[np.isfinite(ends).all(axis=(-1, -2))]
    return solids.rods(solids.mv.point(ends[:, 0]), solids.mv.point(ends[:, 1]), radius)


def picture(lines: core.Bivector, rulings: core.Bivector, across: core.Bivector, crossing: core.Point,
            azimuth: float) -> np.ndarray:
    """The scene ray traced from an azimuth, framed on the ball: `[pixels, pixels, 3]` bytes."""
    view = solids.View(solids.aim(solids.origin, ELEVATION, azimuth), np.full(2, RADIUS * (1 + MARGIN)), N_PIXELS)
    points = solids.mv.point(coordinates(crossing)[real(crossing)])            # [crossings] Point
    hyperboloid = solids.layer([(rods(chords(rulings[0]), RULING_RADIUS * view.pixel), RULING_COLOURS[0]),
                                (rods(chords(rulings[1]), RULING_RADIUS * view.pixel), RULING_COLOURS[1])], view)
    four = solids.layer([(rods(chords(lines[:3]), LINE_RADIUS * view.pixel), LINE_COLOUR),
                         (rods(chords(lines[3:]), LINE_RADIUS * view.pixel), FOURTH_COLOUR),
                         (rods(chords(across)[real(across)], LINE_RADIUS * view.pixel), TRANSVERSAL_COLOUR),
                         (solids.ball(points, CROSSING_RADIUS * view.pixel)[:, None], TRANSVERSAL_COLOUR)], view)
    return compose([(*hyperboloid, RULING_OPACITY), (*four, 1.0)], BACKGROUND)


def draw(lines: core.Bivector, rulings: core.Bivector, across: core.Bivector, crossing: core.Point,
         azimuth: float) -> plt.Figure:
    """The four lines, the hyperboloid of the first three as its two families of lines, and the
    transversals through the points where the fourth line crosses it."""
    return image_figure(picture(lines, rulings, across, crossing, azimuth))


def animate(scenes: list[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]]) -> list[np.ndarray]:
    """One frame per scene, the view turning once around the hyperboloid over the loop."""
    return [picture(*scene, azimuth=AZIMUTH + 2 * np.pi * index / len(scenes)) for index, scene in enumerate(scenes)]
