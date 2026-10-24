"""Drawing for the four-line problem: every line as its chord of a ball about the origin; the lines
on the hyperboloid faint, the three lines that fix it dark, the fourth orange, and the transversals
red where they are real."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor
from examples.math.klein_quadric import core
from examples.util.wgsl import image_figure, solids, strokes

mv = core.mv
# Lines are drawn inside this ball about the origin, seen from this elevation and, at first, this azimuth, in
# radians, azimuth zero looking along +y; the margin around the ball, as a share of its radius; the pixels across.
RADIUS = 3.0
ELEVATION, AZIMUTH = np.radians([18.0, 30.0])
MARGIN = 0.06
N_PIXELS = 650
# Radii in pixels: of the lines on the hyperboloid, of the four lines and the transversals, and of the crossings;
# and how much of what lies behind them each surface of the lines on the hyperboloid hides, so that they stay faint.
RULING_RADIUS, LINE_RADIUS, CROSSING_RADIUS = 1.1, 2.4, 6.0
RULING_OPACITY = 0.3
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


def finite(ends: np.ndarray) -> np.ndarray:
    """Chords, `[..., 2, 3]`, flattened to `[chords, 2, 3]`, the chords of lines that miss the ball left out."""
    ends = ends.reshape(-1, 2, 3)
    return ends[np.isfinite(ends).all(axis=(-1, -2))]


def rods(ends: np.ndarray, radius: float) -> solids.Quadric:
    """Chords, `[..., 2, 3]`, as rods of a radius, `[chords, 3]`, the chords of lines that miss the ball left out."""
    ends = finite(ends)
    return solids.rods(solids.mv.point(ends[:, 0]), solids.mv.point(ends[:, 1]), radius)


def picture(lines: core.Bivector, rulings: core.Bivector, across: core.Bivector, crossing: core.Point,
            azimuth: float) -> np.ndarray:
    """The scene ray traced from an azimuth, framed on the ball: `[pixels, pixels, 3]` bytes."""
    view = solids.View(solids.aim(solids.origin, ELEVATION, azimuth), np.full(2, RADIUS * (1 + MARGIN)), N_PIXELS)
    points = solids.mv.point(coordinates(crossing)[real(crossing)])            # [crossings] Point
    # The hyperboloid's two families of lines, faint: strokes, which the tracer composites sample by sample, where
    # traced rods would each be searched through every layer of the translucent crowd.
    ruling_ends = [solids.mv.point(finite(chords(family))) for family in rulings]   # [chords, 2] Point each
    hyperboloid = [(strokes.segments(ends[:, 0], ends[:, 1], colour, 2 * RULING_RADIUS, view),
                    # A traced rod is crossed twice, a stroke once: its nearest surface hides what both would.
                    1 - (1 - RULING_OPACITY) ** 2) for ends, colour in zip(ruling_ends, RULING_COLOURS)]
    four = [(rods(chords(lines[:3]), LINE_RADIUS * view.pixel), LINE_COLOUR),
            (rods(chords(lines[3:]), LINE_RADIUS * view.pixel), FOURTH_COLOUR),
            (rods(chords(across)[real(across)], LINE_RADIUS * view.pixel), TRANSVERSAL_COLOUR),
            (solids.ball(points, CROSSING_RADIUS * view.pixel)[:, None], TRANSVERSAL_COLOUR)]
    return solids.draw(four, view, BACKGROUND, meshes=hyperboloid)


def draw(lines: core.Bivector, rulings: core.Bivector, across: core.Bivector, crossing: core.Point,
         azimuth: float = AZIMUTH) -> plt.Figure:
    """The four lines, the hyperboloid of the first three as its two families of lines, and the
    transversals through the points where the fourth line crosses it."""
    return image_figure(picture(lines, rulings, across, crossing, azimuth))


def animate(scenes: list[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]]) -> Iterator[np.ndarray]:
    """One frame per scene, the view turning once around the hyperboloid over the loop."""
    for index, scene in enumerate(scenes):
        yield picture(*scene, azimuth=AZIMUTH + 2 * np.pi * index / len(scenes))
