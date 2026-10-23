"""Drawing for the lap: the true, dead-reckoned and most likely paths, with each pose's uncertainty drawn as an
ellipse, the curve where a quadric vanishes, on the GPU by `planar`; the legend over it in matplotlib. Lengths are in
metres."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from examples.util.animation import capture
from examples.util.wgsl import pinned, room
from examples.util.wgsl import planar
from examples.util.wgsl.planar import SOLID, Cap, Curves, Marker, Marks, Strokes, Window, filling
from numga.algebras import PGA2D
from examples import instantiate

# The drawings are in the plane.
core = instantiate("examples.estimation.odometry.core", PGA2D)
mv = core.mv
Point = PGA2D.gatype.antivector()
Plane = PGA2D.gatype.vector()
Quadric = PGA2D.gatype((Plane, Point))                                        # Plane <- Point
# The point each pose carries: the origin, dual to the weight direction w.
ORIGIN = mv.w.dual()                                                          # [] Point
# How far out the ellipses lie, in standard deviations, and the colours of dead reckoning and of the most
# likely poses.
SIGMAS = 2.0
DEAD = "#d9822b"
LIKELY = "#2c5d9e"
# The colours of the true path, the closing reading and the known start.
TRUTH, CLOSING, START = "0.75", "#2a9d4a", "0.2"
# Pixels a point, at the figure's 100 dots per inch. The widths of the true, dead-reckoned and most likely paths, the
# closing reading and the ellipses; the dead-reckoned path's dashes, on and off, in widths of its stroke; and the
# widths of the marks of the most likely poses, the closing reading and the start, their outlines a point wide.
POINT = 100 / 72
TRUTH_WIDTH, DEAD_WIDTH, LIKELY_WIDTH, CLOSING_WIDTH, ELLIPSE_WIDTH = 3.0, 1.4, 1.0, 1.5, 0.8
DASHED = (3.7, 1.6)
LIKELY_MARK, CLOSING_MARK, START_MARK = 2.5, 5.0, 7.0


def ellipses(poses: core.Motor, uncertainty: core.Covariance) -> Quadric:
    """The quadric SIGMAS standard deviations out, within which each pose carries the origin, from each
    pose's own uncertainty.

    A twist moves the carried point by its commutator with the point; reading that motion with a plane
    is reading the twist with a forque, found by the motion's adjugate. So the uncertainty of the
    twist gives the uncertainty of the point, whose second moment's inverse, paired twice with a point
    of unit weight, is one plus the squared number of standard deviations to it.
    """
    here = poses >> ORIGIN                                                    # [...] Point
    shift = core.Twist.commutator(here)(poses >> core.Twist)                  # [...] Point <- Twist
    readout = shift.adjugate()                                                # [...] Forque <- Plane
    moment = here * (Plane & here) + shift(uncertainty(readout))              # [...] Point <- Plane
    return moment.inverse() - (1 + SIGMAS**2) * mv.w * (mv.w & Point)         # [...] Plane <- Point


def xy(points: Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate lines, at unit weight."""
    return np.stack([((line & points) / (mv.w & points)).to_array() for line in (mv.x, mv.y)], axis=-1)


def extent(*paths: core.Motor) -> np.ndarray:
    """[2, 2] the lower and upper corners of a box around the origins the paths carry, with a margin."""
    corners = np.concatenate([xy(path >> ORIGIN).reshape(-1, 2) for path in paths])
    low, high = corners.min(axis=0), corners.max(axis=0)
    margin = 0.15 * (high - low).max()
    return np.stack([low - margin, high + margin])


def rgb(colour: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(colour))


def mark(points: Point, size: float, colour: str) -> Marks:
    """Points as discs `size` points across, with an outline a point wide in their colour."""
    return Marks(points, Marker.DISC, 0.0, (size + 1.0) / 2 * POINT, rgb(colour), 1.0)


def paths(truth: core.Motor, dead: core.Motor, reckoned: core.Covariance) -> list:
    """The true path, and the dead-reckoned one, dashed, with each pose's uncertainty."""
    return [Strokes((truth >> ORIGIN)[None], TRUTH_WIDTH * POINT, rgb(TRUTH), 1.0, SOLID, Cap.PROJECTING),
            Strokes((dead >> ORIGIN)[None], DEAD_WIDTH * POINT, rgb(DEAD), 1.0,
                    (DASHED[0] * DEAD_WIDTH * POINT, DASHED[1] * DEAD_WIDTH * POINT), Cap.BUTT),
            Curves(ellipses(dead, reckoned), ELLIPSE_WIDTH * POINT, rgb(DEAD), 0.35)]


def likely(poses: core.Motor, uncertainty: core.Covariance) -> list:
    """The most likely poses with their uncertainties."""
    return [Curves(ellipses(poses, uncertainty), ELLIPSE_WIDTH * POINT, rgb(LIKELY), 0.7),
            Strokes((poses >> ORIGIN)[None], LIKELY_WIDTH * POINT, rgb(LIKELY), 1.0, SOLID, Cap.PROJECTING),
            mark(poses >> ORIGIN, LIKELY_MARK, LIKELY)]


def start(truth: core.Motor) -> Marks:
    """The known start, where both paths begin."""
    return mark(truth[:1] >> ORIGIN, START_MARK, START)


def framed(box: np.ndarray, labelled: Line2D) -> tuple[plt.Figure, plt.Axes, Window]:
    """A figure for the box, its axes, and the window that fills the room they get; the legend of the true and
    dead-reckoned paths and what else is labelled."""
    figure, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    figure.tight_layout()
    view = filling(box[:, 0], box[:, 1], room(ax))
    entries = [Line2D([], [], color=TRUTH, linewidth=TRUTH_WIDTH, label="truth"),
               Line2D([], [], linestyle="--", color=DEAD, linewidth=DEAD_WIDTH, label="dead reckoning"), labelled]
    ax.legend(handles=entries, loc="lower left", fontsize=8, frameon=False)
    return figure, ax, view


def most_likely() -> Line2D:
    """The legend's entry for the most likely poses."""
    return Line2D([], [], marker="o", color=LIKELY, markersize=LIKELY_MARK, linewidth=LIKELY_WIDTH, label="most likely")


def picture(view: Window, layers: list) -> np.ndarray:
    """The layers over white, in the window: RGB bytes."""
    return planar.draw(view, layers, np.ones(3))


def draw_lap(truth: core.Motor, dead: core.Motor, reckoned: core.Covariance, linked: core.Motor) -> plt.Figure:
    """The true lap and dead reckoning with its uncertainties, and the two dead-reckoned poses a reading
    links."""
    closing = Line2D([], [], marker="o", color=CLOSING, markersize=CLOSING_MARK, linewidth=CLOSING_WIDTH,
                     label="closing reading")
    figure, ax, view = framed(extent(truth, dead), closing)
    linked = linked >> ORIGIN                                                 # [2] Point
    layers = paths(truth, dead, reckoned) + [
        Strokes(linked[None], CLOSING_WIDTH * POINT, rgb(CLOSING), 1.0, SOLID, Cap.PROJECTING),
        mark(linked, CLOSING_MARK, CLOSING), start(truth)]
    pinned(ax, picture(view, layers), view.extent)
    return figure


def draw(truth: core.Motor, dead: core.Motor, reckoned: core.Covariance, poses: core.Motor,
         uncertainty: core.Covariance) -> plt.Figure:
    """The true path, dead reckoning with its uncertainties, and the poses with theirs."""
    figure, ax, view = framed(extent(truth, dead), most_likely())
    pinned(ax, picture(view, paths(truth, dead, reckoned) + likely(poses, uncertainty) + [start(truth)]), view.extent)
    return figure


def animate(truth: core.Motor, dead: core.Motor, reckoned: core.Covariance, closing) -> Iterator[np.ndarray]:
    """A frame for each iteration of the closing, from the poses and uncertainties it yields, over the
    true path and dead reckoning."""
    figure, ax, view = framed(extent(truth, dead), most_likely())
    shown = pinned(ax, np.full(view.shape + (3,), 255, dtype=np.uint8), view.extent)
    under = paths(truth, dead, reckoned)
    for poses, uncertainty in closing:
        shown.set_data(picture(view, under + likely(poses, uncertainty) + [start(truth)]))
        yield capture(figure)
    plt.close(figure)
