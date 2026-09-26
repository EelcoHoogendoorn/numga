"""Drawing for the lap: the true, dead-reckoned and most likely paths, with each pose's uncertainty
drawn as an ellipse, the zero level of a quadric on a grid of points. Lengths are in metres."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from numga.algebras import PGA2D
from examples import instantiate

# The drawings are in the plane.
core = instantiate("examples.geometry.odometry.core", PGA2D)
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


def ellipses(poses: core.Motor, uncertainty: core.Covariance) -> Quadric:
    """The quadric SIGMAS standard deviations out, within which each pose carries the origin, from each
    pose's own uncertainty.

    A twist moves the carried point by its commutator with the point; reading that motion with a plane
    is reading the twist with a line, found by solving the incidence form. So the uncertainty of the
    twist gives the uncertainty of the point, whose second moment's inverse, paired twice with a point
    of unit weight, is one plus the squared number of standard deviations to it.
    """
    here = poses >> ORIGIN                                                    # [...] Point
    shift = core.Twist.commutator(here)(poses >> core.Twist)                  # [...] Point <- Twist
    readout = (core.Line & core.Twist).solve(Plane & shift)                   # [...] Line <- Plane
    moment = here * (Plane & here) + shift(uncertainty(readout))              # [...] Point <- Plane
    return moment.inverse() - (1 + SIGMAS**2) * mv.w * (mv.w & Point)         # [...] Plane <- Point


def point(coords: np.ndarray) -> Point:
    """Points of unit weight at (..., 2) coordinates: the dual of the homogeneous vector."""
    return (mv("x y", coords) + mv.w).dual()


def xy(points: Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate lines, at unit weight."""
    return np.stack([((line & points) / (mv.w & points)).to_array() for line in (mv.x, mv.y)], axis=-1)


def extent(*paths: core.Motor) -> np.ndarray:
    """[2, 2] the lower and upper corners of a box around the origins the paths carry, with a margin."""
    corners = np.concatenate([xy(path >> ORIGIN).reshape(-1, 2) for path in paths])
    low, high = corners.min(axis=0), corners.max(axis=0)
    margin = 0.15 * (high - low).max()
    return np.stack([low - margin, high + margin])


def frame(ax, box: np.ndarray) -> None:
    """Equal axes over the box, without frame or ticks."""
    ax.set_xlim(box[:, 0])
    ax.set_ylim(box[:, 1])
    ax.set_aspect("equal")
    ax.set_axis_off()


def contours(ax, poses: core.Motor, uncertainty: core.Covariance, box: np.ndarray, color: str, alpha: float) -> None:
    """Each pose's ellipse over the box: the zero level of its quadric on a grid of points."""
    x, y = np.meshgrid(*np.linspace(box[0], box[1], 240).T)
    grid = point(np.stack([x, y], axis=-1))                                   # [rows, columns] Point
    levels = (ellipses(poses, uncertainty)[:, None, None](grid) & grid).to_array()   # [poses, rows, columns]
    for level in levels:
        ax.contour(x, y, level, levels=[0.0], colors=color, linewidths=0.8, alpha=alpha)


def draw_paths(ax, truth: core.Motor, dead: core.Motor, reckoned: core.Covariance, box: np.ndarray) -> None:
    """The true path, and the dead-reckoned one with each pose's uncertainty."""
    ax.plot(*xy(truth >> ORIGIN).T, color="0.75", linewidth=3.0, label="truth")
    ax.plot(*xy(dead >> ORIGIN).T, "--", color=DEAD, linewidth=1.4, label="dead reckoning")
    contours(ax, dead, reckoned, box, DEAD, 0.35)
    # The known start, where both paths begin.
    ax.plot(*xy(truth[:1] >> ORIGIN).T, "o", color="0.2", markersize=7, zorder=5)


def draw_lap(truth: core.Motor, dead: core.Motor, reckoned: core.Covariance, linked: core.Motor) -> plt.Figure:
    """The true lap and dead reckoning with its uncertainties, and the two dead-reckoned poses a reading
    links."""
    figure, ax = plt.subplots(figsize=(5, 5))
    box = extent(truth, dead)
    draw_paths(ax, truth, dead, reckoned, box)
    ax.plot(*xy(linked >> ORIGIN).T, "o-", color="#2a9d4a", markersize=5, linewidth=1.5, label="closing reading")
    frame(ax, box)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def draw(truth: core.Motor, dead: core.Motor, reckoned: core.Covariance, poses: core.Motor,
         uncertainty: core.Covariance) -> plt.Figure:
    """The true path, dead reckoning with its uncertainties, and the poses with theirs."""
    figure, ax = plt.subplots(figsize=(5, 5))
    box = extent(truth, dead)
    draw_paths(ax, truth, dead, reckoned, box)
    contours(ax, poses, uncertainty, box, LIKELY, 0.7)
    ax.plot(*xy(poses >> ORIGIN).T, "o-", color=LIKELY, markersize=2.5, linewidth=1.0, label="most likely")
    frame(ax, box)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def still(figure: plt.Figure) -> np.ndarray:
    """The figure's pixels, the figure closed."""
    image = capture(figure)
    plt.close(figure)
    return image


def animate(truth: core.Motor, dead: core.Motor, reckoned: core.Covariance, closing) -> list[np.ndarray]:
    """A frame for each iteration of the closing, from the poses and uncertainties it yields."""
    return [still(draw(truth, dead, reckoned, poses, uncertainty)) for poses, uncertainty in closing]
