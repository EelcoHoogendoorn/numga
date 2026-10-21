"""Drawing for the Kalman example: the drive animated, with the trails of the true, dead-reckoned and filtered
paths, the readings so far, and the filter's 2σ position ellipse."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.animation import capture
from examples.estimation.kalman.core import Line, Motor, Point, Scalar, mv

TRUE_COLOUR, DEAD_COLOUR, FILTER_COLOUR, READING_COLOUR = "#1f2329", "#d0482b", "#2c6fb4", "#3a9a5b"
# A vehicle's outline in its own frame, pointing along +x; the view's margin around the whole drive, as a
# fraction of its larger side, and the figure's width in inches.
OUTLINE = [[0.32, 0.0], [-0.18, 0.16], [-0.18, -0.16]]
MARGIN = 0.08
WIDTH = 6.4


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the lines x and y, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def principal_axes(variances: Scalar, axes: Line) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) variances and (n, 2, 2) unit axis directions, the normals of the axis lines, read by the metric
    against the lines x and y."""
    normals = (axes[..., None] | stack((mv.x, mv.y))).to_array()
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return variances.to_array(), normals.swapaxes(-1, -2)


def ellipse(centre: np.ndarray, values: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """The 2σ ellipse of a position covariance with the given principal variances and axes, `[samples, 2]`."""
    t = np.linspace(0.0, 2.0 * np.pi, 60)
    return centre + (vectors @ (2.0 * np.sqrt(np.clip(values, 0, None))[:, None] * np.stack([np.cos(t), np.sin(t)]))).T


def outlines(poses: Motor) -> np.ndarray:
    """Each pose's vehicle outline, `[..., corners, 2]`: the outline's corners carried by the motor."""
    return xy(poses[..., None] >> mv.point(OUTLINE))


def animate_tracking(truth: Motor, dead: Motor, estimates: Motor, here: Point, variances: Scalar, axes: Line,
                     readings: Motor) -> list[np.ndarray]:
    """One frame per state of the drive: the trails so far of the true path, black, the dead-reckoned one,
    dashed red, and the filtered one, blue; the readings so far, green crosses; the filter's 2σ position
    ellipse; and the three vehicles at their current poses."""
    origin = mv.w.dual()
    true_xy, dead_xy, est_xy, reading_xy = xy(truth >> origin), xy(dead >> origin), xy(here), xy(readings >> origin)
    values, vectors = principal_axes(variances, axes)
    shapes = {colour: outlines(poses) for colour, poses in ((TRUE_COLOUR, truth), (DEAD_COLOUR, dead), (FILTER_COLOUR, estimates))}
    # A reading is new where it differs from the one before.
    fresh = np.r_[False, np.any(np.diff(reading_xy, axis=0) != 0, axis=-1)]
    everything = np.concatenate([true_xy, dead_xy, est_xy, reading_xy[fresh],
                                 *[shape.reshape(-1, 2) for shape in shapes.values()],
                                 *[ellipse(centre, value, vector) for centre, value, vector in zip(est_xy, values, vectors)]])
    low, high = everything.min(axis=0), everything.max(axis=0)
    low, high = low - MARGIN * (high - low).max(), high + MARGIN * (high - low).max()

    fig = plt.figure(figsize=(WIDTH, WIDTH * (high[1] - low[1]) / (high[0] - low[0])), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    frames = []
    for step in range(len(true_xy)):
        ax.cla()
        ax.plot(*true_xy[:step + 1].T, color=TRUE_COLOUR, linewidth=1.6)
        ax.plot(*dead_xy[:step + 1].T, color=DEAD_COLOUR, linestyle="--", linewidth=1.2)
        ax.plot(*est_xy[:step + 1].T, color=FILTER_COLOUR, linewidth=1.2)
        seen = reading_xy[:step + 1][fresh[:step + 1]]
        ax.plot(*seen.T, marker="x", color=READING_COLOUR, linestyle="none", markersize=7, markeredgewidth=1.8)
        ax.fill(*ellipse(est_xy[step], values[step], vectors[step]).T, color=FILTER_COLOUR, alpha=0.18, linewidth=0)
        ax.plot(*ellipse(est_xy[step], values[step], vectors[step]).T, color=FILTER_COLOUR, linewidth=1.0)
        for colour, shape in shapes.items():
            ax.fill(*shape[step].T, color=colour, alpha=0.9, linewidth=0)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
        ax.set_axis_off()
        frames.append(capture(fig))
    plt.close(fig)
    return frames
