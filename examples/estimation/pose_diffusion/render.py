"""Render the station-keeping clouds, their two-standard-deviation position quadrics, and the ellipse half-widths against the controller gain."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.util.animation import capture
from examples.util.wgsl import pinned, planar, room
from examples.util.wgsl.planar import Curves, Regions, filling
from examples.estimation.pose_diffusion.core import Covariance, Motor, Point, Spread, Twist, ga, mv, origin

ELLIPSE_RGB = np.array(mcolors.to_rgb("#c0392b"))
HEADING_FRACTION = 0.08
SIGMAS = 2.0
POINT = 100 / 72
Line = ga.gatype.vector()                           # Vector+
Quadric = ga.gatype((Line, Point))


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """The x and y coordinates of each point, from its pairings with the coordinate lines."""
    return (stack((mv.x, mv.y)) & points.normalized()[..., None]).to_array()


def position_ellipse(covariance: Covariance, sigmas: float) -> Quadric:
    """The confidence quadric `sigmas` standard deviations out around the set point."""
    shift = Twist.commutator(origin)                                            # [] Point- <- Twist+
    readout = shift.adjugate()                                                  # [] Line- <- Line+
    moment = origin * (Line & origin) + shift(covariance(readout))              # [...] Point- <- Line+
    return moment.inverse() - (1 + sigmas**2) * mv.w * (mv.w & Point)           # [...] Line+ <- Point-


def quadric_layers(predicted: Covariance, limit: Covariance) -> list:
    """The planar layers for the shaded steady-state ellipse and the predicted ellipse curve."""
    steady_quadric = position_ellipse(limit, SIGMAS)[None]                      # [1] Line+ <- Point-
    predicted_quadric = position_ellipse(predicted, SIGMAS)[None]               # [1] Line+ <- Point-
    return [
        Regions(steady_quadric, ELLIPSE_RGB, 0.14),
        Curves(steady_quadric, 1.0 * POINT, ELLIPSE_RGB, 0.45),
        Curves(predicted_quadric, 1.4 * POINT, ELLIPSE_RGB, 1.0),
    ]


def arrows(poses: Motor, extent: float) -> tuple[np.ndarray, np.ndarray]:
    """The bodies' positions and heading displacement vectors `[bodies, 2]`."""
    # A point a short way ahead of the set point along the bow.
    bow = origin + mv.direction([HEADING_FRACTION * extent, 0.0])               # [] Point-
    positions = coordinates(poses >> origin)                                    # [bodies, 2]
    return positions, coordinates(poses >> bow) - positions


def draw_cloud(ax, poses: Motor, predicted: Covariance, limit: Covariance, extent: float) -> tuple:
    """Draw the bodies as short arrows along their headings over the predicted and settled position quadrics."""
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    view = filling(np.array([-extent, extent]), np.array([-extent, extent]), room(ax))
    image = pinned(ax, planar.draw(view, quadric_layers(predicted, limit), np.ones(3)), view.extent)
    positions, headings = arrows(poses, extent)
    quiver = ax.quiver(positions[:, 0], positions[:, 1], headings[:, 0], headings[:, 1], angles="xy",
                       scale_units="xy", scale=1, pivot="mid", width=0.002, color="#4a6fa5", alpha=0.35)
    # A faint cross through the set point with its directions named at their ends.
    ax.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.7, zorder=0)
    ax.text(extent, 0, "forward", ha="right", va="top", fontsize=9, color="0.4")
    ax.text(0, extent, "sideways ", ha="right", va="bottom", fontsize=9, color="0.4", rotation=90, rotation_mode="anchor")
    return quiver, image, view


def clouds(states: dict, extent: float) -> tuple[plt.Figure, list]:
    """The figure of fleet clouds over their predicted and settled quadrics, together with its animated artists."""
    figure, row = plt.subplots(1, len(states), figsize=(5.5 * len(states), 5.5), dpi=100, layout="constrained")
    figure.canvas.draw()
    artists = [draw_cloud(ax, poses, predicted, limit, extent)
               for ax, (poses, predicted, limit) in zip(np.atleast_1d(row), states.values())]
    return figure, artists


def draw_settled(states: dict, extent: float) -> plt.Figure:
    """Draw the clouds of each named run over their predicted and settled position quadrics."""
    figure, _ = clouds(states, extent)
    return figure


def animate_clouds(runs: dict, extent: float) -> Iterator[np.ndarray]:
    """Yield one frame per simulation state as the fleet clouds spread and the predicted quadric grows toward the steady state."""
    steps = zip(*runs.values())
    figure, artists = clouds(dict(zip(runs, next(steps))), extent)
    yield capture(figure)
    # Lay the figure out once on its first frame; only the arrows and the quadric images move between frames.
    figure.set_layout_engine("none")
    for states in steps:
        for (quiver, image, view), (poses, predicted, limit) in zip(artists, states):
            positions, headings = arrows(poses, extent)
            quiver.set_offsets(positions)
            quiver.set_UVC(headings[:, 0], headings[:, 1])
            image.set_data(planar.draw(view, quadric_layers(predicted, limit), np.ones(3)))
        yield capture(figure)
    plt.close(figure)


def draw_envelope(gains: np.ndarray, spreads: Spread, tolerance: float) -> plt.Figure:
    """Plot the two-standard-deviation half-widths of the position ellipse along its long and short axes against the controller gain."""
    variances = spreads.eigvalsh().to_array()                                   # [gains, modes]
    half_widths = 2.0 * np.sqrt(np.sort(variances, axis=-1)[:, ::-1])
    figure, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gains, half_widths[:, 0], color="#c0392b")
    ax.plot(gains, half_widths[:, 1], color="#4a6fa5")
    ax.axhline(tolerance, color="0.4", linestyle="--", linewidth=1)
    # The gains are ascending, so the first gain meeting the tolerance is the smallest.
    min_gain = float(gains[half_widths[:, 0] <= tolerance][0])
    ax.axvline(min_gain, color="0.4", linestyle=":", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, gains[-1])
    ax.set_ylim(0, half_widths[:, 0].max() * 1.05)
    ax.set_xticks([0, round(min_gain, 3), gains[-1]])
    ax.set_yticks([0, tolerance])
    ax.set_xlabel("controller gain (1/s)")
    ax.set_ylabel("2σ half-width (m)")
    figure.tight_layout()
    return figure
