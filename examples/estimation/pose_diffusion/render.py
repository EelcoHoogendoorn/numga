"""Render the station-keeping clouds, their two-standard-deviation position quadrics, and the ellipsoid half-widths against the controller gain."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import quadrics, simplices
from examples.util.wgsl.camera import Camera, panel
from examples.util.wgsl.simplices import Cap, Topology
from examples.estimation.pose_diffusion.core import Covariance, Motor, Plane, Point, Spread, Twist, ga, mv, origin

ELLIPSOID_RGB = np.array(mcolors.to_rgb("#c0392b"))
HEADING_FRACTION = 0.08
SIGMAS = 2.0
POINT = 100 / 72
Quadric = ga.gatype((Plane, Point))
# The bodies drawn: the first so many of the fleet, few enough for each to read as an arrow along its heading; the
# arrows' colour, their shafts' width and the length of the strokes of the chevrons at their tips, in pixels; and the
# faint cross's colour.
N_SHOWN = 120
ARROW_RGB = np.array(mcolors.to_rgb("#4a6fa5"))
ARROW_WIDTH, ARROW_CHEVRON = 1.5, 6.0
CROSS_RGB = np.full(3, 0.85)


# --- plumbing -------------------------------------------------------------------------
def position_ellipsoid(covariance: Covariance, sigmas: float) -> Quadric:
    """The confidence quadric `sigmas` standard deviations out around the set point."""
    shift = Twist.commutator(origin)                                            # [] Point- <- Twist+
    readout = shift.adjugate()                                                  # [] Line- <- Plane+
    moment = origin * (Plane & origin) + shift(covariance(readout))             # [...] Point- <- Plane+
    return moment.inverse() - (1 + sigmas**2) * mv.w * (mv.w & Point)           # [...] Plane+ <- Point-


def pictures(poses: Motor, predicted: Covariance, limit: Covariance, extent: float, camera: Camera) -> list[np.ndarray]:
    """A faint cross through the set point, the shaded steady-state ellipsoid, a sample of the bodies drawn as short
    arrows along their headings, and the steady-state and predicted ellipsoids' curves."""
    steady_quadric = position_ellipsoid(limit, SIGMAS)[None]                    # [1] Plane+ <- Point-
    predicted_quadric = position_ellipsoid(predicted, SIGMAS)[None]             # [1] Plane+ <- Point-
    cross = origin + mv.direction([[[-extent, 0.0], [extent, 0.0]], [[0.0, -extent], [0.0, extent]]])   # [2, 2] Point-
    # A point a short way ahead of the set point along the bow, and each body's arrow about its place, along the bow.
    bow = origin + mv.direction([HEADING_FRACTION * extent, 0.0])               # [] Point-
    shown = poses[:N_SHOWN]                                                     # [shown] Motor+
    positions = (shown >> origin).normalized()                                  # [shown] Point-
    headings = (shown >> bow).normalized() - positions                          # [shown] Point-
    arrows = stack((positions - headings / 2, positions + headings / 2), axis=-1)   # [shown, ends] Point-
    return [
        simplices.strokes(cross.reshape(-1), Topology.chains([2, 2]), CROSS_RGB, charts.FRAME_PX, camera, cap=Cap.BUTT),
        quadrics.regions(steady_quadric[:, None], ELLIPSOID_RGB, camera, 0.14),
        simplices.strokes(arrows.reshape(-1), Topology.chains([2] * len(shown)), ARROW_RGB, ARROW_WIDTH, camera,
                          chevrons=np.ones(len(shown)), chevron=ARROW_CHEVRON),
        quadrics.curves(steady_quadric, 1.0 * POINT, ELLIPSOID_RGB, camera, 0.45),
        quadrics.curves(predicted_quadric, 1.4 * POINT, ELLIPSOID_RGB, camera),
    ]


def draw_cloud(ax, poses: Motor, predicted: Covariance, limit: Covariance, extent: float) -> tuple:
    """Draw a sample of the bodies as short arrows along their headings over the predicted and settled position
    quadrics, with a faint cross through the set point, its directions named at their ends; the image and its
    camera, for an animation to draw anew."""
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    camera = panel(ax, mv.x, mv.y, mv.w)
    image = simplices.overlay(ax, pictures(poses, predicted, limit, extent, camera))
    ax.text(extent, 0, "forward", ha="right", va="top", color="0.4")
    ax.text(0, extent, "sideways ", ha="right", va="bottom", color="0.4", rotation=90, rotation_mode="anchor")
    return image, camera


@charts.style()
def clouds(states: dict, extent: float) -> tuple[plt.Figure, list]:
    """The figure of fleet clouds over their predicted and settled quadrics, together with its animated artists."""
    # Half the column a cloud: hundreds of arrows over the quadrics, two clouds filling the column.
    figure, row = charts.subplots(400 * len(states), 400, ncols=len(states), layout="constrained")
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
    # Lay the figure out once on its first frame; only the pictures change between frames.
    figure.set_layout_engine("none")
    for states in steps:
        for (image, camera), (poses, predicted, limit) in zip(artists, states):
            image.set_data(simplices.composited(pictures(poses, predicted, limit, extent, camera)))
        yield capture(figure)
    plt.close(figure)


@charts.style()
def draw_envelope(gains: np.ndarray, spreads: Spread, tolerance: float) -> plt.Figure:
    """Plot the two-standard-deviation half-widths of the position ellipsoid along its long and short axes against the controller gain."""
    # The half-widths of the drawn quadrics, SIGMAS standard deviations out, the widest first.
    half_widths = np.sort((SIGMAS * spreads.eigvalsh().square_root()).to_array(), axis=-1)[:, ::-1]   # [gains, modes]
    # Half the column: two smooth curves, the tolerance and the gain that meets it.
    figure, ax = charts.subplots(420, 280)
    ax.plot(gains, half_widths[:, 0], color="#c0392b")
    ax.plot(gains, half_widths[:, 1], color="#4a6fa5")
    ax.axhline(tolerance, color="0.4", linestyle="--", linewidth=charts.FRAME_WIDTH)
    # The gains are ascending, so the first gain meeting the tolerance is the smallest.
    min_gain = float(gains[half_widths[:, 0] <= tolerance][0])
    ax.axvline(min_gain, color="0.4", linestyle=":", linewidth=charts.FRAME_WIDTH)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, gains[-1])
    ax.set_ylim(0, half_widths[:, 0].max() * 1.05)
    ax.set_xticks([0, round(min_gain, 3), gains[-1]])
    ax.set_yticks([0, tolerance])
    ax.set_xlabel("controller gain (1/s)")
    ax.set_ylabel("2σ half-width (m)")
    figure.tight_layout()
    return figure
