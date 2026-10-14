"""Drawing for the elastica: the curled strip from straight to a double ring, Euler's elastica between their
pins on the line of thrust, the sweep through them, and Kirchhoff's helices with a ribbon showing the twist of
the cross-sections."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.mechanics.elastica import core

ga = core.ga
# The colours of the rods by case, of the line of thrust and the pins, and of the ribbon; the margin around
# the drawn rods, the panels of Euler's plate per row, the spacing of the helices as a share of the widest,
# and their view as (elevation, azimuth).
ROD_MAP = "viridis"
ROD, THRUST, RIBBON = "#1f6f99", "0.55", "#7cc4e4"
MARGIN = 0.08
COLUMNS = 4
SPACING = 1.4
VIEW = (12, -80)
# The uncoiling wire's view: its elevation, and the azimuths the camera turns through over the frames.
UNCOIL_ELEVATION = 16
UNCOIL_AZIMUTHS = (-120, 60)
# The telephone cord's view as (elevation, azimuth), and the colour of its clamps.
CORD_VIEW = (20, -64)
CLAMP = "0.2"


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for checks."""
    return values.cast(ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of points, on the last axis."""
    kernel = points.cast(ga.subspace("ywz wxz xwy xyz")).kernel
    return kernel[..., :3] / kernel[..., 3:]


def plane(points: core.Point) -> np.ndarray:
    """Points of the plane of bending, along the unbent rod and up, on the last axis."""
    return coordinates(points)[..., [2, 1]]


def frame_plane(ax: plt.Axes, low: np.ndarray, high: np.ndarray) -> None:
    ax.set(xlim=(low[0] - MARGIN, high[0] + MARGIN), ylim=(low[1] - MARGIN, high[1] + MARGIN), aspect="equal")
    ax.set_axis_off()


def draw_curls(centres: core.Point) -> plt.Figure:
    """The strip curled by each couple `[curls, pieces + 1]`, from straight, darkest, to a double ring, with
    its clamp."""
    curves = plane(centres)                                                  # [curls, pieces + 1, 2]
    colours = plt.get_cmap(ROD_MAP)(np.linspace(0.05, 0.9, len(curves)))
    figure, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)
    for curve, colour in zip(curves, colours):
        ax.plot(*curve.T, color=colour, lw=2.2)
    low, high = curves.reshape(-1, 2).min(axis=0), curves.reshape(-1, 2).max(axis=0)
    ax.plot([0, 0], [low[1] - MARGIN, high[1] + MARGIN], color="0.2", lw=4, solid_capstyle="butt")
    frame_plane(ax, low, high)
    figure.tight_layout()
    return figure


def draw_rod(ax: plt.Axes, curve: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
    """One of Euler's elastica: the line of thrust, the rod, and its pins at its two ends."""
    ax.axhline(0, color=THRUST, lw=0.8)
    ax.plot(*curve.T, color=ROD, lw=2.2)
    ax.plot(*curve[[0, -1]].T, "^", color="0.2", markersize=6)
    frame_plane(ax, low, high)


def draw_plate(centres: core.Point) -> plt.Figure:
    """Euler's elastica at each end angle `[angles, pieces + 1]`, each in its own panel, all to one scale."""
    curves = plane(centres)                                                  # [angles, pieces + 1, 2]
    low, high = curves.reshape(-1, 2).min(axis=0), curves.reshape(-1, 2).max(axis=0)
    rows = -(-len(curves) // COLUMNS)
    figure, axes = plt.subplots(rows, COLUMNS, figsize=(2.6 * COLUMNS, 2.0 * rows), dpi=110)
    for ax, curve in zip(axes.reshape(-1), curves):
        draw_rod(ax, curve, low, high)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.05, hspace=0.05)
    return figure


def animate_sweep(centres: core.Point) -> list[np.ndarray]:
    """Euler's elastica as the end angle grows `[frames, pieces + 1]`, all to one scale."""
    curves = plane(centres)                                                  # [frames, pieces + 1, 2]
    low, high = curves.reshape(-1, 2).min(axis=0), curves.reshape(-1, 2).max(axis=0)
    images = []
    for curve in curves:
        figure, ax = plt.subplots(figsize=(5.0, 5.0 * (high[1] - low[1] + 2 * MARGIN) / (high[0] - low[0] + 2 * MARGIN)),
                                  dpi=100)
        draw_rod(ax, curve, low, high)
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        images.append(capture(figure))
        plt.close(figure)
    return images


def draw_helices(centres: core.Point, edges: core.Point) -> plt.Figure:
    """The wires coiled by their couples `[tilts, pieces + 1]`, side by side in one view of space, each with a
    ribbon across its cross-sections `[tilts, pieces + 1, 2]` showing how they twist."""
    curves, ribbons = coordinates(centres), coordinates(edges)               # [tilts, pieces + 1, 3], [..., 2, 3]
    # The wires set side by side across, each a little more than the widest wire apart.
    spacing = SPACING * np.ptp(ribbons[..., :2], axis=(1, 2)).max()
    across = spacing * np.arange(len(curves))[:, None, None] * [1, 0, 0]       # [tilts, 1, 3]
    curves, ribbons = curves + across, ribbons + across[:, :, None]
    figure = plt.figure(figsize=(6, 6.5), dpi=110)
    ax = figure.add_subplot(projection="3d")
    for curve, ribbon in zip(curves, ribbons):
        ax.plot_surface(*np.moveaxis(np.moveaxis(ribbon, 1, 0), -1, 0), color=RIBBON, edgecolor="none", shade=True)
        ax.plot(*curve.T, color=ROD, lw=1.0)
    low, high = ribbons.reshape(-1, 3).min(axis=0), ribbons.reshape(-1, 3).max(axis=0)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(high - low, zoom=1.05)
    ax.view_init(*VIEW)
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def animate_uncoiling(centres: core.Point, edges: core.Point) -> list[np.ndarray]:
    """The wire through the sweep of its tilt `[frames, pieces + 1]`, with its ribbon `[frames, pieces + 1, 2]`,
    the camera turning slowly around it, all to one scale."""
    curves, ribbons = coordinates(centres), coordinates(edges)               # [frames, pieces + 1, 3], [..., 2, 3]
    low, high = ribbons.reshape(-1, 3).min(axis=0), ribbons.reshape(-1, 3).max(axis=0)
    # A square footprint about the coil's axis, so that the turning camera sees the same extent.
    middle, reach = (low + high) / 2, (high - low)[:2].max() / 2
    azimuths = np.linspace(*UNCOIL_AZIMUTHS, len(curves), endpoint=False)
    images = []
    for curve, ribbon, azimuth in zip(curves, ribbons, azimuths):
        figure = plt.figure(figsize=(4.6, 6.4), dpi=100)
        ax = figure.add_subplot(projection="3d")
        ax.plot_surface(*np.moveaxis(np.moveaxis(ribbon, 1, 0), -1, 0), color=RIBBON, edgecolor="none", shade=True)
        ax.plot(*curve.T, color=ROD, lw=1.0)
        ax.set(xlim=(middle[0] - reach, middle[0] + reach), ylim=(middle[1] - reach, middle[1] + reach),
               zlim=(low[2], high[2]))
        ax.set_box_aspect((2 * reach, 2 * reach, high[2] - low[2]), zoom=1.1)
        ax.view_init(UNCOIL_ELEVATION, azimuth)
        ax.set_axis_off()
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        images.append(capture(figure))
        plt.close(figure)
    return images


def animate_cord(centres: core.Point, edges: core.Point) -> list[np.ndarray]:
    """The telephone cord at each frame `[frames, drawn]`, as a ribbon `[frames, drawn, 2]` with its base and its
    handset, all from one fixed view and to one scale."""
    curves, ribbons = coordinates(centres), coordinates(edges)               # [frames, drawn, 3], [..., 2, 3]
    low, high = ribbons.reshape(-1, 3).min(axis=0), ribbons.reshape(-1, 3).max(axis=0)
    images = []
    for curve, ribbon in zip(curves, ribbons):
        figure = plt.figure(figsize=(9, 3.4), dpi=100)
        ax = figure.add_subplot(projection="3d")
        drawn = [ax.plot_surface(*np.moveaxis(np.moveaxis(ribbon, 1, 0), -1, 0), color=RIBBON, edgecolor="none",
                                 shade=True),
                 *ax.plot(*curve.T, color=ROD, lw=0.8),
                 ax.scatter(*curve[[0, -1]].T, color=CLAMP, s=60, depthshade=False)]
        # The view is a square the height of the figure; the long cord reaches past it, across the figure.
        for artist in drawn:
            artist.set_clip_on(False)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
        ax.set_box_aspect(high - low, zoom=2.3)
        ax.view_init(*CORD_VIEW)
        ax.set_axis_off()
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        images.append(capture(figure))
        plt.close(figure)
    return images
