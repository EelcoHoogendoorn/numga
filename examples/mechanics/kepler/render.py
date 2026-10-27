"""Orbital paths, their spinor lifts, and integration errors."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from numga import stack

from examples.util.animation import capture
from examples.mechanics.kepler import core

EXACT_COLOUR = "#626c76"
PHYSICAL_COLOUR = "#d06c36"
REGULARIZED_COLOUR = "#287fa0"
BODY_COLOUR = "#ff00b7"
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.4"
# The margin around a lifted path, as a share of its extent.
LABEL_ROOM = 0.12


# --- plumbing -------------------------------------------------------------------------
def plane_coordinates(points: core.Point) -> np.ndarray:
    """x and y of points in the orbit's plane, on the last axis: their pairings with the planes x = 0 and y = 0."""
    return (stack((core.mv.x, core.mv.y)) & points[..., None]).to_array()


def spinor_coordinates(spinors: core.Spinor) -> np.ndarray:
    """The scalar and xy parts of spinors, on the last axis: their scalar products with one and with `-mv.xy`."""
    return np.stack([spinors.scalar_product(core.mv.scalar([1])).to_array(),
                     spinors.scalar_product(-core.mv.xy).to_array()], axis=-1)


def cross(ax: plt.Axes, across: str, up: str) -> None:
    """A plane without scales: a faint cross through the origin, its lines named at their ends where they are in
    view."""
    ax.axhline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.text(1, 0, across, transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=6, color=LABEL_COLOUR,
            clip_on=True)
    ax.text(0, 1, f" {up}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6, color=LABEL_COLOUR,
            clip_on=True)
    ax.set_axis_off()


def draw_lift(positions: core.Point, spinors: core.Spinor) -> plt.Figure:
    """A planar orbital passage and the half-turn of its position spinor."""
    points = plane_coordinates(positions)
    lifted = spinor_coordinates(spinors)
    figure, panels = plt.subplots(1, 2, figsize=(4.5, 1.9), layout="constrained")
    for ax, path, (across, up) in zip(panels, (points, lifted), (("x", "y"), ("scalar", "xy"))):
        ax.plot(*path.T, color=REGULARIZED_COLOUR, linewidth=1.0)
        ax.scatter(*path.T, c=np.linspace(0, 1, len(path)), cmap="viridis", s=5, zorder=3)
        ax.scatter(0, 0, s=25, color="#e5ab32", edgecolors="white", zorder=4)
        # Room beyond the path for the names of the cross.
        ax.margins(LABEL_ROOM)
        ax.set_aspect("equal", adjustable="datalim")
        cross(ax, across, up)
    return figure


def draw_comparison(reference: core.State, physical: core.State, regularized: core.State) -> plt.Figure:
    """One orbit at equal step counts, with a periapsis detail below each full path."""
    exact = plane_coordinates(reference.position)
    direct = plane_coordinates(physical.position)
    lifted = plane_coordinates(regularized.position)
    cases = exact.shape[1]
    figure, panels = plt.subplots(2, cases, figsize=(5.25, 3), squeeze=False, layout="constrained")
    for case in range(cases):
        for row in range(2):
            ax = panels[row, case]
            ax.plot(*exact[:, case].T, color=EXACT_COLOUR, linewidth=1.6, alpha=0.5, label="Exact")
            ax.plot(*direct[:, case].T, color=PHYSICAL_COLOUR, linewidth=0.8, label="Physical clock")
            ax.plot(*lifted[:, case].T, color=REGULARIZED_COLOUR, linewidth=0.9, label="Spinor clock")
            ax.scatter(0, 0, s=12, color="#e5ab32", edgecolors="white", zorder=4)
            low, high = exact[:, case].min(axis=0), exact[:, case].max(axis=0)
            span = np.max(high - low)
            if row == 0:
                centre = (low + high) / 2
                ax.set(xlim=(centre[0] - 0.6 * span, centre[0] + 0.6 * span),
                       ylim=(centre[1] - 0.6 * span, centre[1] + 0.6 * span))
            else:
                half_width = 0.12 * span
                ax.set(xlim=(low[0] - half_width, low[0] + half_width), ylim=(-half_width, half_width))
            ax.set_aspect("equal")
            cross(ax, "x", "y")
            if row == 1:
                # The periapsis detail framed, so that it reads as a close-up rather than as the path running on.
                ax.set_axis_on()
                ax.set(xticks=[], yticks=[])
                ax.spines[:].set_color(CROSS_COLOUR)
    figure.get_layout_engine().set(hspace=0.08)
    figure.legend(*panels[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False,
                  fontsize=6)
    return figure


def draw_errors(counts: np.ndarray, physical: core.Scalar, regularized: core.Scalar) -> plt.Figure:
    """Physical-time-weighted RMS position errors against the exact orbit."""
    direct = physical.to_array()
    lifted = regularized.to_array()
    figure, panels = plt.subplots(1, direct.shape[1], figsize=(5.25, 1.6), squeeze=False,
                                  layout="constrained")
    for case, ax in enumerate(panels[0]):
        ax.loglog(counts, direct[:, case], "o-", color=PHYSICAL_COLOUR, label="Physical clock", markersize=3, linewidth=1)
        ax.loglog(counts, lifted[:, case], "o-", color=REGULARIZED_COLOUR, label="Spinor clock", markersize=3, linewidth=1)
        ax.set_xlabel("Steps per orbit", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xticks(counts, labels=[str(count) for count in counts])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
    panels[0, 0].set_ylabel("RMS error / semimajor axis", fontsize=6)
    panels[0, 0].legend(frameon=False, fontsize=6)
    return figure


def animate_lift(positions: core.Point, spinors: core.Spinor) -> Iterator[np.ndarray]:
    """Two orbital revolutions and one closed spinor loop, sampled in the regularized clock."""
    points = plane_coordinates(positions)
    lifted = spinor_coordinates(spinors)
    figure, panels = plt.subplots(1, 2, figsize=(4, 1.7), layout="constrained")
    markers = []
    for ax, path in zip(panels, (points, lifted)):
        ax.plot(*path.T, color=REGULARIZED_COLOUR, linewidth=1.0)
        ax.scatter(0, 0, s=22, color="#e5ab32", edgecolors="white", zorder=4)
        markers.append(ax.scatter([], [], s=30, color=BODY_COLOUR, edgecolors="white", zorder=5))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    # Lay the figure out once; only the markers move between frames.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    for point, spinor in zip(points, lifted):
        markers[0].set_offsets(point[None])
        markers[1].set_offsets(spinor[None])
        yield capture(figure)
    plt.close(figure)

