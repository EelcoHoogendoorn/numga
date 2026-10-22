"""Orbital paths, their spinor lifts, and integration errors."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from examples.animation import capture
from examples.mechanics.kepler import core

EXACT_COLOUR = "#626c76"
PHYSICAL_COLOUR = "#d06c36"
REGULARIZED_COLOUR = "#287fa0"
BODY_COLOUR = "#ff00b7"
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.4"
# The margin around a lifted path, as a share of its extent.
LABEL_ROOM = 0.12


# --- plumbing -------------------------------------------------------------------------
def cross(ax: plt.Axes, across: str, up: str) -> None:
    """A plane without scales: a faint cross through the origin, its lines named at their ends where they are in
    view."""
    ax.axhline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.text(1, 0, across, transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=9, color=LABEL_COLOUR,
            clip_on=True)
    ax.text(0, 1, f" {up}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=9, color=LABEL_COLOUR,
            clip_on=True)
    ax.set_axis_off()


def draw_lift(positions: core.Point, spinors: core.Spinor) -> plt.Figure:
    """A planar orbital passage and the half-turn of its position spinor."""
    points = positions.cast(core.ga.subspace("yz zx")).kernel
    lifted = spinors.cast(core.ga.subspace("1 xy")).kernel
    figure, panels = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, path, (across, up) in zip(panels, (points, lifted), (("x", "y"), ("scalar", "xy"))):
        ax.plot(*path.T, color=REGULARIZED_COLOUR, linewidth=1.5)
        ax.scatter(*path.T, c=np.linspace(0, 1, len(path)), cmap="viridis", s=15, zorder=3)
        ax.scatter(0, 0, s=65, color="#e5ab32", edgecolors="white", zorder=4)
        # Room beyond the path for the names of the cross.
        ax.margins(LABEL_ROOM)
        ax.set_aspect("equal", adjustable="datalim")
        cross(ax, across, up)
    return figure


def draw_comparison(reference: core.State, physical: core.State, regularized: core.State) -> plt.Figure:
    """One orbit at equal step counts, with a periapsis detail below each full path."""
    exact = reference.position.cast(core.ga.subspace("yz zx")).kernel
    direct = physical.position.cast(core.ga.subspace("yz zx")).kernel
    lifted = regularized.position.cast(core.ga.subspace("yz zx")).kernel
    cases = exact.shape[1]
    figure, panels = plt.subplots(2, cases, figsize=(10.5, 6), squeeze=False, layout="constrained")
    for case in range(cases):
        for row in range(2):
            ax = panels[row, case]
            ax.plot(*exact[:, case].T, color=EXACT_COLOUR, linewidth=2.5, alpha=0.5, label="Exact")
            ax.plot(*direct[:, case].T, color=PHYSICAL_COLOUR, linewidth=1.2, label="Physical clock")
            ax.plot(*lifted[:, case].T, color=REGULARIZED_COLOUR, linewidth=1.3, label="Spinor clock")
            ax.scatter(0, 0, s=30, color="#e5ab32", edgecolors="white", zorder=4)
            low, high = exact[:, case].min(axis=0), exact[:, case].max(axis=0)
            span = np.max(high - low)
            if row == 0:
                centre = (low + high) / 2
                ax.set(xlim=(centre[0] - 0.6 * span, centre[0] + 0.6 * span),
                       ylim=(centre[1] - 0.6 * span, centre[1] + 0.6 * span))
            else:
                reach = 0.12 * span
                ax.set(xlim=(low[0] - reach, low[0] + reach), ylim=(-reach, reach))
            ax.set_aspect("equal")
            cross(ax, "x", "y")
            if row == 1:
                # The periapsis detail framed, so that it reads as a close-up rather than as the path running on.
                ax.set_axis_on()
                ax.set(xticks=[], yticks=[])
                ax.spines[:].set_color(CROSS_COLOUR)
    figure.get_layout_engine().set(hspace=0.08)
    figure.legend(*panels[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False,
                  fontsize=8)
    return figure


def draw_errors(counts: np.ndarray, physical: core.Scalar, regularized: core.Scalar) -> plt.Figure:
    """Physical-time-weighted RMS position errors against the exact orbit."""
    direct = physical.cast(core.ga.subspace.scalar()).kernel[..., 0]
    lifted = regularized.cast(core.ga.subspace.scalar()).kernel[..., 0]
    figure, panels = plt.subplots(1, direct.shape[1], figsize=(10.5, 3.1), squeeze=False,
                                  layout="constrained")
    for case, ax in enumerate(panels[0]):
        ax.loglog(counts, direct[:, case], "o-", color=PHYSICAL_COLOUR, label="Physical clock")
        ax.loglog(counts, lifted[:, case], "o-", color=REGULARIZED_COLOUR, label="Spinor clock")
        ax.set(xlabel="Steps per orbit")
        ax.set_xticks(counts, labels=[str(count) for count in counts])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
    panels[0, 0].set_ylabel("RMS position error / semimajor axis")
    panels[0, 0].legend(frameon=False, fontsize=8)
    return figure


def animate_lift(positions: core.Point, spinors: core.Spinor) -> list[np.ndarray]:
    """Two orbital revolutions and one closed spinor loop, sampled in the regularized clock."""
    points = positions.cast(core.ga.subspace("yz zx")).kernel
    lifted = spinors.cast(core.ga.subspace("1 xy")).kernel
    figure, panels = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    markers = []
    for ax, path in zip(panels, (points, lifted)):
        ax.plot(*path.T, color=REGULARIZED_COLOUR, linewidth=1.5)
        ax.scatter(0, 0, s=55, color="#e5ab32", edgecolors="white", zorder=4)
        markers.append(ax.scatter([], [], s=80, color=BODY_COLOUR, edgecolors="white", zorder=5))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    # Lay the figure out once; only the markers move between frames.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    frames = []
    for point, spinor in zip(points, lifted):
        markers[0].set_offsets(point[None])
        markers[1].set_offsets(spinor[None])
        frames.append(capture(figure))
    plt.close(figure)
    return frames

