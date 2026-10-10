"""Orbital paths, their spinor lifts, and integration errors."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from typing import TYPE_CHECKING

from examples.animation import capture
from examples.mechanics.kepler import core

if TYPE_CHECKING:
    from IPython.display import Image as Shown

EXACT_COLOUR = "#626c76"
PHYSICAL_COLOUR = "#d06c36"
REGULARIZED_COLOUR = "#287fa0"
BODY_COLOUR = "#ff00b7"


# --- plumbing -------------------------------------------------------------------------
def draw_lift(positions: core.Vector, spinors: core.Spinor) -> plt.Figure:
    """A planar orbital passage and the half-turn of its position spinor."""
    points = positions.cast(core.ga.subspace("x y")).kernel
    lifted = spinors.cast(core.ga.subspace("1 xy")).kernel
    figure, panels = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, path, title in zip(panels, (points, lifted), ("Kepler orbit", "Position spinor")):
        ax.plot(*path.T, color=REGULARIZED_COLOUR, linewidth=1.5)
        ax.scatter(*path.T, c=np.linspace(0, 1, len(path)), cmap="viridis", s=15, zorder=3)
        ax.scatter(0, 0, s=65, color="#e5ab32", edgecolors="white", zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    panels[0].set(xlabel="x", ylabel="y")
    panels[1].set(xlabel="scalar", ylabel="xy")
    return figure


def draw_comparison(reference: core.State, physical: core.State,
                    regularized: core.State, labels: tuple[str, ...]) -> plt.Figure:
    """One orbit at equal step counts, with a periapsis detail below each full path."""
    exact = reference.position.cast(core.ga.subspace("x y")).kernel
    direct = physical.position.cast(core.ga.subspace("x y")).kernel
    lifted = regularized.position.cast(core.ga.subspace("x y")).kernel
    figure, panels = plt.subplots(2, len(labels), figsize=(10.5, 6), squeeze=False,
                                  layout="constrained")
    for case, label in enumerate(labels):
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
                ax.set(xlim=(centre[0] - 0.55 * span, centre[0] + 0.55 * span),
                       ylim=(centre[1] - 0.55 * span, centre[1] + 0.55 * span), title=label)
            else:
                reach = 0.12 * span
                ax.set(xlim=(low[0] - reach, low[0] + reach), ylim=(-reach, reach))
            ax.set_aspect("equal")
            ax.spines[["top", "right"]].set_visible(False)
        panels[1, case].set_xlabel("x")
    panels[0, 0].set_ylabel("y")
    panels[1, 0].set_ylabel("Periapsis detail")
    panels[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    return figure


def draw_errors(counts: np.ndarray, physical: core.Scalar, regularized: core.Scalar,
                labels: tuple[str, ...]) -> plt.Figure:
    """Physical-time-weighted RMS position errors against the exact orbit."""
    direct = physical.cast(core.ga.subspace.scalar()).kernel[..., 0]
    lifted = regularized.cast(core.ga.subspace.scalar()).kernel[..., 0]
    figure, panels = plt.subplots(1, len(labels), figsize=(10.5, 3.1), squeeze=False,
                                  layout="constrained")
    for case, (ax, label) in enumerate(zip(panels[0], labels)):
        ax.loglog(counts, direct[:, case], "o-", color=PHYSICAL_COLOUR, label="Physical clock")
        ax.loglog(counts, lifted[:, case], "o-", color=REGULARIZED_COLOUR, label="Spinor clock")
        ax.set(xlabel="Steps per orbit", title=label)
        ax.set_xticks(counts, labels=[str(count) for count in counts])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
    panels[0, 0].set_ylabel("RMS position error / semimajor axis")
    panels[0, 0].legend(frameon=False, fontsize=8)
    return figure


def animate_lift(positions: core.Vector, spinors: core.Spinor) -> list[np.ndarray]:
    """Two orbital revolutions and one closed spinor loop, sampled in the regularized clock."""
    points = positions.cast(core.ga.subspace("x y")).kernel
    lifted = spinors.cast(core.ga.subspace("1 xy")).kernel
    figure, panels = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    markers = []
    for ax, path, title in zip(panels, (points, lifted),
                               ("Orbit · regularized clock", "Position spinor")):
        ax.plot(*path.T, color=REGULARIZED_COLOUR, linewidth=1.5)
        ax.scatter(0, 0, s=55, color="#e5ab32", edgecolors="white", zorder=4)
        markers.append(ax.scatter([], [], s=80, color=BODY_COLOUR, edgecolors="white", zorder=5))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    frames = []
    for point, spinor in zip(points, lifted):
        markers[0].set_offsets(point[None])
        markers[1].set_offsets(spinor[None])
        frames.append(capture(figure))
    plt.close(figure)
    return frames


def inline(frames: list[np.ndarray], duration_ms: int) -> Shown:
    """Encode animation frames for notebook display."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")
