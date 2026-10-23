"""Spinor observables, chiral currents and charge conjugation."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from numga import stack
from examples.util.animation import capture
from examples.relativity.spinors import core

RIGHT_COLOUR, LEFT_COLOUR, INK = "#267fa9", "#d17c24", "#263b51"
# The interference ports and the conjugation eigenspaces share the chirality palette.
PORT_COLOUR, FIXED_COLOUR, NEGATED_COLOUR = RIGHT_COLOUR, RIGHT_COLOUR, LEFT_COLOUR
# The faint cross through the origin of a plot, and the colour of its lines' names.
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.45"


# --- plumbing -----------------------------------------------------------------------
def cross(ax: plt.Axes, across: str, up: str) -> None:
    """No scales and no frame: a faint cross through the origin, its lines named at their ends."""
    # Over filled regions, under what is drawn in them.
    ax.axhline(0, color=CROSS_COLOUR, linewidth=0.8, zorder=1.5)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=0.8, zorder=1.5)
    ax.annotate(across, (1, 0), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", color=LABEL_COLOUR)
    ax.annotate(up, (0, 1), xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=LABEL_COLOUR)
    ax.set_axis_off()


def plain_axes(ax: plt.Axes, across: str, up: str) -> None:
    """Axes named by what runs along them, the quantity's zero as the horizontal one, light and without numbers."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.75")
    ax.spines["bottom"].set_position(("data", 0))
    ax.tick_params(color="0.75")
    ax.set(xticks=[], yticks=[], xlabel=across, ylabel=up)


def _read(directions: core.Antivector, *slices: str) -> np.ndarray:
    """Directions read by pairing with the named slices, on the last axis."""
    mv = directions.context.multivector
    return (stack(tuple(getattr(mv, name) for name in slices)) & directions[..., None]).to_array()


def animate_rotation(
    spin: core.Antivector, bright: core.Scalar, dark: core.Scalar,
) -> Iterator[np.ndarray]:
    """The spin direction beside two interference outputs against a fixed reference."""
    directions = _read(spin, "x", "z")
    figure, (axis, ports) = plt.subplots(1, 2, figsize=(6.8, 3.2), dpi=85, layout="constrained")
    axis.add_patch(Circle((0, 0), 1, edgecolor="0.8", facecolor="none", linewidth=0.8))
    arrow = axis.quiver(0, 0, *directions[0], color=INK, angles="xy", scale_units="xy", scale=1, width=0.02)
    axis.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
    disks = []
    for centre, label in zip(((-0.7, 0), (0.7, 0)), ("sum port", "difference port")):
        ports.add_patch(Circle(centre, 0.47, facecolor="0.94", edgecolor="0.7", linewidth=0.8))
        disk = Circle(centre, 0.45, facecolor=PORT_COLOUR, edgecolor="none")
        ports.add_patch(disk)
        disks.append(disk)
        ports.text(centre[0], -0.7, label, ha="center", fontsize=9)
    ports.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
    ports.set_axis_off()
    cross(axis, "x", "z")
    for direction, first, second in zip(directions, bright.to_array(), dark.to_array()):
        arrow.set_UVC(*direction)
        for disk, intensity in zip(disks, (first, second)):
            disk.set_alpha(float(np.clip(intensity, 0, 1)))
        yield capture(figure)
    plt.close(figure)


def draw_chirality(
    left: core.Antivector, right: core.Antivector, current: core.Antivector,
) -> plt.Figure:
    """Two null currents and their timelike sum in the z–t plane."""
    tips = np.stack([_read(direction, "z", "t") for direction in (left, right, current)])
    extent = tips[:, 1].max() * 1.15
    figure, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    ax.fill([-extent, 0, extent], [extent, 0, extent], color="0.97")
    ax.plot([-extent, 0, extent], [extent, 0, extent], color="0.8", linewidth=0.8)
    for tip, name, colour in zip(tips, ("left", "right", "Dirac"), (LEFT_COLOUR, RIGHT_COLOUR, INK)):
        ax.annotate("", xy=tip, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2})
        ax.annotate(name, xy=tip, xytext=(7, 3), textcoords="offset points", color=colour, fontsize=10)
    ax.set(xlim=(-extent, extent), ylim=(0, extent), aspect="equal")
    cross(ax, "z", "t")
    return figure


def draw_majorana(
    phases: np.ndarray, even_density: core.Scalar, odd_density: core.Scalar,
) -> plt.Figure:
    """A phase turn trades density between the two charge-conjugation eigenspaces."""
    figure, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    ax.plot(phases / np.pi, even_density.to_array(), color=FIXED_COLOUR, label="conjugation +1")
    ax.plot(phases / np.pi, odd_density.to_array(), color=NEGATED_COLOUR, label="conjugation −1")
    ax.set(xlim=(phases[0] / np.pi, phases[-1] / np.pi), ylim=(0, 1.05))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=9)
    plain_axes(ax, "phase angle", "density")
    # The phases that keep the Majorana condition, the only values marked.
    ax.set(xticks=[0, 1, 2], xticklabels=["0", "π", "2π"])
    return figure

