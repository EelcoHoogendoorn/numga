"""Spinor observables, chiral currents and charge conjugation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from examples.animation import capture
from examples.relativity.spinors import core

RIGHT_COLOUR, LEFT_COLOUR, INK = "#267fa9", "#d17c24", "#263b51"
# The interference ports and the conjugation eigenspaces share the chirality palette.
PORT_COLOUR, FIXED_COLOUR, NEGATED_COLOUR = RIGHT_COLOUR, RIGHT_COLOUR, LEFT_COLOUR


# --- plumbing -----------------------------------------------------------------------
def plain_axes(ax: plt.Axes) -> None:
    """Keep the curves prominent and their axes light."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.75")
    ax.tick_params(color="0.75")


def animate_rotation(
    spin: core.Vector, bright: core.Scalar, dark: core.Scalar,
) -> list[np.ndarray]:
    """The spin direction beside two interference outputs against a fixed reference."""
    directions = spin.cast(core.ga.subspace("x z")).kernel
    figure, (axis, ports) = plt.subplots(1, 2, figsize=(6.8, 3.2), dpi=85, layout="constrained")
    axis.add_patch(Circle((0, 0), 1, edgecolor="0.8", facecolor="none", linewidth=0.8))
    arrow = axis.quiver(0, 0, *directions[0], color=INK, angles="xy", scale_units="xy", scale=1, width=0.02)
    axis.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal", xlabel="x", ylabel="z", xticks=[], yticks=[])
    disks = []
    for centre, label in zip(((-0.7, 0), (0.7, 0)), ("sum port", "difference port")):
        ports.add_patch(Circle(centre, 0.47, facecolor="0.94", edgecolor="0.7", linewidth=0.8))
        disk = Circle(centre, 0.45, facecolor=PORT_COLOUR, edgecolor="none")
        ports.add_patch(disk)
        disks.append(disk)
        ports.text(centre[0], -0.7, label, ha="center", fontsize=9)
    ports.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
    ports.set_axis_off()
    plain_axes(axis)
    frames = []
    for direction, first, second in zip(directions, bright.to_array(), dark.to_array()):
        arrow.set_UVC(*direction)
        for disk, intensity in zip(disks, (first, second)):
            disk.set_alpha(float(np.clip(intensity, 0, 1)))
        frames.append(capture(figure))
    plt.close(figure)
    return frames


def draw_chirality(
    left: core.Vector, right: core.Vector, current: core.Vector,
) -> plt.Figure:
    """Two null currents and their timelike sum in the z–t plane."""
    tips = np.stack([
        vector.cast(core.ga.subspace("z t")).kernel
        for vector in (left, right, current)
    ])
    extent = tips[:, 1].max() * 1.15
    figure, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    ax.fill([-extent, 0, extent], [extent, 0, extent], color="0.97")
    ax.plot([-extent, 0, extent], [extent, 0, extent], color="0.8", linewidth=0.8)
    for tip, name, colour in zip(tips, ("left", "right", "Dirac"), (LEFT_COLOUR, RIGHT_COLOUR, INK)):
        ax.annotate("", xy=tip, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2})
        ax.annotate(name, xy=tip, xytext=(7, 3), textcoords="offset points", color=colour, fontsize=10)
    ax.set(xlim=(-extent, extent), ylim=(0, extent), xlabel="z current", ylabel="time current", aspect="equal")
    plain_axes(ax)
    return figure


def draw_majorana(
    phases: np.ndarray, even_density: core.Scalar, odd_density: core.Scalar,
) -> plt.Figure:
    """A phase turn trades density between the two charge-conjugation eigenspaces."""
    figure, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    ax.plot(phases / np.pi, even_density.to_array(), color=FIXED_COLOUR, label="conjugation +1")
    ax.plot(phases / np.pi, odd_density.to_array(), color=NEGATED_COLOUR, label="conjugation −1")
    ax.set(xlim=(phases[0] / np.pi, phases[-1] / np.pi), ylim=(0, 1.05), xlabel="phase angle / π", ylabel="density")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=9)
    plain_axes(ax)
    return figure

