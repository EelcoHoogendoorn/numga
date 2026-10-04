"""The two Fresnel sheets and their meridian sections."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from examples.animation import capture
from examples.electromagnetism.fresnel import core

COLOURS = ("#2563eb", "#ea580c")
LIMIT = 2.05
LATITUDE_STRIDE = 4
LONGITUDE_STRIDE = 8


# --- plumbing -------------------------------------------------------------------------
def surface(ax: Axes, sheets: core.Vector) -> None:
    points = sheets.cast(core.ga.subspace("x y z")).kernel
    for mode, colour in enumerate(COLOURS):
        sheet = points[..., mode, :]
        ax.plot_surface(sheet[..., 0], sheet[..., 1], sheet[..., 2], color=colour,
                        alpha=0.16, shade=False, linewidth=0, antialiased=True,
                        rcount=sheet.shape[0], ccount=sheet.shape[1])
        ax.plot_wireframe(sheet[..., 0], sheet[..., 1], sheet[..., 2], color=colour,
                          rstride=LATITUDE_STRIDE, cstride=LONGITUDE_STRIDE,
                          linewidth=0.5, alpha=0.55)
    ax.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT), zlim=(-LIMIT, LIMIT),
           xlabel="$k_x$", ylabel="$k_y$", zlabel="$k_z$")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-53)


def section(ax: Axes, sections: core.Vector) -> None:
    points = sections.cast(core.ga.subspace("x y z")).kernel
    for mode, (colour, style, label) in enumerate(zip(COLOURS, ("-", "--"), ("inner sheet", "outer sheet"))):
        ax.plot(points[:, mode, 0], points[:, mode, 2], color=colour, linestyle=style, linewidth=2, label=label)
    ax.axhline(0, color="#cbd5e1", linewidth=0.7)
    ax.axvline(0, color="#cbd5e1", linewidth=0.7)
    ax.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT), xlabel="$k_x$", ylabel="$k_z$")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)


def comparison(sheets: core.Vector, sections: core.Vector, names: tuple[str, ...]) -> plt.Figure:
    figure = plt.figure(figsize=(12, 7.8), layout="constrained")
    count = len(names)
    for i, name in enumerate(names):
        ax = figure.add_subplot(2, count, i + 1, projection="3d")
        surface(ax, sheets[i])
        ax.set_title(name, fontsize=11)
        ax = figure.add_subplot(2, count, count + i + 1)
        section(ax, sections[i])
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    return figure


def frame(sheets: core.Vector, sections: core.Vector) -> np.ndarray:
    figure = plt.figure(figsize=(9, 4.5), dpi=85, layout="constrained")
    surface(figure.add_subplot(1, 2, 1, projection="3d"), sheets)
    section(figure.add_subplot(1, 2, 2), sections)
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(frames: Iterable[tuple[core.Vector, core.Vector]]) -> list[np.ndarray]:
    return [frame(sheets, sections) for sheets, sections in frames]
