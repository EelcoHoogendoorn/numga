"""Drawing for Kirchhoff's equations: the added mass of spheroids by aspect, and bodies gliding through
the fluid, seen from above, each case in its own lane with the line of its impulse."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.util.animation import capture
from examples.fluids.kirchhoff import core

mv = core.mv
# The colours of the three coefficients: shifting along the axis of symmetry, shifting across it, and
# turning it. The colours of each case's body, its outline and its path, and of the impulse.
COEFFICIENTS = ["#0369a1", "#f97316", "#16a34a"]
BODIES = ["#cbd5e1", "#7dd3fc", "#fdba74"]
EDGES = ["#64748b", "#0369a1", "#c2410c"]
PATHS = ["#94a3b8", "#0284c7", "#ea580c"]
IMPULSE = "#475569"
# The free space between the lanes, and around the bodies at the ends of the view.
GAP = 1.0


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x and y of points, on the last axis, the view from above: each point paired with the planes across
    x and y, over its pairing with the plane at infinity."""
    return ((stack((mv.x, mv.y), axis=-1) & points[..., None]) / (mv.w & points[..., None])).to_array()


def draw_coefficients(aspects: np.ndarray, ratios: core.Scalar) -> plt.Figure:
    """The fluid's kinetic energy over the solid's for spheroids of each aspect `[aspects, 3]`, against
    the aspect: discs on the left, rods on the right."""
    values = ratios.to_array()                                                # [aspects, 3]
    figure, ax = plt.subplots(figsize=(6.0, 3.6), dpi=120)
    for column, colour in enumerate(COEFFICIENTS):
        ax.plot(aspects, values[:, column], color=colour, linewidth=1.8)
    ax.axvline(1, color="#cbd5e1", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    # The ratios from their zero, read off at the ticks; the aspect without numbers: discs on the left, rods on the
    # right, the sphere at the grey line.
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlabel("axis of symmetry over the other semi-axes")
    ax.set_ylabel("kinetic energy, fluid over solid")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def impulse_line(impulse: core.Forque, extent: tuple[float, float]) -> np.ndarray:
    """The ends of the impulse's line within the given extent along the first axis `[2, 2]`: where it
    meets the planes square to that axis there."""
    ends = impulse ^ (mv.x - np.array(extent) * mv.w)                         # [2] Point
    return coordinates(ends)


def animate_gliding(impulse: core.Forque,
                    frames: Iterable[tuple[core.Point, core.Point, core.Point]]) -> list[np.ndarray]:
    """Each frame of every case's outline `[cases, outline + 1]`, centre `[cases]` and nose `[cases]`: the
    bodies seen from above, each in its own lane, with its path so far and its impulse."""
    frames = [tuple(coordinates(points) for points in frame) for frame in frames]
    shapes = np.stack([outlines for outlines, _, _ in frames])               # [frames, cases, outline + 1, 2]
    paths = np.stack([centres for _, centres, _ in frames])                  # [frames, cases, 2]
    count = paths.shape[1]
    # Each lane as wide as the bodies ever reach across it, and the view as long as they reach along it.
    lane = 2 * np.abs(shapes[..., 1]).max() + GAP
    lanes = lane * (count - 1 - np.arange(count))[:, None] * np.array([0.0, 1.0])   # [cases, 2]
    extent = (shapes[..., 0].min() - GAP / 2, shapes[..., 0].max() + GAP / 2)
    line = impulse_line(impulse, extent)                                     # [2, 2]
    height = lane * count
    images = []
    for index, (outlines, _, noses) in enumerate(frames):
        figure = plt.figure(figsize=(12.0, 12.0 * height / (extent[1] - extent[0])), dpi=100)
        ax = figure.add_axes((0, 0, 1, 1))
        for case in range(count):
            ax.plot(*(line + lanes[case]).T, color=IMPULSE, linewidth=0.8, linestyle=(0, (6, 4)), zorder=0)
            ax.plot(*(paths[:index + 1, case] + lanes[case]).T, color=PATHS[case], linewidth=1.4, zorder=1)
            ax.fill(*(outlines[case] + lanes[case]).T, facecolor=BODIES[case], edgecolor=EDGES[case],
                    linewidth=1.4, zorder=2)
            ax.plot(*(noses[case] + lanes[case]), "o", color=EDGES[case], markersize=5, zorder=3)
        ax.set_xlim(*extent)
        ax.set_ylim(-lane / 2, height - lane / 2)
        ax.set_aspect("equal")
        ax.set_axis_off()
        images.append(capture(figure))
        plt.close(figure)
    return images
