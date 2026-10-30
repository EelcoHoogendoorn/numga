"""Drawing for Kirchhoff's equations: the added mass of spheroids by aspect, and bodies gliding through
the fluid, seen from above, each case in its own lane with the line of its impulse."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.fluids.kirchhoff import core

mv = core.mv
# The colours of the three coefficients: shifting along the axis of symmetry, shifting across it, and
# turning it. The colours of each case's body, its outline and its path, and of the impulse.
coefficient_colours = ["#0369a1", "#f97316", "#16a34a"]
body_colours = ["#cbd5e1", "#7dd3fc", "#fdba74"]
edge_colours = ["#64748b", "#0369a1", "#c2410c"]
path_colours = ["#94a3b8", "#0284c7", "#ea580c"]
impulse_colour = "#475569"
# The free space between the lanes, and around the bodies at the ends of the view.
gap = 1.0


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x and y of points, on the last axis, the view from above: each point, at unit weight, paired with the planes
    across x and y."""
    return (stack((mv.x, mv.y), axis=-1) & points.normalized()[..., None]).to_array()


@charts.style()
def draw_coefficients(aspects: np.ndarray, ratios: core.Scalar) -> plt.Figure:
    """The fluid's kinetic energy over the solid's for spheroids of each aspect `[aspects, 3]`, against
    the aspect: discs on the left, rods on the right."""
    values = ratios.to_array()                                                # [aspects, 3]
    # Half the column: three smooth curves, and room for the two long axis names.
    figure, ax = charts.subplots(420, 280)
    for column, colour in enumerate(coefficient_colours):
        ax.plot(aspects, values[:, column], color=colour)
    ax.axvline(1, color="#cbd5e1", linewidth=charts.FRAME_WIDTH, zorder=0)
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


def animate_gliding(impulse: core.Forque,
                    frames: Iterable[tuple[core.Point, core.Point, core.Point]]) -> Iterator[np.ndarray]:
    """Each frame of every case's outline `[cases, outline + 1]`, centre `[cases]` and nose `[cases]`: the
    bodies seen from above, each in its own lane, with its path so far and its impulse."""
    frames = [tuple(points.normalized() for points in frame) for frame in frames]
    outlines = stack([outline for outline, _, _ in frames])                  # [frames, cases, outline + 1] Point-
    centres = stack([middle for _, middle, _ in frames])                     # [frames, cases] Point-
    noses = stack([nose for _, _, nose in frames])                           # [frames, cases] Point-
    count = centres.shape[1]
    # Each lane as wide as the bodies ever reach across it, and the view as long as they reach along it.
    widths = (mv.y & outlines).abs().reshape(-1)
    lane = 2 * widths.max() + gap
    reach = (mv.x & outlines).reshape(-1)
    extent = stack((reach.min() - gap / 2, reach.max() + gap / 2))
    # Each case moved across into its own lane, the first furthest.
    lanes = core.directions[1] * (lane * (count - 1 - np.arange(count)))     # [cases] Direction-
    # The ends of the impulse's line within the extent: where it meets the planes square to x there.
    ends = (impulse ^ (mv.x - extent * mv.w)).normalized()                    # [2] Point-
    line = coordinates(ends + lanes[:, None])                                # [cases, 2, 2]
    shapes = coordinates(outlines + lanes[:, None])                          # [frames, cases, outline + 1, 2]
    paths = coordinates(centres + lanes)                                     # [frames, cases, 2]
    tips = coordinates(noses + lanes)                                        # [frames, cases, 2]
    height = lane * count
    # The lanes and their impulses set up once; each body, its path so far and its nose move.
    aspect = float((height / (extent[1] - extent[0])).to_array())
    figure = plt.figure(figsize=(4.0, 4.0 * aspect), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    trails, bodies, dots = [], [], []
    for case in range(count):
        ax.plot(*line[case].T, color=impulse_colour, linewidth=0.8, linestyle=(0, (6, 4)), zorder=0)
        trails += ax.plot([], [], color=path_colours[case], linewidth=1.4, zorder=1)
        bodies += ax.fill(*shapes[0, case].T, facecolor=body_colours[case], edgecolor=edge_colours[case],
                          linewidth=1.4, zorder=2)
        dots += ax.plot([], [], "o", color=edge_colours[case], markersize=5, zorder=3)
    ax.set_xlim(*extent.to_array())
    ax.set_ylim(*stack((-lane / 2, height - lane / 2)).to_array())
    ax.set_aspect("equal")
    ax.set_axis_off()
    for index in range(len(frames)):
        for case in range(count):
            trails[case].set_data(*paths[:index + 1, case].T)
            bodies[case].set_xy(shapes[index, case])
            dots[case].set_data(*tips[index, case, :, None])
        yield capture(figure)
    plt.close(figure)
