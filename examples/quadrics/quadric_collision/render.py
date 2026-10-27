"""The second ellipse sliding along the normal through contact with the first, beside the determinant of their blend."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.util.animation import capture
from examples.quadrics.quadric_collision.core import (
    Line, Point, Polarity, Quadric, Scalar, infinity, mv, origin, tangent_line,
)

# The height of the view over its width, and the margin around what the view frames, over its extent.
ASPECT = 0.75
MARGIN = 1.15
# The grid the ellipses and the blend at the peak are evaluated on, per side.
RESOLUTION = 250
FIRST_COLOURS, SECOND_COLOURS = ("#3b82f6", "#1d4ed8"), ("#f97316", "#ea580c")
# While the pair is apart and once it overlaps: the midline, determinant curve and peak; the blend's edge; its fill
# and the fill's opacity.
APART_COLOURS = ("#16a34a", "#059669", "#10b981", 0.10)
OVERLAP_COLOURS = ("#9333ea", "#7c3aed", "#a855f7", 0.22)


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 2]`: their pairings with the lines x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (infinity & points)[..., None]).to_array()


def bounds(dual: Quadric) -> np.ndarray:
    """The low and high corners of the box around an ellipse: the contact points of its tangents facing along and
    against x and y."""
    contacts = xy(dual(tangent_line(dual, stack((mv.x, mv.y, -mv.x, -mv.y)))))
    return np.stack([contacts.min(axis=0), contacts.max(axis=0)])


def draw_ellipse(ax, dual: Quadric, grid: Point, colours: tuple[str, str]) -> list:
    """An ellipse filled and outlined where its form over the grid changes sign: each point paired with its polar
    line, under the inverse of the dual, signed by the centre's so that it is positive inside; the artists, for an
    animation to remove."""
    primal = dual.inverse()                                                    # [] Line+ <- Point-
    centre = dual(infinity)                                                    # [] Point-
    field = ((grid & primal(grid)) * (centre & primal(centre))).to_array()
    coordinates = xy(grid)
    xs, ys = coordinates[..., 0], coordinates[..., 1]
    return [ax.contourf(xs, ys, field, levels=[0.0, field.max()], colors=[colours[0]], alpha=0.35),
            ax.contour(xs, ys, field, levels=[0.0], colors=[colours[1]], linewidths=1.2)]


def line_ends(line: Line) -> tuple[np.ndarray, np.ndarray]:
    """Two points of the line: its foot from the origin, where it meets its perpendicular through the origin, and
    one step along it, its direction where it meets the line at infinity."""
    foot = xy(line ^ (line | origin))
    step = (stack((mv.x, mv.y)) & (line ^ infinity)).to_array()
    return foot, foot + step / np.linalg.norm(step)


def animate_collision(
    first_ellipse: Quadric, first_tangent: Line, shares: Scalar,
    states: Iterable[tuple[Quadric, Line, Line, Polarity, Scalar, Scalar, Scalar]],
) -> Iterator[np.ndarray]:
    """The second ellipse at each moment, with the tangent lines of both ellipses along the normal, their midline,
    their contact points and the blend at the determinant's peak: a hyperbola parting the pair while apart, an
    ellipse bridging it once it overlaps; beside it, the blend's determinant against its share, with its peak.

    The view frames both ellipses of the first moment, and the determinant axis its curve and zero, `MARGIN` over
    their extent."""
    states = iter(states)
    first_state = next(states)
    placed_second, determinants = first_state[0], first_state[4].to_array()
    extent = np.concatenate([bounds(first_ellipse), bounds(placed_second)])
    low, high = extent.min(axis=0), extent.max(axis=0)
    centre, size = (low + high) / 2, high - low
    half_width = MARGIN * max(size[0], size[1] / ASPECT) / 2
    xlim = (centre[0] - half_width, centre[0] + half_width)
    ylim = (centre[1] - half_width * ASPECT, centre[1] + half_width * ASPECT)
    xs, ys = np.meshgrid(np.linspace(*xlim, RESOLUTION), np.linspace(*ylim, RESOLUTION))
    grid = mv.point(np.stack([xs, ys], axis=-1))
    bottom, top = min(determinants.min(), 0.0), max(determinants.max(), 0.0)
    middle, half_height = (bottom + top) / 2, MARGIN * (top - bottom) / 2

    figure, (ax_pair, ax_det) = plt.subplots(1, 2, figsize=(5.75, 2.5), dpi=90, gridspec_kw={"width_ratios": [1.45, 1]})
    draw_ellipse(ax_pair, first_ellipse, grid, FIRST_COLOURS)
    ax_pair.scatter(*xy(first_ellipse(infinity)), color=FIRST_COLOURS[1], s=12, zorder=5)
    second_centre = ax_pair.scatter([], [], color=SECOND_COLOURS[1], s=12, zorder=5)
    near = xy(first_ellipse(first_tangent))
    ax_pair.axline(*line_ends(first_tangent), color=FIRST_COLOURS[1], linestyle="--", linewidth=0.8, zorder=4)
    second_line = ax_pair.axline((0, 0), (1, 0), color=SECOND_COLOURS[1], linestyle="--", linewidth=0.8, zorder=4)
    midline_line = ax_pair.axline((0, 0), (1, 0), linestyle="-", linewidth=1.1, zorder=4)
    ax_pair.scatter(*near, color=FIRST_COLOURS[1], s=18, zorder=6)
    far_point = ax_pair.scatter([], [], color=SECOND_COLOURS[1], s=18, zorder=6)
    witness, = ax_pair.plot([], [], "k:", linewidth=1.0, zorder=5)
    ax_pair.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax_pair.set_axis_off()

    curve, = ax_det.plot(shares.to_array(), determinants, linewidth=1.3)
    peak_marker = ax_det.scatter([], [], s=21, zorder=6)
    ax_det.axhline(0.0, color="black", linestyle="--", linewidth=0.7, alpha=0.8)
    # Only the sign matters: the determinant's zero across.
    ax_det.set(xlim=(0.0, 1.0), ylim=(middle - half_height, middle + half_height), xticks=[], yticks=[])
    ax_det.set_xlabel("share of the blend", fontsize=7)
    ax_det.set_ylabel("determinant", fontsize=7)
    ax_det.spines[["top", "right"]].set_visible(False)
    # Lay the figure out once; only the second ellipse, its witnesses, the blend and the curve change between frames.
    figure.tight_layout()
    figure.set_layout_engine("none")

    changing = []
    for placed_second, second_tangent, midline, peak, determinants, parameter, maximum in chain([first_state], states):
        line_colour, edge_colour, fill_colour, fill_alpha = APART_COLOURS if maximum > 0 else OVERLAP_COLOURS
        second_centre.set_offsets(xy(placed_second(infinity))[None])
        for artist, line in ((second_line, second_tangent), (midline_line, midline)):
            start, end = line_ends(line)
            artist.set_xy1(*start)
            artist.set_xy2(*end)
        midline_line.set_color(line_colour)
        far = xy(placed_second(second_tangent))
        far_point.set_offsets(far[None])
        witness.set_data([near[0], far[0]], [near[1], far[1]])
        # The blend at the peak is drawn where each point lies on its own polar line, and filled on the side the
        # sign of its determinant selects: within the hyperbola's branches, and inside the bridging ellipse.
        for artist in changing:
            artist.remove()
        field = (grid & peak(grid)).to_array()
        side = field * np.sign(maximum.to_array())
        changing = draw_ellipse(ax_pair, placed_second, grid, SECOND_COLOURS) + [
            ax_pair.contour(xs, ys, field, levels=[0.0], colors=[edge_colour], linestyles=["-."], linewidths=0.9),
            ax_pair.contourf(xs, ys, side, levels=[0.0, np.abs(side).max()], colors=[fill_colour], alpha=fill_alpha),
        ]
        curve.set_ydata(determinants.to_array())
        curve.set_color(line_colour)
        peak_marker.set_offsets([[parameter.to_array(), maximum.to_array()]])
        peak_marker.set_color(line_colour)
        yield capture(figure)
    plt.close(figure)
