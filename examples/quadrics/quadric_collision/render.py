"""The second ellipse sliding along the normal through contact with the first, beside the determinant of their blend."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.util.animation import capture
from examples.quadrics.quadric_collision.core import (
    Line, Point, Polarity, Quadric, Scalar, ga, infinity, mv, tangent_line,
)

# The height of the view over its width.
ASPECT = 0.75
# The grid the blend at the peak is evaluated on, per side, and the tangent normals an outline is swept over.
RESOLUTION = 250
N_OUTLINE = 150
FIRST_COLOURS, SECOND_COLOURS = ("#3b82f6", "#1d4ed8"), ("#f97316", "#ea580c")
# While the pair is apart and once it overlaps: the midline, determinant curve and peak; the blend's edge; its fill
# and the fill's opacity.
APART_COLOURS = ("#16a34a", "#059669", "#10b981", 0.10)
OVERLAP_COLOURS = ("#9333ea", "#7c3aed", "#a855f7", 0.22)


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 2]`: their pairings with the lines x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (infinity & points)[..., None]).to_array()


def outline(dual: Quadric) -> np.ndarray:
    """Coordinates of the ellipse's contact points, swept over the tangent normals."""
    theta = np.linspace(0, 2 * np.pi, N_OUTLINE)
    return xy(dual(tangent_line(dual, mv.x * np.cos(theta) + mv.y * np.sin(theta))))


def line_ends(line: Line) -> tuple[np.ndarray, np.ndarray]:
    """Two points of the line `mv.x * a + mv.y * b + mv.w * c`: its foot from the origin, and one step along it."""
    a, b, c = line.cast(ga.subspace("x y w")).kernel
    foot = -np.array([a, b]) * c / (a * a + b * b)
    return foot, foot + np.array([-b, a])


def animate_collision(
    fixed: Quadric, first: Line, shares: Scalar,
    states: Iterable[tuple[Quadric, Line, Line, Polarity, Scalar, Scalar, Scalar]], margin: float = 1.15,
) -> Iterator[np.ndarray]:
    """The second ellipse at each moment, with the tangent lines of both ellipses along the normal, their midline,
    their contact points and the blend at the determinant's peak: a hyperbola parting the pair while apart, an
    ellipse bridging it once it overlaps; beside it, the blend's determinant against its share, with its peak.

    The view frames both ellipses of the first moment, and the determinant axis its curve and zero, `margin` over
    their extent."""
    states = iter(states)
    first_state = next(states)
    moved, determinants = first_state[0], first_state[4].to_array()
    fixed_points = outline(fixed)
    extent = np.concatenate([fixed_points, outline(moved)])
    low, high = extent.min(axis=0), extent.max(axis=0)
    centre, size = (low + high) / 2, high - low
    half_width = margin * max(size[0], size[1] / ASPECT) / 2
    xlim = (centre[0] - half_width, centre[0] + half_width)
    ylim = (centre[1] - half_width * ASPECT, centre[1] + half_width * ASPECT)
    xs, ys = np.meshgrid(np.linspace(*xlim, RESOLUTION), np.linspace(*ylim, RESOLUTION))
    grid = mv.point(np.stack([xs, ys], axis=-1))
    bottom, top = min(determinants.min(), 0.0), max(determinants.max(), 0.0)
    middle, half_height = (bottom + top) / 2, margin * (top - bottom) / 2

    figure, (ax_pair, ax_det) = plt.subplots(1, 2, figsize=(11.5, 5), dpi=90, gridspec_kw={"width_ratios": [1.45, 1]})
    ax_pair.fill(fixed_points[:, 0], fixed_points[:, 1], color=FIRST_COLOURS[0], alpha=0.35)
    ax_pair.plot(fixed_points[:, 0], fixed_points[:, 1], color=FIRST_COLOURS[1], linewidth=2.2)
    ax_pair.scatter(*xy(fixed(infinity)), color=FIRST_COLOURS[1], s=40, zorder=5)
    moved_fill, = ax_pair.fill([], [], color=SECOND_COLOURS[0], alpha=0.35)
    moved_edge, = ax_pair.plot([], [], color=SECOND_COLOURS[1], linewidth=2.2)
    moved_centre = ax_pair.scatter([], [], color=SECOND_COLOURS[1], s=40, zorder=5)
    near = xy(fixed(first))
    ax_pair.axline(*line_ends(first), color=FIRST_COLOURS[1], linestyle="--", linewidth=1.5, zorder=4)
    second_line = ax_pair.axline((0, 0), (1, 0), color=SECOND_COLOURS[1], linestyle="--", linewidth=1.5, zorder=4)
    midline_line = ax_pair.axline((0, 0), (1, 0), linestyle="-", linewidth=2.0, zorder=4)
    ax_pair.scatter(*near, color=FIRST_COLOURS[1], s=60, zorder=6)
    far_point = ax_pair.scatter([], [], color=SECOND_COLOURS[1], s=60, zorder=6)
    witness, = ax_pair.plot([], [], "k:", linewidth=1.8, zorder=5)
    ax_pair.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax_pair.set_axis_off()

    curve, = ax_det.plot(shares.to_array(), determinants, linewidth=2.4)
    peak_marker = ax_det.scatter([], [], s=70, zorder=6)
    ax_det.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    # Only the sign matters: the determinant's zero across.
    ax_det.set(xlim=(0.0, 1.0), ylim=(middle - half_height, middle + half_height), xticks=[], yticks=[],
               xlabel="share of the blend", ylabel="determinant")
    ax_det.spines[["top", "right"]].set_visible(False)
    # Lay the figure out once; only the second ellipse, its witnesses, the blend and the curve change between frames.
    figure.tight_layout()
    figure.set_layout_engine("none")

    blend_artists = []
    for moved, second, midline, peak, determinants, parameter, maximum in chain([first_state], states):
        line_colour, edge_colour, fill_colour, fill_alpha = APART_COLOURS if maximum.to_array() > 0 else OVERLAP_COLOURS
        points = outline(moved)
        moved_fill.set_xy(points)
        moved_edge.set_data(points[:, 0], points[:, 1])
        moved_centre.set_offsets(xy(moved(infinity))[None])
        for artist, line in ((second_line, second), (midline_line, midline)):
            start, end = line_ends(line)
            artist.set_xy1(*start)
            artist.set_xy2(*end)
        midline_line.set_color(line_colour)
        far = xy(moved(second))
        far_point.set_offsets(far[None])
        witness.set_data([near[0], far[0]], [near[1], far[1]])
        # The blend at the peak is drawn where each point lies on its own polar line, and filled on the side the
        # sign of its determinant selects: within the hyperbola's branches, and inside the bridging ellipse.
        for artist in blend_artists:
            artist.remove()
        field = (grid & peak(grid)).to_array()
        side = field * np.sign(maximum.to_array())
        blend_artists = [
            ax_pair.contour(xs, ys, field, levels=[0.0], colors=[edge_colour], linestyles=["-."], linewidths=1.6),
            ax_pair.contourf(xs, ys, side, levels=[0.0, np.abs(side).max()], colors=[fill_colour], alpha=fill_alpha),
        ]
        curve.set_ydata(determinants.to_array())
        curve.set_color(line_colour)
        peak_marker.set_offsets([[parameter.to_array(), maximum.to_array()]])
        peak_marker.set_color(line_colour)
        yield capture(figure)
    plt.close(figure)
