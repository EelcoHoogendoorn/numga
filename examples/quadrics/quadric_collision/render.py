"""The second ellipsoid sliding along the normal through contact with the first, beside the determinant of their blend."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.quadrics.quadric_collision.core import (
    Plane, Point, Polarity, Quadric, Scalar, infinity, mv, origin, tangent_plane,
)

# The height of the view over its width, and the margin around what the view frames, over its extent.
ASPECT = 0.75
MARGIN = 1.15
# The area of an ellipsoid's centre's dot in square points: four pixels across, smaller than the contact points'.
CENTRE_SIZE = charts.points(4) ** 2
# The grid the ellipsoids and the blend at the peak are evaluated on, per side.
RESOLUTION = 250
FIRST_COLOURS, SECOND_COLOURS = ("#3b82f6", "#1d4ed8"), ("#f97316", "#ea580c")
# While the pair is apart and once it overlaps: the midplane, determinant curve and peak; the blend's edge; its fill
# and the fill's opacity.
APART_COLOURS = ("#16a34a", "#059669", "#10b981", 0.10)
OVERLAP_COLOURS = ("#9333ea", "#7c3aed", "#a855f7", 0.22)


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 2]`: their pairings with the planes x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (infinity & points)[..., None]).to_array()


def bounds(dual: Quadric) -> np.ndarray:
    """The low and high corners of the box around an ellipsoid: the contact points of its tangents facing along and
    against x and y."""
    contacts = xy(dual(tangent_plane(dual, stack((mv.x, mv.y, -mv.x, -mv.y)))))
    return np.stack([contacts.min(axis=0), contacts.max(axis=0)])


def draw_ellipsoid(ax, dual: Quadric, grid: Point, colours: tuple[str, str]) -> list:
    """An ellipsoid filled and outlined where its form over the grid changes sign: each point paired with its polar
    plane, under the inverse of the dual, signed by the centre's so that it is positive inside; the artists, for an
    animation to remove."""
    primal = dual.inverse()                                                    # [] Plane+ <- Point-
    centre = dual(infinity)                                                    # [] Point-
    field = ((grid & primal(grid)) * (centre & primal(centre))).to_array()
    coordinates = xy(grid)
    xs, ys = coordinates[..., 0], coordinates[..., 1]
    return [ax.contourf(xs, ys, field, levels=[0.0, field.max()], colors=[colours[0]], alpha=0.35),
            ax.contour(xs, ys, field, levels=[0.0], colors=[colours[1]])]


def plane_ends(plane: Plane) -> tuple[np.ndarray, np.ndarray]:
    """Two points of the plane, to draw it through: its foot from the origin, where it meets its perpendicular through
    the origin, and one step along it, its direction where it meets the plane at infinity."""
    foot = xy(plane ^ (plane | origin))
    step = (stack((mv.x, mv.y)) & (plane ^ infinity)).to_array()
    return foot, foot + step / np.linalg.norm(step)


def animate_collision(
    first_ellipsoid: Quadric, first_tangent: Plane, shares: Scalar,
    states: Iterable[tuple[Quadric, Plane, Plane, Polarity, Scalar, Scalar, Scalar]],
) -> Iterator[np.ndarray]:
    """The second ellipsoid at each moment, with the tangent planes of both ellipsoids along the normal, their midplane,
    their contact points and the blend at the determinant's peak: a hyperboloid parting the pair while apart, an
    ellipsoid bridging it once it overlaps; beside it, the blend's determinant against its share, with its peak.

    The view frames both ellipsoids of the first moment, and the determinant axis its curve and zero, `MARGIN` over
    their extent."""
    states = iter(states)
    first_state = next(states)
    placed_second, determinants = first_state[0], first_state[4].to_array()
    extent = np.concatenate([bounds(first_ellipsoid), bounds(placed_second)])
    low, high = extent.min(axis=0), extent.max(axis=0)
    centre, size = (low + high) / 2, high - low
    half_width = MARGIN * max(size[0], size[1] / ASPECT) / 2
    xlim = (centre[0] - half_width, centre[0] + half_width)
    ylim = (centre[1] - half_width * ASPECT, centre[1] + half_width * ASPECT)
    xs, ys = np.meshgrid(np.linspace(*xlim, RESOLUTION), np.linspace(*ylim, RESOLUTION))
    grid = mv.point(np.stack([xs, ys], axis=-1))
    bottom, top = min(determinants.min(), 0.0), max(determinants.max(), 0.0)
    middle, half_height = (bottom + top) / 2, MARGIN * (top - bottom) / 2

    # Three quarters of the column: the pair beside the determinant's curve and its two labels. The frames draw new
    # contours, so the charts' style holds while they are drawn.
    with charts.style():
        figure, (ax_pair, ax_det) = charts.subplots(600, 260, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
        draw_ellipsoid(ax_pair, first_ellipsoid, grid, FIRST_COLOURS)
        ax_pair.scatter(*xy(first_ellipsoid(infinity)), color=FIRST_COLOURS[1], s=CENTRE_SIZE, zorder=5)
        second_centre = ax_pair.scatter([], [], color=SECOND_COLOURS[1], s=CENTRE_SIZE, zorder=5)
        near = xy(first_ellipsoid(first_tangent))
        # The tangent planes thin guides, a frame's width, under the midplane.
        ax_pair.axline(*plane_ends(first_tangent), color=FIRST_COLOURS[1], linestyle="--", linewidth=charts.FRAME_WIDTH,
                       zorder=4)
        second_tangent_stroke = ax_pair.axline((0, 0), (1, 0), color=SECOND_COLOURS[1], linestyle="--",
                                               linewidth=charts.FRAME_WIDTH, zorder=4)
        midplane_stroke = ax_pair.axline((0, 0), (1, 0), linestyle="-", zorder=4)
        ax_pair.scatter(*near, color=FIRST_COLOURS[1], zorder=6)
        far_point = ax_pair.scatter([], [], color=SECOND_COLOURS[1], zorder=6)
        witness, = ax_pair.plot([], [], "k:", zorder=5)
        ax_pair.set(xlim=xlim, ylim=ylim, aspect="equal")
        ax_pair.set_axis_off()

        curve, = ax_det.plot(shares.to_array(), determinants)
        peak_marker = ax_det.scatter([], [], zorder=6)
        ax_det.axhline(0.0, color="black", linestyle="--", linewidth=charts.FRAME_WIDTH, alpha=0.8)
        # Only the sign matters: the determinant's zero across.
        ax_det.set(xlim=(0.0, 1.0), ylim=(middle - half_height, middle + half_height), xticks=[], yticks=[])
        ax_det.set_xlabel("share of the blend")
        ax_det.set_ylabel("determinant")
        ax_det.spines[["top", "right"]].set_visible(False)
        # Lay the figure out once; only the second ellipsoid, its witnesses, the blend and the curve change between
        # frames.
        figure.tight_layout()
        figure.set_layout_engine("none")

        changing = []
        for (placed_second, second_tangent, midplane, peak, determinants, parameter,
             maximum) in chain([first_state], states):
            line_colour, edge_colour, fill_colour, fill_alpha = APART_COLOURS if maximum > 0 else OVERLAP_COLOURS
            second_centre.set_offsets(xy(placed_second(infinity))[None])
            for artist, plane in ((second_tangent_stroke, second_tangent), (midplane_stroke, midplane)):
                start, end = plane_ends(plane)
                artist.set_xy1(*start)
                artist.set_xy2(*end)
            midplane_stroke.set_color(line_colour)
            far = xy(placed_second(second_tangent))
            far_point.set_offsets(far[None])
            witness.set_data([near[0], far[0]], [near[1], far[1]])
            # The blend at the peak is drawn where each point lies on its own polar plane, and filled on the side the
            # sign of its determinant selects: within the hyperboloid's branches, and inside the bridging ellipsoid.
            for artist in changing:
                artist.remove()
            field = (grid & peak(grid)).to_array()
            side = field * np.sign(maximum.to_array())
            changing = draw_ellipsoid(ax_pair, placed_second, grid, SECOND_COLOURS) + [
                ax_pair.contour(xs, ys, field, levels=[0.0], colors=[edge_colour], linestyles=["-."]),
                ax_pair.contourf(xs, ys, side, levels=[0.0, np.abs(side).max()], colors=[fill_colour],
                                 alpha=fill_alpha),
            ]
            curve.set_ydata(determinants.to_array())
            curve.set_color(line_colour)
            peak_marker.set_offsets([[parameter.to_array(), maximum.to_array()]])
            peak_marker.set_color(line_colour)
            yield capture(figure)
        plt.close(figure)
