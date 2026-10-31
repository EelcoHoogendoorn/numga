"""The second ellipsoid sliding along the normal through contact with the first, beside the determinant of their blend."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from numga import concatenate, stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import contours, pinned, solids
from examples.util.wgsl.flow import axes_view
from examples.quadrics.quadric_collision.core import (
    Plane, Point, Polarity, Quadric, Scalar, infinity, mv, origin, tangent_plane,
)

# The height of the view over its width, and the margin around what the view frames, over its extent.
aspect = 0.75
margin = 1.15
# The area of an ellipsoid's centre's dot in square points: four pixels across, smaller than the contact points'.
centre_size = charts.points(4) ** 2
# The grid the ellipsoids and the blend at the peak are evaluated on, per side.
resolution = 250
first_colours, second_colours = ("#3b82f6", "#1d4ed8"), ("#f97316", "#ea580c")
# The opacity of the ellipsoids' fills.
ellipsoid_opacity = 0.35
# While the pair is apart and once it overlaps: the midplane, determinant curve and peak; the blend's edge; its fill
# and the fill's opacity.
apart_colours = ("#16a34a", "#059669", "#10b981", 0.10)
overlap_colours = ("#9333ea", "#7c3aed", "#a855f7", 0.22)
# The width of the blend's edge in pixels, thinner than the ellipsoids' outlines.
blend_edge_width = 1.0
# A volume, the join of three points, is an intrinsic scalar, and its unit carries the extrinsic to the intrinsic.
unit_volume = mv.antipseudoscalar([1])                                         # [] Antipseudoscalar-


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 2]`: their pairings with the planes x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (infinity & points)[..., None]).to_array()


def bounds(dual: Quadric) -> Point:
    """The low and high corners of the box around an ellipsoid: the contact points of its tangents facing along and
    against x and y."""
    contacts = dual(tangent_plane(dual, stack((mv.x, mv.y, -mv.x, -mv.y))))
    axes = stack((mv.x, mv.y))
    readings = axes[:, None] & (contacts / (infinity & contacts))
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
    return origin + (axes.dual() * stack((low, high))).sum(axis=-1)


def ellipsoid_form(dual: Quadric, grid: Point) -> np.ndarray:
    """An ellipsoid's form over the grid's points `[points]`: each point paired with its polar plane, under the inverse
    of the dual, signed by the centre's so that it is positive inside."""
    primal = dual.inverse()                                                    # [] Plane+ <- Point-
    centre = dual(infinity)                                                    # [] Point-
    return ((grid & primal(grid)) * (centre & primal(centre))).to_array()


def inside(colour: str, opacity: float) -> contours.Bands:
    """A field's positive side filled in a colour at an opacity, its negative side clear: two bands that meet at zero,
    each reaching on."""
    return contours.Bands(-1.0, 1.0, np.array([np.zeros(4), to_rgba(colour, opacity)]))


def clear_picture(ax, view: solids.View, zorder: float):
    """A clear picture pinned over the panel at a z-order, for an animation to set its data."""
    shown = pinned(ax, np.zeros(view.shape + (4,)), (*ax.get_xlim(), *ax.get_ylim()))
    shown.set_zorder(zorder)
    return shown


def plane_ends(plane: Plane) -> tuple[np.ndarray, np.ndarray]:
    """Two points of the plane, to draw it through: its foot from the origin, where it meets its perpendicular through
    the origin, and that foot moved along the plane by its direction, where it meets the plane at infinity."""
    foot = plane ^ (plane | origin)                                            # [] Point-
    # The meet of two planes is extrinsic; the intrinsic scalar one makes the direction a point like the foot.
    direction = (plane ^ infinity) * unit_volume                              # [] Point-
    return xy(foot), xy(foot + direction)


def animate_collision(
    first_ellipsoid: Quadric, first_tangent: Plane, shares: Scalar,
    states: Iterable[tuple[Quadric, Plane, Plane, Polarity, Scalar, Scalar, Scalar]],
) -> Iterator[np.ndarray]:
    """The second ellipsoid at each moment, with the tangent planes of both ellipsoids along the normal, their midplane,
    their contact points and the blend at the determinant's peak: a hyperboloid parting the pair while apart, an
    ellipsoid bridging it once it overlaps; beside it, the blend's determinant against its share, with its peak.

    The view frames both ellipsoids of the first moment, and the determinant axis its curve and zero, `margin` over
    their extent."""
    states = iter(states)
    first_state = next(states)
    placed_second, determinants = first_state[0], first_state[4].to_array()
    extent = concatenate([bounds(first_ellipsoid), bounds(placed_second)])
    readings = stack((mv.x, mv.y))[:, None] & extent
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
    middle = origin + (stack((mv.x, mv.y)).dual() * ((low + high) / 2)).sum()
    widths = (high - low) / np.array([1, aspect])
    half_width = margin * widths.max() / 2
    across = half_width * np.linspace(-1, 1, resolution)
    up = across * aspect
    # The grid's points numbered row by row, one row for each step up.
    grid = (middle + mv.x.dual() * across[None, :] + mv.y.dual() * up[:, None]).reshape(-1)   # [points] Point-
    mesh = contours.Mesh.grid(resolution, resolution)
    places = solids.context.include_from_algebra(grid)                         # [points] Point-
    # The panel's limits, at the grid's first and last corners.
    xlim, ylim = np.stack((xy(grid[0]), xy(grid[-1])), axis=-1)
    bottom, top = min(determinants.min(), 0.0), max(determinants.max(), 0.0)
    middle, half_height = (bottom + top) / 2, margin * (top - bottom) / 2

    # Three quarters of the column: the pair beside the determinant's curve and its two labels. The frames are drawn
    # anew, so the charts' style holds while they are drawn.
    with charts.style():
        figure, (ax_pair, ax_det) = charts.subplots(600, 260, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
        ax_pair.scatter(*xy(first_ellipsoid(infinity)), color=first_colours[1], s=centre_size, zorder=5)
        second_centre = ax_pair.scatter([], [], color=second_colours[1], s=centre_size, zorder=5)
        near = xy(first_ellipsoid(first_tangent))
        # The tangent planes thin guides, a frame's width, under the midplane.
        ax_pair.axline(*plane_ends(first_tangent), color=first_colours[1], linestyle="--", linewidth=charts.FRAME_WIDTH,
                       zorder=4)
        second_tangent_stroke = ax_pair.axline((0, 0), (1, 0), color=second_colours[1], linestyle="--",
                                               linewidth=charts.FRAME_WIDTH, zorder=4)
        midplane_stroke = ax_pair.axline((0, 0), (1, 0), linestyle="-", zorder=4)
        ax_pair.scatter(*near, color=first_colours[1], zorder=6)
        far_point = ax_pair.scatter([], [], color=second_colours[1], zorder=6)
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

        # The ellipsoids and the blend as clear pictures over the panel, at its own pixels: the fills under the
        # outlines, the first ellipsoid's drawn once, the second's and the blend's anew each frame.
        view = axes_view(ax_pair)
        first_form = ellipsoid_form(first_ellipsoid, grid)                    # [points]
        first_ink = np.array(to_rgba(first_colours[1]))
        clear_picture(ax_pair, view, 1.0).set_data(contours.picture(
            places, mesh, first_form, np.zeros(3), view, bands=(inside(first_colours[0], ellipsoid_opacity),),
            opacity=0.0))
        second_fill, blend_fill = clear_picture(ax_pair, view, 1.1), clear_picture(ax_pair, view, 1.2)
        clear_picture(ax_pair, view, 2.0).set_data(contours.picture(
            places, mesh, first_form, np.zeros(3), view, levels=(contours.Level(0.0, first_ink, charts.LINE_PX),),
            opacity=0.0))
        second_outline, blend_edge = clear_picture(ax_pair, view, 2.1), clear_picture(ax_pair, view, 2.2)
        second_ink = np.array(to_rgba(second_colours[1]))

        for (placed_second, second_tangent, midplane, peak, determinants, parameter,
             maximum) in chain([first_state], states):
            line_colour, edge_colour, fill_colour, fill_alpha = apart_colours if maximum > 0 else overlap_colours
            second_centre.set_offsets(xy(placed_second(infinity))[None])
            for artist, plane in ((second_tangent_stroke, second_tangent), (midplane_stroke, midplane)):
                start, end = plane_ends(plane)
                artist.set_xy1(*start)
                artist.set_xy2(*end)
            midplane_stroke.set_color(line_colour)
            far = xy(placed_second(second_tangent))
            far_point.set_offsets(far[None])
            witness.set_data([near[0], far[0]], [near[1], far[1]])
            second_form = ellipsoid_form(placed_second, grid)                 # [points]
            second_fill.set_data(contours.picture(
                places, mesh, second_form, np.zeros(3), view, bands=(inside(second_colours[0], ellipsoid_opacity),),
                opacity=0.0))
            second_outline.set_data(contours.picture(
                places, mesh, second_form, np.zeros(3), view,
                levels=(contours.Level(0.0, second_ink, charts.LINE_PX),), opacity=0.0))
            # The blend at the peak is drawn where each point lies on its own polar plane, and filled on the side the
            # sign of its determinant selects: within the hyperboloid's branches, and inside the bridging ellipsoid.
            side = (grid & peak(grid)).to_array() * np.sign(maximum.to_array())   # [points]
            blend_fill.set_data(contours.picture(
                places, mesh, side, np.zeros(3), view, bands=(inside(fill_colour, fill_alpha),), opacity=0.0))
            blend_edge.set_data(contours.picture(
                places, mesh, side, np.zeros(3), view,
                levels=(contours.Level(0.0, np.array(to_rgba(edge_colour)), blend_edge_width),), opacity=0.0))
            curve.set_ydata(determinants.to_array())
            curve.set_color(line_colour)
            peak_marker.set_offsets([[parameter.to_array(), maximum.to_array()]])
            peak_marker.set_color(line_colour)
            yield capture(figure)
        plt.close(figure)
