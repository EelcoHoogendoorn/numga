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
from examples.util.wgsl import contours, pinned, simplices
from examples.util.wgsl.camera import Camera, panel
from examples.quadrics.quadric_collision.core import (
    Plane, Point, Polarity, Quadric, Scalar, infinity, mv, origin, tangent_plane,
)

# The height of the view over its width, and the margin around what the view frames, over its extent.
aspect = 0.75
margin = 1.15
# The radius of an ellipsoid's centre's dot in pixels, four across, smaller than the contact points'.
centre_radius = 2.0
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
# The tangent planes' dashes and the witness's dots: a dash and a gap, in pixels.
tangent_dashes, witness_dots = (4.0, 2.0), (0.0, 4.0)
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


def plane_ends(plane: Plane, reach: float) -> Point:
    """Two points of the plane, to draw it through the panel: `reach` either way along it from its foot from the
    origin, where it meets its perpendicular through the origin; its direction is where it meets the plane at
    infinity. `[2] Point-`"""
    unit = plane.normalized()                                                  # [] Plane+
    foot = unit ^ (unit | origin)                                              # [] Point-
    # The meet of two planes is extrinsic; the intrinsic scalar one makes the direction a point like the foot.
    direction = (unit ^ infinity) * unit_volume                                # [] Point-
    return stack((foot - direction * reach, foot + direction * reach))


def dots(points: Point, colour: str, radius: float, camera: Camera) -> np.ndarray:
    """Points as dots of a colour, `radius` pixels, read by the camera: their picture."""
    return simplices.markers(points, np.array(to_rgba(colour))[:3], radius, camera)


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
    topology = contours.Topology.grid(resolution, resolution)
    places = grid                                                                             # [points] Point-
    # The panel's limits, at the grid's first and last corners.
    xlim, ylim = np.stack((xy(grid[0]), xy(grid[-1])), axis=-1)
    bottom, top = min(determinants.min(), 0.0), max(determinants.max(), 0.0)
    middle, half_height = (bottom + top) / 2, margin * (top - bottom) / 2

    # Three quarters of the column: the pair beside the determinant's curve and its two labels. The frames are drawn
    # anew, so the charts' style holds while they are drawn.
    with charts.style():
        figure, (ax_pair, ax_det) = charts.subplots(600, 260, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
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

        # The pair as one picture over the panel, at its own pixels, laid anew each frame from its layers: the fills
        # under the outlines, under the planes, under the centres, the witness and the contact points. The first
        # ellipsoid's layers are drawn once.
        camera = panel(ax_pair, mv.x, mv.y, infinity)
        pair = pinned(ax_pair, np.zeros(camera.shape + (4,)), (*ax_pair.get_xlim(), *ax_pair.get_ylim()))
        # The planes run on past the panel's corners either way from their feet: a plane's foot lies no farther from
        # the origin than its points in the panel, so twice the farthest corner's distance, the length of its join
        # with the origin, reaches past them all.
        corners = grid[[0, resolution - 1, -resolution, -1]]                  # [4] Point-
        reach = 2 * (origin & corners).norm().max()                            # [] Scalar+
        first_form = ellipsoid_form(first_ellipsoid, grid)                    # [points]
        first_ink = np.array(to_rgba(first_colours[1]))
        second_ink = np.array(to_rgba(second_colours[1]))
        first_fill = contours.picture(places, topology, first_form, np.zeros(3), camera,
                                      bands=(inside(first_colours[0], ellipsoid_opacity),), opacity=0.0)
        first_outline = contours.picture(places, topology, first_form, np.zeros(3), camera,
                                         levels=(contours.Level(0.0, first_ink, charts.LINE_PX),), opacity=0.0)
        near = first_ellipsoid(first_tangent)                                  # [] Point-
        first_centre = dots(first_ellipsoid(infinity)[None], first_colours[1], centre_radius, camera)
        near_point = dots(near[None], first_colours[1], charts.MARKER_PX / 2, camera)
        # The tangent planes thin guides, a frame's width, under the midplane.
        tangent_strokes = simplices.Topology.chains([2])
        first_guide = simplices.strokes(plane_ends(first_tangent, reach), tangent_strokes, first_ink[:3],
                                        charts.FRAME_PX, camera, dashes=tangent_dashes, cap=simplices.Cap.BUTT)

        for (placed_second, second_tangent, midplane, peak, determinants, parameter,
             maximum) in chain([first_state], states):
            line_colour, edge_colour, fill_colour, fill_alpha = apart_colours if maximum > 0 else overlap_colours
            second_form = ellipsoid_form(placed_second, grid)                 # [points]
            second_fill = contours.picture(places, topology, second_form, np.zeros(3), camera,
                                           bands=(inside(second_colours[0], ellipsoid_opacity),), opacity=0.0)
            second_outline = contours.picture(places, topology, second_form, np.zeros(3), camera,
                                              levels=(contours.Level(0.0, second_ink, charts.LINE_PX),), opacity=0.0)
            # The blend at the peak is drawn where each point lies on its own polar plane, and filled on the side the
            # sign of its determinant selects: within the hyperboloid's branches, and inside the bridging ellipsoid.
            side = (grid & peak(grid)).to_array() * np.sign(maximum.to_array())  # [points]
            blend_fill = contours.picture(places, topology, side, np.zeros(3), camera,
                                          bands=(inside(fill_colour, fill_alpha),), opacity=0.0)
            blend_edge = contours.picture(places, topology, side, np.zeros(3), camera,
                                          levels=(contours.Level(0.0, np.array(to_rgba(edge_colour)),
                                                                 blend_edge_width),), opacity=0.0)
            second_guide = simplices.strokes(plane_ends(second_tangent, reach), tangent_strokes, second_ink[:3],
                                             charts.FRAME_PX, camera, dashes=tangent_dashes, cap=simplices.Cap.BUTT)
            separator = simplices.strokes(plane_ends(midplane, reach), tangent_strokes,
                                          np.array(to_rgba(line_colour))[:3], charts.LINE_PX, camera)
            far = placed_second(second_tangent)                                # [] Point-
            witness = simplices.strokes(stack((near, far)), tangent_strokes, np.zeros(3), charts.LINE_PX, camera,
                                        dashes=witness_dots)
            pair.set_data(simplices.composited([
                first_fill, second_fill, blend_fill, first_outline, second_outline, blend_edge,
                first_guide, second_guide, separator,
                first_centre, dots(placed_second(infinity)[None], second_colours[1], centre_radius, camera), witness,
                near_point, dots(far[None], second_colours[1], charts.MARKER_PX / 2, camera)]))
            curve.set_ydata(determinants.to_array())
            curve.set_color(line_colour)
            peak_marker.set_offsets([[parameter.to_array(), maximum.to_array()]])
            peak_marker.set_color(line_colour)
            yield capture(figure)
        plt.close(figure)
