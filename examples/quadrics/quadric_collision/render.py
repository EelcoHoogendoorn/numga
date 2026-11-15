"""Implicit planar rendering shows the sliding ellipsoid, both outward tangents, their midplane, the peak blend conic, and the cubic determinant curve."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from numga import concatenate, stack
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.quadrics.quadric_collision.core import (
    Plane, Point, Polarity, Quadric, Scalar, infinity, mv, origin, tangent_plane,
)
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import panel, pinned

# Height-to-width ratio of the geometric panel and margin multiplier around the initial pair.
aspect = 0.75
margin = 1.15
# Radius of each ellipsoid centre dot in pixels, smaller than the contact markers.
centre_radius = 2.0
# Per-side sample count of the grid on which the ellipsoids and peak blend are contoured.
n_grid = 250
first_colours, second_colours = ("#3b82f6", "#1d4ed8"), ("#f97316", "#ea580c")
ellipsoid_opacity = 0.35
# Line/curve colour, blend outline colour, blend fill colour, and fill opacity when apart vs. overlapping.
apart_colours = ("#16a34a", "#059669", "#10b981", 0.10)
overlap_colours = ("#9333ea", "#7c3aed", "#a855f7", 0.22)
blend_edge_width = 1.0
tangent_dashes, witness_dots = (4.0, 2.0), (0.0, 4.0)


# --- plumbing -------------------------------------------------------------------------
def point_coordinates(points: Point) -> np.ndarray:
    """Coordinate readout (`[..., 2]`) pairs `points` (`[...] Point-`) against `x = 0` and `y = 0` over their projective weight."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (infinity & points)[..., None]).to_array()


def bounding_corners(dual: Quadric) -> Point:
    """The axis-aligned bounding-box corners (`[2] Point-`) of `dual` come from its four coordinate-axis tangent contacts."""
    axes = stack((mv.x, mv.y))                                                 # [2] Plane+
    contacts = dual(tangent_plane(dual, stack((mv.x, mv.y, -mv.x, -mv.y))))    # [4] Point-
    readings = axes[:, None] & (contacts / (infinity & contacts))              # [2, 4] Scalar+
    low = readings.min(axis=-1)                                                # [2] Scalar+
    high = readings.max(axis=-1)                                               # [2] Scalar+
    return origin + (axes.dual() * stack((low, high))).sum(axis=-1)


def ellipsoid_field(dual: Quadric, grid: Point) -> np.ndarray:
    """Pairing `grid` (`[...] Point-`) with its polar planes under `dual.solve` and signing by the centre makes the interior positive."""
    centre = dual(infinity)                                                    # [] Point-
    return ((dual.solve(grid) & grid) * (dual.solve(centre) & centre)).to_array()


def interior_band(colour: str, opacity: float) -> contours.Bands:
    """A two-band fill leaves the negative side transparent and colours the positive interior at `opacity`."""
    return contours.Bands(-1.0, 1.0, np.array([np.zeros(4), to_rgba(colour, opacity)]))


def plane_segment(plane: Plane, reach: Scalar) -> Point:
    """Two points (`[2] Point-`) at distance `reach` either side of the perpendicular foot from `origin` span `plane` across the panel."""
    unit = plane.normalized()                                                  # [] Plane+
    perp = unit | origin                                                       # [] Line-
    foot = unit ^ perp                                                         # [] Point-
    # Contracting `origin` with `unit` and wedging with `infinity` gives the unit ideal point along `unit`.
    tangent_direction = infinity ^ perp                                        # [] Point-
    return stack((foot - tangent_direction * reach, foot + tangent_direction * reach))


def point_markers(points: Point, colour: str, radius: float, camera: Camera) -> np.ndarray:
    """Renders `points` (`[...] Point-`) through `camera` as circular markers of `radius` pixels."""
    return simplices.markers(points, np.array(to_rgba(colour))[:3], radius, camera)


def collision_animation(
    first_ellipsoid: Quadric,
    first_tangent: Plane,
    shares: Scalar,
    states: Iterable[tuple[Quadric, Plane, Plane, Polarity, Scalar, Scalar, Scalar]],
) -> Iterator[np.ndarray]:
    """Yields frames showing the sliding ellipsoid, both outward tangents, their midplane, the peak blend conic, and the determinant curve."""
    states = iter(states)
    first_state = next(states)
    placed_second, determinants = first_state[0], first_state[4].to_array()
    extent = concatenate([bounding_corners(first_ellipsoid), bounding_corners(placed_second)])   # [4] Point-
    readings = stack((mv.x, mv.y))[:, None] & extent                           # [2, 4] Scalar+
    low = readings.min(axis=-1)                                                # [2] Scalar+
    high = readings.max(axis=-1)                                               # [2] Scalar+
    view_centre = origin + (stack((mv.x, mv.y)).dual() * ((low + high) / 2)).sum()   # [] Point-
    widths = (high - low).to_array() / np.array([1.0, aspect])
    half_width = margin * widths.max() / 2
    across = half_width * np.linspace(-1.0, 1.0, n_grid)
    up = across * aspect
    grid = (view_centre + mv.x.dual() * across[None, :] + mv.y.dual() * up[:, None]).reshape(-1)   # [n_grid * n_grid] Point-
    topology = Topology.grid(n_grid, n_grid)
    xlim, ylim = np.stack((point_coordinates(grid[0]), point_coordinates(grid[-1])), axis=-1)
    bottom, top = min(determinants.min(), 0.0), max(determinants.max(), 0.0)
    det_mid, half_height = (bottom + top) / 2, margin * (top - bottom) / 2

    with charts.style():
        figure, (ax_pair, ax_det) = charts.subplots(600, 260, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
        ax_pair.set(xlim=xlim, ylim=ylim, aspect="equal")
        ax_pair.set_axis_off()

        curve, = ax_det.plot(shares.to_array(), determinants)
        peak_marker = ax_det.scatter([], [], zorder=6)
        ax_det.axhline(0.0, color="black", linestyle="--", linewidth=charts.FRAME_WIDTH, alpha=0.8)
        ax_det.set(xlim=(0.0, 1.0), ylim=(det_mid - half_height, det_mid + half_height), xticks=[], yticks=[])
        ax_det.set_xlabel("share of the blend")
        ax_det.set_ylabel("determinant")
        ax_det.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.set_layout_engine("none")

        camera = panel(ax_pair, mv.x, mv.y, infinity)
        pair_layer = pinned(ax_pair, np.zeros(camera.shape + (4,)), (*ax_pair.get_xlim(), *ax_pair.get_ylim()))
        # Twice the farthest panel corner's distance from the origin guarantees every line's segment clears the frame.
        panel_corners = grid[[0, n_grid - 1, -n_grid, -1]]                     # [4] Point-
        reach = 2 * (origin & panel_corners).norm().max()                      # [] Scalar+
        first_form = ellipsoid_field(first_ellipsoid, grid)                    # [n_grid * n_grid]
        first_ink = np.array(to_rgba(first_colours[1]))
        second_ink = np.array(to_rgba(second_colours[1]))
        first_fill = contours.picture(grid, topology, first_form, np.zeros(3), camera,
                                      bands=(interior_band(first_colours[0], ellipsoid_opacity),), opacity=0.0)
        first_outline = contours.picture(grid, topology, first_form, np.zeros(3), camera,
                                         levels=(contours.Level(0.0, first_ink, charts.LINE_PX),), opacity=0.0)
        near_contact = first_ellipsoid(first_tangent)                          # [] Point-
        first_centre = point_markers(first_ellipsoid(infinity)[None], first_colours[1], centre_radius, camera)
        near_marker = point_markers(near_contact[None], first_colours[1], charts.MARKER_PX / 2, camera)
        segment_chain = Topology.chains([2])
        first_guide = simplices.strokes(plane_segment(first_tangent, reach), segment_chain, first_ink[:3],
                                        charts.FRAME_PX, camera, dashes=tangent_dashes, cap=simplices.Cap.BUTT)

        for (placed_second, second_tangent, midplane, peak_polarity, det_values, peak_share,
             peak_det) in chain([first_state], states):
            line_colour, edge_colour, fill_colour, fill_alpha = apart_colours if peak_det > 0 else overlap_colours
            second_form = ellipsoid_field(placed_second, grid)                 # [n_grid * n_grid]
            second_fill = contours.picture(grid, topology, second_form, np.zeros(3), camera,
                                           bands=(interior_band(second_colours[0], ellipsoid_opacity),), opacity=0.0)
            second_outline = contours.picture(grid, topology, second_form, np.zeros(3), camera,
                                              levels=(contours.Level(0.0, second_ink, charts.LINE_PX),), opacity=0.0)
            # Signing `peak_polarity(grid) & grid` by `peak_det` fills between the separating hyperbola branches when apart and inside the bridging ellipse when overlapping.
            side = (peak_polarity(grid) & grid).to_array() * np.sign(peak_det.to_array())   # [n_grid * n_grid]
            blend_fill = contours.picture(grid, topology, side, np.zeros(3), camera,
                                          bands=(interior_band(fill_colour, fill_alpha),), opacity=0.0)
            blend_edge = contours.picture(grid, topology, side, np.zeros(3), camera,
                                          levels=(contours.Level(0.0, np.array(to_rgba(edge_colour)),
                                                                 blend_edge_width),), opacity=0.0)
            second_guide = simplices.strokes(plane_segment(second_tangent, reach), segment_chain, second_ink[:3],
                                             charts.FRAME_PX, camera, dashes=tangent_dashes, cap=simplices.Cap.BUTT)
            separator = simplices.strokes(plane_segment(midplane, reach), segment_chain,
                                          np.array(to_rgba(line_colour))[:3], charts.LINE_PX, camera)
            far_contact = placed_second(second_tangent)                        # [] Point-
            witness = simplices.strokes(stack((near_contact, far_contact)), segment_chain, np.zeros(3), charts.LINE_PX,
                                        camera, dashes=witness_dots)
            pair_layer.set_data(drawing.composited([
                first_fill, second_fill, blend_fill, first_outline, second_outline, blend_edge,
                first_guide, second_guide, separator,
                first_centre, point_markers(placed_second(infinity)[None], second_colours[1], centre_radius, camera),
                witness, near_marker, point_markers(far_contact[None], second_colours[1], charts.MARKER_PX / 2, camera),
            ]))
            curve.set_ydata(det_values.to_array())
            curve.set_color(line_colour)
            peak_marker.set_offsets([[peak_share.to_array(), peak_det.to_array()]])
            peak_marker.set_color(line_colour)
            yield capture(figure)
        plt.close(figure)
