"""GPU views render suspended plates and their rigid-body vibration modes in the plate's own algebra."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from examples.util.animation import capture
from examples.util.panels import overlay, panel
from numga import concatenate, stack
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.mechanics.modes.core import Point, Scalar, Twist, mv


body_colour, ghost_colour, support_colour, spring_colour, plate_colour = (
    to_rgb(colour) for colour in ("#254e70", "#929ba7", "#3a414b", "#5b6470", "#d4e3ed")
)
body_width, ghost_width, spring_width = 2.0, 1.5, 1.5
ghost_dots = (0.0, 3.5)
support_radius, attachment_radius = 2.6, 2.0
peak_corner_excursion = 0.20

# Each spring's coil path combines straight end leads with alternating transverse zigzags.
n_coil_turns = 15
n_coil_points = n_coil_turns + 4
coil_fractions = np.r_[0.0, 0.16, np.linspace(0.20, 0.80, n_coil_turns), 0.84, 1.0]
coil_offsets = np.r_[0.0, 0.0, 0.035 * (-1.0) ** np.arange(n_coil_turns), 0.0, 0.0]


def coordinates(points: Point) -> np.ndarray:
    """Pairing with the coordinate planes reads `[..., 2]` Cartesian coordinates of unit-weight points or directions."""
    return (stack((mv.x, mv.y)) & points[..., None]).to_array()


def bounding_box(point_batches: list[Point]) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate pairings across `point_batches` give the `[2]` lower and upper framing bounds."""
    readings = concatenate([stack((mv.x, mv.y))[:, None] & batch.reshape(-1) for batch in point_batches], axis=-1)
    return readings.min(axis=-1).to_array(), readings.max(axis=-1).to_array()


def spring_paths(starts: Point, ends: Point) -> Point:
    """Straight end leads and alternating transverse offsets trace `[..., n_coil_points]` coil vertices (`Point-`) between `starts` and `ends`."""
    spans = ends - starts                                                        # [...] Point-
    # Commuting with the origin bivector turns each span a quarter turn, divided by the join line's Euclidean length.
    transverse = spans.commutator(mv.xy) / (starts & ends).norm()                # [...] Point-
    return starts[:, None] + spans[:, None] * coil_fractions + transverse[:, None] * coil_offsets


def plate_surface(n_corners: int) -> Topology:
    """Triangles fanning from the first vertex cover a convex `n_corners`-gon."""
    indices = np.arange(n_corners)
    return Topology(n_corners, np.stack([np.zeros(n_corners - 2, dtype=int), indices[1:-1], indices[2:]], -1))


def suspension_layers(camera: Camera, corners: Point, anchors: Point, attachments: Point) -> list[np.ndarray]:
    """Filled plate triangles, rim strokes, spring coils, and endpoint markers form the RGBA layers of one suspension."""
    n_corners = len(corners)
    n_springs = len(anchors)
    surface = plate_surface(n_corners)
    coil_points = spring_paths(anchors, attachments).reshape(-1)                 # [n_springs * n_coil_points] Point-
    coils = Topology.chains([n_coil_points] * n_springs)
    return [
        contours.picture(corners, surface, np.zeros(n_corners), plate_colour, camera),
        simplices.strokes(corners, surface.border, body_colour, body_width, camera),
        simplices.markers(anchors, support_colour, support_radius, camera),
        simplices.strokes(coil_points, coils, spring_colour, spring_width, camera),
        simplices.markers(attachments, body_colour, attachment_radius, camera),
    ]


def mode_canvas(corners: Point, anchors: Point, dpi: int) -> tuple[plt.Figure, list[plt.Axes]]:
    """Three equal-aspect axes in one row frame `corners` and `anchors`."""
    fig, axes_row = plt.subplots(1, 3, figsize=(9, 2.7), dpi=dpi, facecolor="white")
    low, high = bounding_box([corners, anchors])
    lower, upper = low - 0.35, high + 0.25
    axes = list(axes_row)
    for ax in axes:
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.04, hspace=0.04)
    return fig, axes


def excursion_scales(corners: Point, mode_twists: Twist) -> Scalar:
    """Display multipliers (`[3] Scalar+`) normalize each mode twist to a peak corner offset of `peak_corner_excursion`."""
    corner_offsets = corners[None, :].commutator(mode_twists[:, None])           # [3, 4] Point-
    return peak_corner_excursion / corner_offsets.antinorm().max(axis=1)


def panel_layers(
    camera: Camera,
    reference: np.ndarray,
    corners: Point,
    anchors: Point,
    attachments: Point,
    twist: Twist,
) -> np.ndarray:
    """Displaced plate and spring layers composite over the dotted equilibrium `reference` outline for `twist` (`[] Twist+`)."""
    displaced_corners = corners + corners.commutator(twist)                      # [4] Point-
    displaced_attachments = attachments + attachments.commutator(twist)          # [...] Point-
    return drawing.composited([reference, *suspension_layers(camera, displaced_corners, anchors, displaced_attachments)])


def mode_figure(corners: Point, anchors: Point, attachments: Point, mode_twists: Twist) -> plt.Figure:
    """Three side-by-side panels show the peak mode shapes against the dotted equilibrium plate."""
    fig, axes = mode_canvas(corners, anchors, dpi=75)
    rim = plate_surface(len(corners)).border
    scaled_twists = excursion_scales(corners, mode_twists) * mode_twists         # [3] Twist+
    for mode_index, ax in enumerate(axes):
        camera = panel(ax, mv.x, mv.y, mv.w)
        reference = simplices.strokes(corners, rim, ghost_colour, ghost_width, camera, dashes=ghost_dots)
        overlay(ax, [panel_layers(camera, reference, corners, anchors, attachments, scaled_twists[mode_index])])
    return fig


def mode_frames(
    corners: Point,
    anchors: Point,
    attachments: Point,
    mode_twists: Twist,
    states: Iterable[Twist],
) -> Iterator[np.ndarray]:
    """Yields RGB animation frames for a stream of three-mode twist `states` (`[3] Twist+`)."""
    fig, axes = mode_canvas(corners, anchors, dpi=67)
    rim = plate_surface(len(corners)).border
    scales = excursion_scales(corners, mode_twists)                              # [3] Scalar+
    cameras = [panel(ax, mv.x, mv.y, mv.w) for ax in axes]
    references = [
        simplices.strokes(corners, rim, ghost_colour, ghost_width, camera, dashes=ghost_dots)
        for camera in cameras
    ]
    initial_twists = scales * mode_twists                                        # [3] Twist+
    images = [
        overlay(ax, [panel_layers(camera, reference, corners, anchors, attachments, initial_twists[mode_index])])
        for mode_index, (ax, camera, reference) in enumerate(zip(axes, cameras, references))
    ]
    for state in states:
        scaled_twists = scales * state                                           # [3] Twist+
        for mode_index, (image, camera, reference) in enumerate(zip(images, cameras, references)):
            image.set_data(panel_layers(camera, reference, corners, anchors, attachments, scaled_twists[mode_index]))
        yield capture(fig)
    plt.close(fig)


def setup_figure(corners: Point, anchors: Point, attachments: Point) -> plt.Figure:
    """Equilibrium layers show the suspended plate, its springs, and their anchors in a single panel."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120, facecolor="white")
    low, high = bounding_box([corners, anchors])
    lower, upper = low - 0.35, high + 0.35
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    overlay(ax, suspension_layers(panel(ax, mv.x, mv.y, mv.w), corners, anchors, attachments))
    return fig
