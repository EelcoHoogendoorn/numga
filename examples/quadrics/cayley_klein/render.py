"""GPU rendering of Cayley-Klein level curves, planes, lines, and vertex angle labels."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.cayley_klein.core import Line, Plane, Point, Polarity, Scalar, mv
from examples.util import charts
from examples.util.panels import overlay, panel
from numga import stack
from numga.render import Camera, quadrics, read, simplices
from numga.render.drawing import Draw
from numga.topology import Topology

view_box = (-1.2, 1.9, -1.2, 1.6)
# Convert point sizes at 120 dpi to screen pixels for marker areas and stroke widths.
pixels_per_point = 120 / 72
outline_width = 1.0
frame_width = 1.2


def chart_coordinates(points: Point) -> np.ndarray:
    """Euclidean chart coordinates (`[..., 2]`) read by pairing unit-weighted `points` with `x` and `y`."""
    flat_points = points.batch()                                           # [...] Point-
    return ((stack((mv.x, mv.y)) & flat_points[..., None]) / (mv.w & flat_points)[..., None]).to_array()


def rgb_colour(colour: str) -> np.ndarray:
    """Linear RGB triple (`[3]`) parsed from a matplotlib colour string."""
    return np.array(mcolors.to_rgb(colour))


def panel_cameras(ax: plt.Axes) -> tuple[Camera, Camera]:
    """Point-reading plane camera and meet-reading line camera sharing `ax`'s view box."""
    ax.set(xlim=view_box[:2], ylim=view_box[2:])
    origin = mv.w.dual()                                                   # [] Point-
    # Coordinate lines through the origin and the line at infinity pair with meets of planes.
    line_camera = panel(ax, mv.y.dual() & origin, origin & mv.x.dual(), mv.x.dual() & mv.y.dual())
    return panel(ax, mv.x, mv.y, mv.w), line_camera


def disc_marks(points: Point, colour: str, area: float, camera: Camera) -> Draw:
    """Filled circular markers of `area` square points drawn in `colour`."""
    radius = (np.sqrt(area) / 2 + outline_width / 2) * pixels_per_point
    return simplices.markers(points.batch().reshape(-1), rgb_colour(colour), radius, camera)


def square_marks(points: Point, colour: str, area: float, turn_angle: float, camera: Camera) -> Draw:
    """Outlined four-corner markers of `area` square points turned by `turn_angle` from screen horizontal."""
    screen_step = camera.across.dual() / read(camera.across, camera.across.dual())  # [] Point-
    turns = (mv.xy * (-(turn_angle + np.pi / 2 * np.arange(4)) / 2)).exp()          # [4] Even+
    half_diagonal = np.sqrt(2 * area) / 2 * pixels_per_point * camera.pixel
    corner_offsets = (turns >> screen_step) * half_diagonal                         # [4] Point-
    flat_points = points.batch().reshape(-1)                                        # [n_points] Point-
    n_points = len(flat_points)
    places = (flat_points / (mv.w & flat_points))[:, None] + corner_offsets         # [n_points, 4] Point-
    return simplices.strokes(
        places.reshape(-1), Topology.loops([4] * n_points), rgb_colour(colour), frame_width * pixels_per_point, camera,
    )


def polarity_curves(polarities: Polarity, colour: str, width: float, camera: Camera) -> Draw:
    """Implicit null curves `polarity(P) & P == 0` drawn `width` points wide."""
    return quadrics.curves(polarities.batch().reshape(-1), width * pixels_per_point, rgb_colour(colour), camera)


@charts.style()
def hyperbolic_figure(
    absolute: Polarity,
    vertices: Point,
    sides: Plane,
    angles: Scalar,
    probe: Point,
    foot: Point,
    normal: Line,
    reflected_point: Point,
    pole: Point,
    sphere_quadrics: Polarity,
    centres: Point,
) -> plt.Figure:
    """Two-panel Beltrami-Klein figure showing the hyperbolic triangle, perpendicular, and reflection on
    the left and two families of concentric hyperbolic spheres on the right."""
    fig, (left, right) = charts.subplots(780, 360, ncols=2)
    for ax in (left, right):
        ax.set_axis_off()
    fig.tight_layout()
    plane_camera, line_camera = panel_cameras(left)
    overlay(
        left,
        [
            polarity_curves(absolute[None], "black", 1.2, plane_camera),
            quadrics.flats(sides.batch().reshape(-1), 1.2 * pixels_per_point, rgb_colour("#2563eb"), plane_camera),
            quadrics.flats(normal.batch().reshape(-1), 1.0 * pixels_per_point, rgb_colour("#dc2626"), line_camera),
            disc_marks(vertices, "#1d4ed8", 14, plane_camera),
            disc_marks(probe[None], "#dc2626", 18, plane_camera),
            square_marks(foot, "#b91c1c", 14, np.pi / 4, plane_camera),
            disc_marks(reflected_point, "#f97316", 16, plane_camera),
            square_marks(pole, "#a855f7", 16, 0.0, plane_camera),
        ],
    )
    for xy, theta in zip(chart_coordinates(vertices), angles.batch().reshape(-1).to_array()):
        left.annotate(f"{np.degrees(theta):.1f}°", xy, textcoords="offset points", xytext=(3, 3))

    plane_camera, _ = panel_cameras(right)
    overlay(
        right,
        [polarity_curves(absolute[None], "black", 1.2, plane_camera)]
        + [
            polarity_curves(family, colour, 0.8, plane_camera)
            for family, colour in zip(sphere_quadrics, ("#3b82f6", "#f97316"))
        ]
        + [disc_marks(centres, "#111827", 13, plane_camera)],
    )
    return fig
