"""GPU rendering for three-point pose reconstruction draws the moving triangle as a translucent plate with solid edges and the measured markers as balls joined by a dashed outline."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from numga import concatenate
from examples.geometry.three_point_pose import core
from examples.util.wgsl import image_figure, solids

colour = np.array(to_rgb("#218c9d"))
target_colour = np.array(to_rgb("#ce713f"))
# How much of what lies behind each of the triangle's two faces hides.
triangle_opacity = 0.26
order = [0, 1, 2, 0]
# The view's elevation and azimuth, in radians, azimuth zero looking along +y; the margin as a share of the larger
# side, and the pixels across a view.
elevation, azimuth = np.radians(55.0), np.radians(-110.0 + 90.0)
margin, n_pixels = 0.28, 600
# How many times over the animation's view frames the triangle's start and the markers, leaving room for the
# triangle's path between them.
view_room = 1.5
# The widths of the triangle's edges and of the target's dashes, the dashes' length with their gaps, and the markers'
# radius, in pixels.
edge_width, dash_width, dash_period, marker_radius = 5.0, 4.5, 18.0, 10.0


# --- plumbing -------------------------------------------------------------------------
def framing(points: list[core.Point]) -> solids.View:
    """The camera view frames every given marker triple."""
    return solids.view(points, 0.0, elevation, azimuth, margin, n_pixels)


def dashes(corners: core.Point, period: float) -> tuple[core.Point, core.Point]:
    """The dashes' starts and ends along a polyline of points, one dash and one gap per period along each side."""
    corners = corners.normalized()                                           # [corners] Point-
    pieces = []
    for start, end in zip(corners[:-1], corners[1:]):
        # A side's length is the size of the line joining its ends.
        count = max(1, round(float((start & end).norm().to_array() / period)))
        fractions = np.arange(count) / count
        pieces.append((start + (end - start) * fractions, start + (end - start) * (fractions + 0.5 / count)))
    return concatenate([starts for starts, _ in pieces]), concatenate([ends for _, ends in pieces])


def markers(target: core.Point, view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The measured markers and their dashed outline are built as solid quadrics."""
    starts, ends = dashes(target[order], dash_period * view.pixel)            # [dashes] Point-, Point-
    return [(solids.ball(target, marker_radius * view.pixel)[:, None], target_colour),
            (solids.rods(starts, ends, dash_width / 2 * view.pixel), target_colour)]


def triangle(current: core.Point, view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]],
                                                              list[tuple[solids.Quadric, np.ndarray, float]]]:
    """The moving triangle's solid edges and translucent plate are built as quadric batches."""
    rods, joints = solids.edges(current[order[:-1]], current[order[1:]], edge_width / 2 * view.pixel)
    # The triangle as a plate a pixel thick.
    plate = solids.plates(current[None], view.pixel)                         # [1, 5] Plane+
    return [(rods, colour), (joints, colour)], [(plate, colour, triangle_opacity)]


def scene(target: core.Point, current: core.Point, view: solids.View) -> np.ndarray:
    """Render the measured markers with their dashed outline and the translucent moving triangle."""
    edges, translucent = triangle(current, view)
    return solids.draw(markers(target, view) + edges, view, np.ones(3), inks=[], translucent=translucent, meshes=[])


def draw_pose(source: core.Point, target: core.Point, current: core.Point) -> plt.Figure:
    """Draw the moving triangle over the measured markers in a shared frame."""
    return image_figure(scene(target, current, framing([source, target, current])))


def animate(states: Iterator[core.Point], source: core.Point, target: core.Point,
            room: float = view_room) -> Iterator[np.ndarray]:
    """Yield rendered frames as the triangle moves through each alignment stage."""
    framed = framing([source, target])
    view = framed._replace(half=framed.half * room)
    measured = markers(target, view)
    for state in states:
        edges, translucent = triangle(state, view)
        yield solids.draw(measured + edges, view, np.ones(3), inks=[], translucent=translucent, meshes=[])
