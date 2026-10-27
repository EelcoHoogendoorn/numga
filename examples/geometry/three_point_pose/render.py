"""GPU rendering for three-point pose reconstruction draws the moving triangle as a translucent plate with solid edges and the measured markers as balls joined by a dashed outline."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from numga import stack
from examples.geometry.three_point_pose import core
from examples.util.wgsl import image_figure, solids

COLOUR = np.array(to_rgb("#218c9d"))
TARGET = np.array(to_rgb("#ce713f"))
# How much of what lies behind each of the triangle's two faces hides.
TRIANGLE_OPACITY = 0.26
ORDER = [0, 1, 2, 0]
# The view's elevation and azimuth, in radians, azimuth zero looking along +y; the margin as a share of the larger
# side, and the pixels across a view.
ELEVATION, AZIMUTH = np.radians(55.0), np.radians(-110.0 + 90.0)
MARGIN, N_PIXELS = 0.28, 600
# How many times over the animation's view frames the triangle's start and the markers, leaving room for the
# triangle's path between them.
VIEW_ROOM = 1.5
# The widths of the triangle's edges and of the target's dashes, the dashes' length with their gaps, and the markers'
# radius, in pixels.
EDGE_WIDTH, DASH_WIDTH, DASH_PERIOD, MARKER_RADIUS = 5.0, 4.5, 18.0, 10.0


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """The Euclidean coordinates of points are their pairings with the coordinate planes after normalization."""
    mv = core.mv
    return (stack((mv.x, mv.y, mv.z)) & points.normalized()[..., None]).to_array()


def framing(points: list[core.Point]) -> solids.View:
    """The camera view frames every given marker triple."""
    return solids.view(points, 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def dashes(corners: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """The dash start and end coordinates trace a closed outline with one dash and one gap per period."""
    pieces = []
    for start, end in zip(corners[:-1], corners[1:]):
        count = max(1, round(np.linalg.norm(end - start) / period))
        fractions = np.arange(count)[:, None] / count
        pieces.append((start + fractions * (end - start), start + (fractions + 0.5 / count) * (end - start)))
    return np.concatenate([starts for starts, _ in pieces]), np.concatenate([ends for _, ends in pieces])


def markers(target: core.Point, view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The measured markers and their dashed outline are built as solid quadrics."""
    starts, ends = dashes(coordinates(target)[ORDER], DASH_PERIOD * view.pixel)
    return [(solids.ball(target, MARKER_RADIUS * view.pixel)[:, None], TARGET),
            (solids.rods(solids.mv.point(starts), solids.mv.point(ends), DASH_WIDTH / 2 * view.pixel), TARGET)]


def triangle(current: core.Point, view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]],
                                                              list[tuple[solids.Quadric, np.ndarray, float]]]:
    """The moving triangle's solid edges and translucent plate are built as quadric batches."""
    rods, joints = solids.edges(current[ORDER[:-1]], current[ORDER[1:]], EDGE_WIDTH / 2 * view.pixel)
    # The triangle as a plate a pixel thick.
    plate = solids.plates(current[None], view.pixel)                         # [1, 5] Plane+
    return [(rods, COLOUR), (joints, COLOUR)], [(plate, COLOUR, TRIANGLE_OPACITY)]


def scene(target: core.Point, current: core.Point, view: solids.View) -> np.ndarray:
    """Render the measured markers with their dashed outline and the translucent moving triangle."""
    edges, translucent = triangle(current, view)
    return solids.draw(markers(target, view) + edges, view, np.ones(3), inks=[], translucent=translucent, meshes=[])


def draw_pose(source: core.Point, target: core.Point, current: core.Point) -> plt.Figure:
    """Draw the moving triangle over the measured markers in a shared frame."""
    return image_figure(scene(target, current, framing([source, target, current])))


def animate(states: Iterator[core.Point], source: core.Point, target: core.Point,
            room: float = VIEW_ROOM) -> Iterator[np.ndarray]:
    """Yield rendered frames as the triangle moves through each alignment stage."""
    framed = framing([source, target])
    view = framed._replace(half=framed.half * room)
    measured = markers(target, view)
    for state in states:
        edges, translucent = triangle(state, view)
        yield solids.draw(measured + edges, view, np.ones(3), inks=[], translucent=translucent, meshes=[])
