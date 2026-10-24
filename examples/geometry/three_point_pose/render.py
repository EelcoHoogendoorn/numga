"""Marker triangles before, during and after geometric pose reconstruction, drawn on the GPU: the placed triangle
translucent with solid edges, the measured markers as balls joined by a dashed outline."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

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
    values = points.cast(core.ga.subspace("wzy wxz wyx xyz")).kernel
    return values[..., :3] / values[..., 3:]


def framing(points: list[core.Point]) -> solids.View:
    """One view of every triangle given, about its markers."""
    return solids.view(points, 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def dashes(corners: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """The starts and ends of dashes along a closed outline `[corners, 3]`, a dash and a gap to each period."""
    pieces = []
    for start, end in zip(corners[:-1], corners[1:]):
        count = max(1, round(np.linalg.norm(end - start) / period))
        fractions = np.arange(count)[:, None] / count
        pieces.append((start + fractions * (end - start), start + (fractions + 0.5 / count) * (end - start)))
    return np.concatenate([starts for starts, _ in pieces]), np.concatenate([ends for _, ends in pieces])


def markers(target: core.Point, view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The measured markers, joined by a dashed outline, as solids."""
    starts, ends = dashes(coordinates(target)[ORDER], DASH_PERIOD * view.pixel)
    return [(solids.ball(target, MARKER_RADIUS * view.pixel)[:, None], TARGET),
            (solids.rods(solids.mv.point(starts), solids.mv.point(ends), DASH_WIDTH / 2 * view.pixel), TARGET)]


def triangle(placed: core.Point, view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]],
                                                             list[tuple[solids.Quadric, np.ndarray, float]]]:
    """The placed triangle's solid edges, and the triangle itself, translucent."""
    rods, joints = solids.edges(placed[ORDER[:-1]], placed[ORDER[1:]], EDGE_WIDTH / 2 * view.pixel)
    # The triangle as a plate a pixel thick.
    plate = solids.plates(placed[None], view.pixel)                          # [1, 5] Plane
    return [(rods, COLOUR), (joints, COLOUR)], [(plate, COLOUR, TRIANGLE_OPACITY)]


def scene(target: core.Point, placed: core.Point, view: solids.View) -> np.ndarray:
    """The measured markers, joined by a dashed outline, and the placed triangle, translucent with solid edges,
    `[rows, columns, 3]`."""
    edges, translucent = triangle(placed, view)
    return solids.draw(markers(target, view) + edges, view, np.ones(3), inks=[], translucent=translucent, meshes=[])


def draw_pose(source: core.Point, target: core.Point, placed: core.Point) -> plt.Figure:
    """The placed triangle over the measured markers, framed with the markers' places on the object too, so that
    every stage is seen alike."""
    return image_figure(scene(target, placed, framing([source, target, placed])))


def animate(states: Iterator[core.Point], source: core.Point, target: core.Point,
            room: float = VIEW_ROOM) -> Iterator[np.ndarray]:
    """The triangle carried through each alignment over the measured markers, which are set up once, all from one
    view framing the triangle's start and the markers, `room` times over."""
    framed = framing([source, target])
    view = framed._replace(half=framed.half * room)
    measured = markers(target, view)
    for state in states:
        edges, translucent = triangle(state, view)
        yield solids.draw(measured + edges, view, np.ones(3), inks=[], translucent=translucent, meshes=[])
