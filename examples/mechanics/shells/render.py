"""Drawing for discrete shells, on the GPU: the triangles of a sheet lit by their normals, the faces of a settled
sheet side by side, and a sheet relaxing frame by frame."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.shells import core
from examples.util.wgsl import meshes, show_pixels, solids

# The sheet's colour and its slight sheen, the view, elevation and azimuth in radians, azimuth zero looking along
# +y, the margin around a sheet, the room an animation's view leaves around the sheet's first shape for it to spread
# into, as a share of its size, the pixels across a picture, and the duration of a frame.
colour = np.array([0.82, 0.62, 0.45])
gloss = 0.08
elevation, azimuth = np.radians([24.0, 40.0])
margin = 0.1
room = 0.08
n_pixels = 480
background = np.ones(3)
DURATION_MS = 50


def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points `[..., 3]`, at unit weight."""
    places = points.cast(core.ga.subspace("wzy wxz wyx xyz")).kernel
    return places[..., :3] / places[..., 3:]


def centre(points: core.Point) -> core.Point:
    """The sheet's mean point."""
    return points.batch().mean(axis=0)


def extent(points: core.Point) -> np.ndarray:
    """Half the width and height of a view of the sheet about its mean, square."""
    places = points.batch()
    framed = solids.view([places - centre(points) + solids.origin], 0.0, elevation, azimuth, margin, n_pixels)
    return np.full(2, framed.half.max())


def picture(faces: np.ndarray, points: core.Point, half: np.ndarray) -> np.ndarray:
    """A sheet's triangles `[F, 3]` at the given points `Point[V]`, flat, lit on the side the camera sees, seen
    about its mean."""
    places = points.batch()
    view = solids.View(solids.aim(centre(points), elevation, azimuth), half, n_pixels)
    return meshes.draw(places, faces, colour, meshes.flat(places, faces), view, background, gloss=gloss)


def draw_settled(sheets: list[tuple[np.ndarray, core.Point]]) -> plt.Figure:
    """Settled sheets of faces and points side by side."""
    figure, axes = plt.subplots(1, len(sheets), figsize=(n_pixels * len(sheets) / 100, n_pixels / 100), dpi=100)
    for ax, (faces, points) in zip(axes, sheets):
        show_pixels(ax, picture(faces, points, extent(points)))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate(faces: np.ndarray, frames: Iterable[core.Point]) -> Iterator[np.ndarray]:
    """Frames of a sheet relaxing, all to the scale of its first shape, with room to spread."""
    frames = iter(frames)
    first = next(frames)
    half = extent(first) * (1 + room)
    for points in chain([first], frames):
        yield picture(faces, points, half)
