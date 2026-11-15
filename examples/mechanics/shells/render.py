"""GPU rendering shades discrete shell triangles by their flat face patches, framing settled shells
side by side and streaming relaxation trajectories frame by frame."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.shells import core
from examples.util.panels import show_pixels
from numga.render import meshes, solids

colour = np.array([0.82, 0.62, 0.45])
gloss = 0.08
elevation, azimuth = np.radians([24.0, 40.0])
margin = 0.1
room = 0.08
n_pixels = 240
background = np.ones(3)
frame_duration_ms = 40


def centre(points: core.Point) -> core.Point:
    """Mean vertex position (`[] Point-`) centres the camera on the shell."""
    return points.batch().mean(axis=0)


def extent(points: core.Point) -> np.ndarray:
    """Square half-extents (`[2]`) frame `points` (`Point-[V]`) around its centroid."""
    places = points.batch()                                                      # [V] Point-
    framed = solids.view([places - centre(points) + solids.origin], 0.0, elevation, azimuth, margin, n_pixels)
    return np.full(2, framed.half.max())


def picture(faces: np.ndarray, points: core.Point, half: np.ndarray) -> np.ndarray:
    """Flat-shaded triangles (`[F, 3]`) at `points` (`Point-[V]`) rasterize into an RGB frame of
    shape `[n_pixels, n_pixels, 3]`."""
    places = points.batch()                                                      # [V] Point-
    view = solids.View(solids.aim(centre(points), elevation, azimuth), half, n_pixels)
    return meshes.draw(
        places, faces, colour, meshes.flat(places, faces), view, background,
        face_values=np.zeros(4), colour=meshes.plain, gloss=gloss,
    )


def draw_settled(shells: list[tuple[np.ndarray, core.Point]]) -> plt.Figure:
    """Side-by-side panels display each settled shell's triangles (`[F, 3]`) and vertices
    (`Point-[V]`)."""
    n_shells = len(shells)
    figure, axes = plt.subplots(
        1, n_shells, figsize=(n_pixels * n_shells / 100, n_pixels / 100), dpi=100, squeeze=False,
    )
    for ax, (faces, points) in zip(axes[0], shells):
        show_pixels(ax, picture(faces, points, extent(points)))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate(faces: np.ndarray, frames: Iterable[core.Point]) -> Iterator[np.ndarray]:
    """Yields RGB frames (`[n_pixels, n_pixels, 3]`) of `faces` (`[F, 3]`) along `frames`, locked to
    the initial shape's extent plus `room`."""
    frames = iter(frames)
    first = next(frames)                                                         # Point-[V]
    half = extent(first) * (1 + room)                                            # [2]
    for points in chain([first], frames):
        yield picture(faces, points, half)
