"""Drawing for discrete shells: the triangles of a sheet lit by their normals, the faces of a settled sheet
side by side, and a sheet relaxing frame by frame."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.mechanics.shells import core

# The sheet's colour, the light's direction and the share of light that falls everywhere, the view as
# (elevation, azimuth) in degrees, the margin around a sheet, and the duration of a frame.
COLOUR = np.array([0.82, 0.62, 0.45])
LIGHT_DIRECTION, AMBIENT = np.array([-0.3, -0.5, 0.8]), 0.35
VIEW = (24.0, -50.0)
MARGIN = 1.1
DURATION_MS = 50


def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points `[..., 3]`, at unit weight."""
    places = points.cast(core.ga.subspace("wzy wxz wyx xyz")).kernel
    return places[..., :3] / places[..., 3:]


def draw_sheet(ax, faces: np.ndarray, points: core.Point, reach: float) -> None:
    """A sheet's triangles `[F, 3]` at the given points `Point[V]`, lit on either side, centred in a cube of
    the given half width."""
    places = coordinates(points.batch())
    places = places - places.mean(axis=0)
    triangles = places[faces]                                                    # [F, 3, 3]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    light = AMBIENT + (1 - AMBIENT) * np.abs(normals @ (LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)))
    colours = light[:, None] * COLOUR
    # Faces filled without antialiasing meet without hairline gaps.
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colours, linewidths=0, antialiased=False))
    ax.set(xlim=(-reach, reach), ylim=(-reach, reach), zlim=(-reach, reach))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(*VIEW)
    ax.set_axis_off()


def extent(points: core.Point) -> float:
    """The half width of a cube around the centred sheet."""
    places = coordinates(points.batch())
    return MARGIN * np.abs(places - places.mean(axis=0)).max()


def draw_settled(sheets: list[tuple[np.ndarray, core.Point]]) -> plt.Figure:
    """Settled sheets of faces and points side by side."""
    figure = plt.figure(figsize=(4.5 * len(sheets), 4.5), dpi=110)
    for case, (faces, points) in enumerate(sheets):
        draw_sheet(figure.add_subplot(1, len(sheets), case + 1, projection="3d"), faces, points, extent(points))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate(faces: np.ndarray, frames: list[core.Point]) -> list[np.ndarray]:
    """Frames of a sheet relaxing, all in the cube of its settled shape."""
    reach = extent(frames[-1])
    images = []
    for points in frames:
        figure = plt.figure(figsize=(4.5, 4.5), dpi=90)
        draw_sheet(figure.add_axes((0, 0, 1, 1), projection="3d"), faces, points, reach)
        images.append(capture(figure))
        plt.close(figure)
    return images
