"""Drawing for developability: flat-shaded surfaces, each face lit by its own normal, so that the
patches read as flat planes and the creases between them as sharp changes of shade."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.surfaces.developability import core

# The surface's colour, the light's direction and the share of light that falls everywhere, the views'
# half extent, and the view as (elevation, azimuth) in degrees.
SURFACE_COLOUR = np.array([0.90, 0.89, 0.86])
LIGHT_DIRECTION, AMBIENT = np.array([-0.4, -0.7, 0.6]), 0.25
REACH = 1.15
ELEVATION, AZIMUTH = 24.0, -36.0


def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x, y and z coordinates of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def draw_surface(ax, mesh: core.Mesh, vertices: core.Vector) -> None:
    """One surface, every face shaded flat by the light falling on it."""
    triangles = coordinates(vertices)[mesh.faces]                              # [F, 3, 3]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    light = AMBIENT + (1 - AMBIENT) * np.clip(normals @ (LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)), 0, 1)
    colours = light[:, None] * SURFACE_COLOUR
    # Faces filled without antialiasing meet without hairline gaps; the figure's resolution, or the
    # animation's supersampling, smooths the outline.
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colours, linewidths=0, antialiased=False))
    ax.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), zlim=(-REACH, REACH))
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def draw_before_after(flowed: list[tuple[core.Mesh, list[tuple[core.Vector, core.Scalar]]]]) -> plt.Figure:
    """Each shape before its flow and after it, side by side."""
    figure = plt.figure(figsize=(3.4 * 2 * len(flowed), 3.4), dpi=220)
    for case, (mesh, states) in enumerate(flowed):
        for column, (vertices, _) in enumerate((states[0], states[-1])):
            draw_surface(figure.add_subplot(1, 2 * len(flowed), 2 * case + column + 1, projection="3d"), mesh, vertices)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate_flow(mesh: core.Mesh, states: list[tuple[core.Vector, core.Scalar]]) -> list[np.ndarray]:
    """Frames of a surface flowing towards developable patches."""
    images = []
    for vertices, _ in states:
        figure = plt.figure(figsize=(4.0, 4.0), dpi=90)
        draw_surface(figure.add_axes((0, 0, 1, 1), projection="3d"), mesh, vertices)
        images.append(capture(figure))
        plt.close(figure)
    return images
