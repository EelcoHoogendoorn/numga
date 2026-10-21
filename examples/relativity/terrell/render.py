"""Drawing for Terrell rotation: photographs of a passing sphere and cube, one lane per speed, on a
stereographic map of the camera's sky, which draws circles on the sky as circles."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.animation import capture
from examples.relativity.terrell import core

mv = core.mv
# The colours of the cube's faces, in the order of its turns: top, bottom, the face ahead, the face
# behind, and the faces towards and away from the camera. The sphere's fill, grid and outline, and the
# edges of the faces.
face_colours = ["#fde68a", "#a8a29e", "#fca5a5", "#93c5fd", "#c4b5fd", "#86efac"]
sphere_colour = "#e0f2fe"
grid_colour = "#0369a1"
outline_colour = "#0c4a6e"
edge_colour = "#334155"
# The free space between the lanes and around the pictures, in the map's units.
gap = 0.06


# --- plumbing -------------------------------------------------------------------------
def sky(rays: core.Antivector) -> np.ndarray:
    """Where rays arriving at the camera appear on the stereographic map of its sky, centred on the line
    of sight square to the track, on the last axis: right along the motion, and up."""
    t, x, y, z = np.moveaxis((stack((mv.t, mv.x, mv.y, mv.z)) & rays[..., None]).to_array(), -1, 0)
    # A ray's source is seen against its direction of travel; the map projects that direction from the
    # point of the sky straight behind the camera.
    return np.stack([-x, -z], axis=-1) / (t - y)[..., None]


def animate_photographs(frames: Iterable[tuple[core.Antivector, core.Scalar, core.Antivector, core.Scalar,
                                               core.Antivector]]) -> list[np.ndarray]:
    """Each frame's photographs in every case `[cases]`: the cube's faces turned towards the camera, the
    sphere's outline, and the part of its grid on the camera's side."""
    frames = [(sky(faces), facing.to_array() > 0, sky(grid), seen.to_array() > 0, sky(outline))
              for faces, facing, grid, seen, outline in frames]
    pictures = np.concatenate([np.concatenate([faces.reshape(-1, 2), outline.reshape(-1, 2)])
                               for faces, _, _, _, outline in frames])        # [samples, 2]
    count = frames[0][0].shape[0]
    low, high = pictures.min(axis=0) - gap, pictures.max(axis=0) + gap
    lane = high[1] - low[1]
    images = []
    for faces, facing, grid, seen, outline in frames:
        figure = plt.figure(figsize=(9.0, 9.0 * lane * count / (high[0] - low[0])), dpi=100)
        ax = figure.add_axes((0, 0, 1, 1))
        for case in range(count):
            shift = np.array([0.0, -lane * case])
            for face in np.flatnonzero(facing[case]):
                ax.fill(*(faces[case, face] + shift).T, facecolor=face_colours[face], edgecolor=edge_colour, linewidth=1.2)
            ax.fill(*(outline[case] + shift).T, facecolor=sphere_colour, edgecolor="none")
            lines = np.where(seen[case][..., None], grid[case], np.nan) + shift   # [lines, grid, 2]
            for line in lines:
                ax.plot(*line.T, color=grid_colour, linewidth=0.8)
            ax.plot(*(outline[case] + shift).T, color=outline_colour, linewidth=1.4)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1] - lane * (count - 1), high[1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        images.append(capture(figure))
        plt.close(figure)
    return images
