"""The laminate midsurfaces under a growing pull, side by side in one scene, drawn on the GPU: the strips flat-shaded
meshes, the top ply's fibres strokes over them."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from contourpy import contour_generator
from matplotlib.colors import to_rgb
from scipy.ndimage import map_coordinates

from examples.mechanics.composite_strip import core
from examples.util.wgsl import meshes, solids, strokes

sm = solids.mv
# The pixels across the frames; the view's elevation and azimuth in radians, azimuth zero looking along +y, and
# the margin around the strips as a share of the larger side; the strip's colour and its fibres', and the fibres'
# width in pixels.
N_PIXELS = 560
AZIMUTH, ELEVATION = np.deg2rad(-74.0), np.deg2rad(22.0)
MARGIN = 0.1
FACE = np.array(to_rgb("#c6a15b"))
FIBRE = np.array(to_rgb("#4a3a1f"))
FIBRE_WIDTH = 1.4
background = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def level_lines(field: np.ndarray, spacing: float) -> list[np.ndarray]:
    """The level lines of a sampled field `[rows, columns]` at every multiple of `spacing` in its range,
    as `[vertices, 2]` arrays of fractional (row, column) sample positions; a vertex where a line passes through a
    sample twice over is kept once, so that every step along a line has a direction."""
    generator = contour_generator(z=field, line_type="Separate")
    levels = spacing * np.arange(np.ceil(field.min() / spacing), np.floor(field.max() / spacing) + 1)
    lines = [line[:, ::-1] for level in levels for line in generator.lines(level)]
    return [line[np.concatenate([[True], np.linalg.norm(np.diff(line, axis=0), axis=-1) > 1e-9])] for line in lines]


def grid_faces(shape: tuple[int, ...]) -> np.ndarray:
    """The triangles of grids `[..., rows, columns]` of points, flattened in that order: two to each cell, as indices
    `[faces, 3]`."""
    index = np.arange(np.prod(shape)).reshape(shape)
    corners = np.stack([index[..., :-1, :-1], index[..., 1:, :-1], index[..., 1:, 1:], index[..., :-1, 1:]], axis=-1)
    return corners[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)


def animate_strips(rest: core.Vector, frames: core.Vector, across_fibres: core.Scalar, spacing: float) -> Iterator[np.ndarray]:
    """The strips deformed at each frame, `[frames, cases, length samples, width samples]`, seen from
    behind the ends that stay in place: flat-shaded meshes with the top ply's fibres over them, the level lines
    of `across_fibres`, `[cases, length samples, width samples]`, `spacing` apart, carried on each deformed
    surface. `rest` sets the common framing."""
    rests = rest.cast(core.ga.subspace("x y z")).kernel                # [cases, length samples, width samples, 3]
    deformed = frames.cast(core.ga.subspace("x y z")).kernel           # [frames, cases, length samples, width samples, 3]
    view = solids.view([sm.point(rests.reshape(-1, 3)), sm.point(deformed.reshape(-1, 3))], 0.0, ELEVATION, AZIMUTH,
                       MARGIN, N_PIXELS)
    faces = grid_faces(deformed.shape[1:-1])                           # [faces, 3]
    # Each fibre line as fractional sample positions on its own strip's grid.
    lines = [(case, line) for case, field in enumerate(across_fibres.to_array())
             for line in level_lines(field, spacing)]

    for grids in deformed:
        points = sm.point(grids.reshape(-1, 3))                        # [vertices] Point
        strips = meshes.samples(points, faces, FACE, meshes.flat(points, faces), view)
        # The deformed surface, interpolated linearly between its samples at each fibre line's positions.
        threads = [np.stack([map_coordinates(grids[case][..., axis], line.T, order=1, mode="nearest") for axis in range(3)], axis=-1)
                   for case, line in lines]                            # [lines] of [vertices, 3]
        fibres = strokes.polylines([(thread, FIBRE) for thread in threads], FIBRE_WIDTH, view)
        yield solids.draw([], view, background, meshes=[(strips, 1.0), (fibres, 1.0)])
