"""The laminate midsurfaces under a growing pull, side by side in one scene, drawn on the GPU: the strips flat-shaded
meshes, the top ply's fibres strokes over them."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from contourpy import contour_generator
from matplotlib.colors import to_rgb

from examples.mechanics.composite_strip import core
from examples.util.wgsl import meshes, solids, strokes

sm = solids.mv
# The pixels across the frames; the view's elevation and azimuth in radians, azimuth zero looking along +y, and
# the margin around the strips as a share of the larger side; the strip's colour and its fibres', and the fibres'
# width in pixels.
N_PIXELS = 300
AZIMUTH, ELEVATION = np.deg2rad(-74.0), np.deg2rad(22.0)
MARGIN = 0.1
FACE = np.array(to_rgb("#c6a15b"))
FIBRE = np.array(to_rgb("#4a3a1f"))
FIBRE_WIDTH = 0.8
background = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def gpu_points(points: core.Direction) -> solids.Point:
    """The GPU library's own points at the places of the strip's: their coordinates, read by pairing with the planes
    x, y and z, are the one hand-off from this algebra to the renderer's."""
    return sm.point((stack((core.mv.x, core.mv.y, core.mv.z)) & points[..., None]).to_array())


def level_lines(field: np.ndarray, spacing: float) -> list[np.ndarray]:
    """The level lines of a sampled field `[rows, columns]` at every multiple of `spacing` in its range,
    as `[vertices, 2]` arrays of fractional (row, column) sample positions; a vertex where a line passes through a
    sample twice over is kept once, so that every step along a line has a direction."""
    generator = contour_generator(z=field, line_type="Separate")
    levels = spacing * np.arange(np.ceil(field.min() / spacing), np.floor(field.max() / spacing) + 1)
    lines = [line[:, ::-1] for level in levels for line in generator.lines(level)]
    return [line[np.concatenate([[True], np.abs(np.diff(line, axis=0)).max(axis=-1) > 1e-9])] for line in lines]


def grid_faces(shape: tuple[int, ...]) -> np.ndarray:
    """The triangles of grids `[..., rows, columns]` of points, flattened in that order: two to each cell, as indices
    `[faces, 3]`."""
    index = np.arange(np.prod(shape)).reshape(shape)
    corners = np.stack([index[..., :-1, :-1], index[..., 1:, :-1], index[..., 1:, 1:], index[..., :-1, 1:]], axis=-1)
    return corners[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)


def animate_strips(rest: core.Direction, frames: core.Direction, across_fibres: core.Scalar, spacing: float) -> Iterator[np.ndarray]:
    """The strips deformed at each frame, `[frames, cases, length samples, width samples]`, seen from
    behind the ends that stay in place: flat-shaded meshes with the top ply's fibres over them, the level lines
    of `across_fibres`, `[cases, length samples, width samples]`, `spacing` apart, carried on each deformed
    surface. `rest` sets the common framing."""
    view = solids.view([gpu_points(rest.reshape(-1)), gpu_points(frames.reshape(-1))], 0.0, ELEVATION, AZIMUTH,
                       MARGIN, N_PIXELS)
    faces = grid_faces(frames.shape[1:])                               # [faces, 3]
    # Each fibre line as fractional sample positions on its own strip's grid, all lines one after another.
    lines = [(case, line) for case, field in enumerate(across_fibres.to_array()) for line in level_lines(field, spacing)]
    cases = np.concatenate([np.full(len(line), case) for case, line in lines])   # [fibre vertices]
    positions = np.concatenate([line for _, line in lines])            # [fibre vertices, 2]
    # The cell of the grid each vertex lies in, by its first sample, and the vertex's shares along and across it.
    first = np.minimum(np.floor(positions).astype(int), np.array(frames.shape[2:]) - 2)   # [fibre vertices, 2]
    along, across = (positions - first).T                              # [fibre vertices], [fibre vertices]
    rows, columns = first.T
    bounds = np.cumsum([0] + [len(line) for _, line in lines])        # [lines + 1]

    for grids in frames:
        points = gpu_points(grids.reshape(-1))                         # [vertices] Point-
        strips = meshes.samples(points, faces, FACE, meshes.flat(points, faces), view, face_values=np.zeros(4),
                                colour=meshes.plain, gloss=solids.gloss)
        # The deformed surface at each fibre vertex: the four corners of its cell blended by its shares, linear
        # along each side of the cell, as the surface's samples are.
        threads = ((grids[cases, rows, columns] * (1 - across) + grids[cases, rows, columns + 1] * across) * (1 - along)
                   + (grids[cases, rows + 1, columns] * (1 - across) + grids[cases, rows + 1, columns + 1] * across)
                   * along)                                            # [fibre vertices] Direction-
        drawn = gpu_points(threads)                                    # [fibre vertices] Point-
        fibres = strokes.polylines([(drawn[start:end], FIBRE) for start, end in zip(bounds[:-1], bounds[1:])],
                                   FIBRE_WIDTH, view)
        yield solids.draw([], view, background, inks=[], translucent=[], meshes=[(strips, 1.0), (fibres, 1.0)])
