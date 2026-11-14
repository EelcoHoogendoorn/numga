"""Flat-shaded GPU rendering draws the laminate midsurfaces side by side under a growing pull,
with the top ply's fibre trajectories stroked over each surface."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from contourpy import contour_generator
from matplotlib.colors import to_rgb

from examples.mechanics.composite_strip import core
from numga.render import meshes, simplices, solids
from numga.topology import Topology

n_pixels = 300
azimuth, elevation = np.deg2rad(-74.0), np.deg2rad(22.0)
margin = 0.1
face_colour = np.array(to_rgb("#c6a15b"))
fibre_colour = np.array(to_rgb("#4a3a1f"))
fibre_width = 2.0
background = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def level_lines(field: np.ndarray, spacing: float) -> list[np.ndarray]:
    """Level contours of `field` (`[n_rows, n_columns]`) at multiples of `spacing` form fractional
    `(row, column)` vertex polylines (`[..., 2]`), deduplicating repeated sample crossings so every
    edge step has a non-zero direction."""
    generator = contour_generator(z=field, line_type="Separate")
    levels = spacing * np.arange(np.ceil(field.min() / spacing), np.floor(field.max() / spacing) + 1)
    lines = [line[:, ::-1] for level in levels for line in generator.lines(level)]
    return [line[np.concatenate([[True], np.abs(np.diff(line, axis=0)).max(axis=-1) > 1e-9])] for line in lines]


def grid_faces(shape: tuple[int, ...]) -> np.ndarray:
    """Each quad of a flattened `[..., n_rows, n_columns]` point grid splits into two triangles,
    giving triangle vertex indices of shape `[F, 3]`."""
    index = np.arange(np.prod(shape)).reshape(shape)
    corners = np.stack([index[..., :-1, :-1], index[..., 1:, :-1], index[..., 1:, 1:], index[..., :-1, 1:]], axis=-1)
    return corners[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)


def animate_strips(rest: core.Direction, bounds: core.Direction, across_fibres: core.Scalar, spacing: float,
                   states: Iterable[core.Direction]) -> Iterator[np.ndarray]:
    """Yields `[..., ..., 3]` RGB frames (`n_pixels` along the wider side) of the side-by-side strips
    in `states`, seen from behind the fixed ends with the top ply's fibres stroked along level
    contours of `across_fibres` spaced `spacing` apart; `rest` and `bounds` fix the camera framing."""
    resting, peak = (solids.origin + solids.context.include_from_algebra(places) for places in (rest, bounds))
    view = solids.view([resting.reshape(-1), peak.reshape(-1)], 0.0, elevation, azimuth, margin, n_pixels)
    faces = grid_faces(rest.shape)                                             # [F, 3]
    # Each fibre line gives fractional sample coordinates on its own strip's grid, concatenated into one chain set.
    lines = [(case, line) for case, field in enumerate(across_fibres.to_array()) for line in level_lines(field, spacing)]
    cases = np.concatenate([np.full(len(line), case) for case, line in lines])  # [...]
    positions = np.concatenate([line for _, line in lines])                    # [..., 2]
    # Locate the grid quad containing each fibre vertex and its bilinear shares along and across that quad.
    lower_indices = np.minimum(np.floor(positions).astype(int), np.array(rest.shape[1:]) - 2)  # [..., 2]
    length_share, width_share = (positions - lower_indices).T                  # [...], [...]
    rows, columns = lower_indices.T
    fibre_chains = Topology.chains([len(line) for _, line in lines])

    for grids in states:
        placed = solids.origin + solids.context.include_from_algebra(grids)    # [...] Point-
        points = placed.reshape(-1)                                            # [...] Point-
        strips = meshes.samples(points, faces, face_colour, meshes.flat(points, faces), view, face_values=np.zeros(4),
                                colour=meshes.plain, gloss=solids.gloss)
        # Bilinearly blend the four corners of each quad so every fibre stroke rides on the deformed surface.
        threads = ((placed[cases, rows, columns] * (1 - width_share) + placed[cases, rows, columns + 1] * width_share)
                   * (1 - length_share)
                   + (placed[cases, rows + 1, columns] * (1 - width_share)
                      + placed[cases, rows + 1, columns + 1] * width_share) * length_share)  # [...] Point-
        fibres = simplices.tubes(threads, fibre_chains, fibre_colour, fibre_width, view, gloss=solids.gloss)
        yield solids.draw([], view, background, inks=[], translucent=[], meshes=[(strips, 1.0), (fibres, 1.0)])
