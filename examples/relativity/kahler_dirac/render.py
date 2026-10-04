"""Drawing for the field over the whole algebra: each tile a field on the lattice over time, its
planes xt, yt and xy as red, green and blue."""

from __future__ import annotations

import numpy as np

from examples.relativity.kahler_dirac import core

# The planes shown as red, green and blue, and each tile's exposure: the percentile of its amplitudes
# over all its frames that shows at full brightness.
CHANNELS = ("xt", "yt", "xy")
EXPOSURE = 99.3


def colours(grid: core.Grid, space: core.Space, time: core.Time) -> np.ndarray:
    """The absolute amplitudes of the planes xt, yt and xy `[..., side, side, 3]`, x to the right and y
    up."""
    planes = np.concatenate([time.cast(core.ga.subspace("xt yt")).kernel, space.cast(core.ga.subspace("xy")).kernel], axis=-1)
    image = np.abs(planes).reshape(planes.shape[:-2] + (grid.side, grid.side, 3))
    return np.swapaxes(image, -3, -2)[..., ::-1, :, :]


def animate(grid: core.Grid, space: core.Space, time: core.Time, columns: int, pixels: int) -> list[np.ndarray]:
    """Frames of fields `[tiles, frames, cells]`, the tiles in rows of the given count, each exposed
    over all its frames, each cell drawn the given number of pixels across."""
    rgb = colours(grid, space, time)                                          # [tiles, frames, side, side, 3]
    exposure = np.percentile(rgb, EXPOSURE, axis=(1, 2, 3, 4), keepdims=True)
    rgb = np.clip(rgb / exposure, 0, 1)
    tiles, frames = rgb.shape[:2]
    rgb = np.concatenate([rgb, np.zeros((-tiles % columns,) + rgb.shape[1:])])
    rows = rgb.reshape((-1, columns) + rgb.shape[1:])                         # [rows, columns, frames, side, side, 3]
    pictures = np.moveaxis(rows, 2, 0)                                         # [frames, rows, columns, side, side, 3]
    pictures = pictures.transpose(0, 1, 3, 2, 4, 5).reshape(frames, rows.shape[0] * grid.side, columns * grid.side, 3)
    return list(np.repeat(np.repeat((255 * pictures).astype(np.uint8), pixels, axis=1), pixels, axis=2))
