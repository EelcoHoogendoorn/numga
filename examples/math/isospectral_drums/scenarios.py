"""Two differently shaped drums with corresponding standing waves."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from numga.sparse import SparseExtensor
from examples.math.isospectral_drums.core import Point, Scalar, context, mv, tile_mixing
from examples.math.isospectral_drums.numerics import Mesh, drum, triangle

n_lattice = 4
source_indices = np.array([
    [0, 1, 4], [5, 1, 4], [5, 9, 4], [5, 9, 6],
    [10, 9, 6], [10, 9, 14], [10, 11, 6],
])
target_indices = np.array([
    [5, 1, 4], [5, 9, 4], [8, 9, 4], [5, 9, 6],
    [10, 9, 6], [10, 11, 6], [10, 11, 14],
])
contributors = np.array([
    [1, 0, 6], [0, 2, 4], [1, 2, 3], [3, 0, 5],
    [4, 1, 5], [5, 2, 6], [4, 3, 6],
])
signs = np.array([
    [1, -1, 1], [1, 1, 1], [1, -1, 1], [1, -1, 1],
    [1, -1, -1], [1, -1, 1], [1, -1, -1],
])


# The samples along each tile's legs; the standing waves solved for, the three mixed into the vibration
# and their amplitudes; the displacement at the ends of the colour scale; the destination triangle whose
# pieces are drawn; the frames, over one period of the slowest wave mixed in, and their duration.
n_subdivisions = 24
n_modes = 9
mixed_modes = np.array([0, 2, 5])
mix_amplitudes = np.array([0.30, 0.22, 0.26])
exposure = 0.45
target_tile = 0
n_frames = 72
duration_ms = 40
# The animation's palette: the smooth field takes coarse banding.
n_colours = 32


# --- math -----------------------------------------------------------------------------
def drums() -> tuple[Point, tuple[Point, Point], tuple[Mesh, Mesh], Scalar, Scalar, SparseExtensor,
                     SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]:
    """The two drums' tiles `[drums, tiles, corners]`, vertices and meshes, the source drum's standing waves and their
    squared frequencies `[modes]`, the two wave operators, the transplantation, the restriction of the source field
    to its tiles, and the maps that put the fixed boundary back."""
    offsets = np.arange(n_lattice)
    lattice = (mv.x.dual() * offsets[None, :] + mv.y.dual() * offsets[:, None]).reshape((-1,))  # [lattice vertices] Point-
    tiles = stack((lattice[source_indices], lattice[target_indices]))   # [drums, tiles, corners] Point-
    source_vertices, source_mesh, source_neighbors, source_restriction, _, source_lift = drum(
        source_indices, n_lattice, n_subdivisions)
    target_vertices, target_mesh, target_neighbors, _, target_assembly, target_lift = drum(
        target_indices, n_lattice, n_subdivisions)
    spacing = 1 / n_subdivisions
    source_laplacian = (4 * Scalar - source_neighbors) / spacing**2    # (Scalar+ <- Scalar+)[source sites <- source sites]
    target_laplacian = (4 * Scalar - target_neighbors) / spacing**2    # (Scalar+ <- Scalar+)[target sites <- target sites]

    n_tiles = len(source_indices)
    n_samples = source_restriction.shape[0] // n_tiles
    mixing = tile_mixing(contributors, signs, n_samples)              # (Scalar+ <- Scalar+)[tile samples <- tile samples]
    transplant = target_assembly(mixing(source_restriction))          # (Scalar+ <- Scalar+)[target sites <- source sites]

    # Inserting a zero boundary preserves the interior field's inner product.
    measure = source_lift.adjoint()(source_lift)                     # (Scalar+ <- Scalar+)[source sites <- source sites]
    eigenvalues, modes = source_laplacian.eigh(measure, n_modes)       # [modes] Scalar+, [modes] Scalar+[source sites]
    return (tiles, (source_vertices, target_vertices), (source_mesh, target_mesh), modes / spacing, eigenvalues,
            source_laplacian, target_laplacian, transplant, source_restriction, source_lift, target_lift)


def vibration(modes: Scalar, eigenvalues: Scalar, transplant: SparseExtensor, source_lift: SparseExtensor,
              target_lift: SparseExtensor) -> Iterator[tuple[Scalar, Scalar]]:
    """A mix of standing waves on the source drum, each at its own frequency, and its transplant on the
    target drum, frame by frame over one period of the slowest."""
    frequencies = eigenvalues[mixed_modes].square_root()                # [mixed] Scalar+
    period = 2 * np.pi / frequencies[0]                                 # [] Scalar+
    times = period * np.linspace(0, 1, n_frames, endpoint=False)[:, None]   # [frames, 1] Scalar+
    weights = mix_amplitudes * (frequencies * times).cos()              # [frames, mixed] Scalar+
    for frame_weights in weights:
        source = (modes[mixed_modes] * frame_weights).sum(axis=0)       # Scalar+[source sites]
        yield source_lift(source), target_lift(transplant(source))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.isospectral_drums import render

    (tiles, vertices, meshes, modes, eigenvalues, source_laplacian, target_laplacian, transplant, source_restriction,
     source_lift, target_lift) = drums()
    save_figure(render.draw_tiles(tiles), "isospectral_drums_tiles")

    # Three signed pieces of the source wave make one destination triangle's piece.
    n_tiles = len(source_indices)
    n_samples = source_restriction.shape[0] // n_tiles
    tile_indices = np.arange(n_tiles * n_samples).reshape(n_tiles, n_samples)
    read_tile = SparseExtensor.selection(context, tile_indices, n_tiles * n_samples) * Scalar  # [tiles] (Scalar+ <- Scalar+)[samples <- tile samples]
    source_pieces = read_tile(source_restriction(modes[mixed_modes[-1]] * mix_amplitudes[-1]))   # [tiles] Scalar+[samples]
    contributions = source_pieces[contributors[target_tile]] * signs[target_tile]  # [terms] Scalar+[samples]
    pieces = concatenate((contributions, contributions.sum(axis=0)[None]), axis=0)  # [terms + 1] Scalar+[samples]
    local_vertices, local_mesh = triangle(n_subdivisions)
    save_figure(render.draw_pieces(local_vertices, local_mesh, pieces, exposure), "isospectral_drums_pieces")

    save_animation(render.frames(vertices, meshes, tiles, vibration(modes, eigenvalues, transplant, source_lift, target_lift),
                                 exposure), "isospectral_drums", duration_ms, colors=n_colours)

    # checks
    # The transplant carries the wave operator across, so each standing wave goes to one of the same frequency.
    commutator = target_laplacian(transplant) - transplant(source_laplacian)   # (Scalar+ <- Scalar+)[target sites <- source sites]
    np.testing.assert_allclose(commutator(modes).to_array(), 0, atol=1e-6)


if __name__ == "__main__":
    main()
