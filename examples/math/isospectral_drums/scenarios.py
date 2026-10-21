"""Two differently shaped drums with corresponding standing waves."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from numga.sparse import SparseExtensor
from examples.math.isospectral_drums.core import Scalar, Vector, context, mv, tile_mixing
from examples.math.isospectral_drums.numerics import drum, triangle

N_LATTICE = 4
SOURCE_INDICES = np.array([
    [0, 1, 4], [5, 1, 4], [5, 9, 4], [5, 9, 6],
    [10, 9, 6], [10, 9, 14], [10, 11, 6],
])
TARGET_INDICES = np.array([
    [5, 1, 4], [5, 9, 4], [8, 9, 4], [5, 9, 6],
    [10, 9, 6], [10, 11, 6], [10, 11, 14],
])
CONTRIBUTORS = np.array([
    [1, 0, 6], [0, 2, 4], [1, 2, 3], [3, 0, 5],
    [4, 1, 5], [5, 2, 6], [4, 3, 6],
])
SIGNS = np.array([
    [1, -1, 1], [1, 1, 1], [1, -1, 1], [1, -1, 1],
    [1, -1, -1], [1, -1, 1], [1, -1, -1],
])


# The samples along each tile's legs; the standing waves solved for, the three mixed into the vibration
# and their amplitudes; the displacement at the ends of the colour scale; the destination triangle whose
# pieces are drawn; the frames, over one period of the slowest wave mixed in, and their duration.
N_SUBDIVISIONS = 24
N_MODES = 9
MIXED_MODES = np.array([0, 2, 5])
MIX_AMPLITUDES = np.array([0.30, 0.22, 0.26])
EXPOSURE = 0.45
TARGET_TILE = 0
N_FRAMES = 96
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def drums() -> tuple[Vector, tuple[Vector, Vector], tuple[np.ndarray, np.ndarray], Scalar, Scalar, Scalar, SparseExtensor,
                     SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]:
    """The two drums' tiles `[2, tiles, 3]`, vertices and faces, the source drum's standing waves and their
    squared frequencies `[modes]`, the transplantation, the restriction of the source field to its tiles,
    the two wave operators, and the maps that put the fixed boundary back."""
    offsets = np.arange(N_LATTICE)
    lattice = (mv.x * offsets[None, :] + mv.y * offsets[:, None]).reshape((-1,))  # [lattice vertices] Vector
    tiles = stack((lattice[SOURCE_INDICES], lattice[TARGET_INDICES]))   # [2, tiles, 3] Vector
    source_vertices, source_faces, source_neighbors, source_restriction, _, source_lift = drum(
        SOURCE_INDICES, N_LATTICE, N_SUBDIVISIONS)
    target_vertices, target_faces, target_neighbors, _, target_assembly, target_lift = drum(
        TARGET_INDICES, N_LATTICE, N_SUBDIVISIONS)
    spacing = 1 / N_SUBDIVISIONS
    source_laplacian = (4 * Scalar - source_neighbors) / spacing**2    # Scalar[source sites] <- Scalar[source sites]
    target_laplacian = (4 * Scalar - target_neighbors) / spacing**2    # Scalar[target sites] <- Scalar[target sites]

    n_tiles = len(SOURCE_INDICES)
    n_samples = source_restriction.shape[0] // n_tiles
    mixing = tile_mixing(CONTRIBUTORS, SIGNS, n_samples)              # Scalar[tile samples] <- Scalar[tile samples]
    transplant = target_assembly(mixing(source_restriction))          # Scalar[target sites] <- Scalar[source sites]

    # Inserting a zero boundary preserves the interior field's inner product.
    measure = source_lift.adjoint()(source_lift)                     # Scalar[source sites] <- Scalar[source sites]
    eigenvalues, modes = source_laplacian.eigh(measure, N_MODES)       # [modes] Scalar, [modes] Scalar[source sites]
    return (tiles, (source_vertices, target_vertices), (source_faces, target_faces), modes / spacing, eigenvalues,
            source_laplacian, target_laplacian, transplant, source_restriction, source_lift, target_lift)


def vibration(modes: Scalar, eigenvalues: Scalar, transplant: SparseExtensor, source_lift: SparseExtensor,
              target_lift: SparseExtensor) -> Iterator[tuple[Scalar, Scalar]]:
    """A mix of standing waves on the source drum, each at its own frequency, and its transplant on the
    target drum, frame by frame over one period of the slowest."""
    frequencies = eigenvalues[MIXED_MODES].square_root().to_array()      # [mixed]
    times = np.linspace(0, 2 * np.pi / frequencies[0], N_FRAMES, endpoint=False)
    weights = MIX_AMPLITUDES * np.cos(times[:, None] * frequencies)     # [frames, mixed]
    for frame_weights in weights:
        source = (modes[MIXED_MODES] * frame_weights).sum(axis=0)       # Scalar[source sites]
        yield source_lift(source), target_lift(transplant(source))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.math.isospectral_drums import render

    (tiles, vertices, faces, modes, eigenvalues, source_laplacian, target_laplacian, transplant, source_restriction,
     source_lift, target_lift) = drums()
    save_figure(render.draw_tiles(tiles), "isospectral_drums_tiles")

    # Three signed pieces of the source wave make one destination triangle's piece.
    n_tiles = len(SOURCE_INDICES)
    n_samples = source_restriction.shape[0] // n_tiles
    tile_indices = np.arange(n_tiles * n_samples).reshape(n_tiles, n_samples)
    read_tile = SparseExtensor.selection(context, tile_indices, n_tiles * n_samples) * Scalar  # [tiles] Scalar[samples] <- Scalar[tile samples]
    source_pieces = read_tile(source_restriction(modes[MIXED_MODES[-1]] * MIX_AMPLITUDES[-1]))   # [tiles] Scalar[samples]
    contributions = source_pieces[CONTRIBUTORS[TARGET_TILE]] * SIGNS[TARGET_TILE]  # [terms] Scalar[samples]
    pieces = concatenate((contributions, contributions.sum(axis=0)[None]), axis=0)  # [terms + 1] Scalar[samples]
    local_vertices, local_faces = triangle(N_SUBDIVISIONS)
    save_figure(render.draw_pieces(local_vertices, local_faces, pieces, EXPOSURE), "isospectral_drums_pieces")

    save_animation(render.frames(vertices, faces, tiles, vibration(modes, eigenvalues, transplant, source_lift, target_lift),
                                 EXPOSURE), "isospectral_drums", DURATION_MS)

    # checks
    # The transplant carries the wave operator across, so each standing wave goes to one of the same frequency.
    commutator = target_laplacian(transplant) - transplant(source_laplacian)   # Scalar[target sites] <- Scalar[source sites]
    np.testing.assert_allclose(commutator(modes).to_array(), 0, atol=1e-6)


if __name__ == "__main__":
    main()
