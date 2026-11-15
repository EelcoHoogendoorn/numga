"""Seven-triangle source and target drum parameters drive standing-wave transplantation figures and animation."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.math.isospectral_drums.core import (
    Scalar,
    lattice_tiles,
    standing_waves,
    tile_mixing,
    tile_pieces,
    vibration_states,
)
from examples.math.isospectral_drums.numerics import drum, triangle

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

n_subdivisions = 24
n_modes = 9
mixed_modes = np.array([0, 2, 5])
mix_amplitudes = np.array([0.30, 0.22, 0.26])
exposure = 0.45
target_tile = 0
n_frames = 72
frame_duration_ms = 40
n_colours = 32


def main() -> None:
    """Renders the drum tilings, local signed pieces, and vibrating drums before checking operator intertwining."""
    from examples.util.animation import save_animation, save_figure
    from examples.math.isospectral_drums import render

    n_tiles = len(source_indices)
    tiles = stack((lattice_tiles(source_indices, n_lattice), lattice_tiles(target_indices, n_lattice)))  # [2, n_tiles, 3] Point-
    save_figure(render.tiles_figure(tiles), "isospectral_drums_tiles")

    source_vertices, source_topology, source_neighbors, source_restriction, _, source_extension = drum(
        source_indices, n_lattice, n_subdivisions,
    )
    target_vertices, target_topology, target_neighbors, target_restriction, target_assembly, target_extension = drum(
        target_indices, n_lattice, n_subdivisions,
    )
    source_laplacian = (4 * Scalar - source_neighbors) * n_subdivisions**2      # (Scalar+ <- Scalar+)[V_source <- V_source]
    target_laplacian = (4 * Scalar - target_neighbors) * n_subdivisions**2      # (Scalar+ <- Scalar+)[V_target <- V_target]
    eigenvalues, waves = standing_waves(source_laplacian, source_extension, n_subdivisions, n_modes)

    n_samples = source_restriction.shape[0] // n_tiles
    mixing = tile_mixing(contributors, signs, n_samples)                        # (Scalar+ <- Scalar+)[n_tiles * n_samples <- n_tiles * n_samples]
    transplant = target_assembly(mixing(source_restriction))                    # (Scalar+ <- Scalar+)[V_target <- V_source]
    pieces = tile_pieces(waves[mixed_modes[-1]] * mix_amplitudes[-1], source_restriction, contributors, signs, target_tile)  # [4] Scalar+[n_samples]
    local_vertices, local_topology = triangle(n_subdivisions)
    save_figure(render.pieces_figure(local_vertices, local_topology, pieces, exposure), "isospectral_drums_pieces")

    states = []
    stream = (
        states.append(state) or state
        for state in vibration_states(
            waves, eigenvalues, mixed_modes, mix_amplitudes,
            transplant, source_extension, target_extension, n_frames,
        )
    )
    save_animation(
        render.drum_frames((source_vertices, target_vertices), (source_topology, target_topology), tiles, exposure, stream),
        "isospectral_drums", frame_duration_ms, colors=n_colours,
    )

    # checks
    double_areas = (tiles[:, :, 1] - tiles[:, :, 0]) & (tiles[:, :, 2] - tiles[:, :, 0])  # [2, n_tiles] Scalar-
    assert ((double_areas.squared() - 1).abs() < 1e-9).all()

    commutator = target_laplacian(transplant) - transplant(source_laplacian)    # (Scalar+ <- Scalar+)[V_target <- V_source]
    assert (commutator(waves).abs() < 1e-6).all()
    assert ((target_restriction(transplant(waves)) - mixing(source_restriction(waves))).abs() < 1e-9).all()

    source_norms = stack([pair[0].batch().squared().sum() for pair in states])  # [n_frames] Scalar+
    target_norms = stack([pair[1].batch().squared().sum() for pair in states])  # [n_frames] Scalar+
    assert ((target_norms - 2 * source_norms).abs() < 1e-9).all()
    assert all((pair[0].batch()[source_topology.exterior[0]].abs() < 1e-9).all() for pair in states)
    assert all((pair[1].batch()[target_topology.exterior[0]].abs() < 1e-9).all() for pair in states)


if __name__ == "__main__":
    main()
