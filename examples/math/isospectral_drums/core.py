"""Transplantation maps carry standing waves and vibrations between two differently shaped seven-triangle drums at identical frequencies.

Restriction maps read each drum's seven right triangles into shared local coordinates, signed three-piece sums form each destination triangle, and seam assembly welds those triangles back into one interior field that intertwines the two Dirichlet wave operators.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import VGA2D
from numga.sparse import Sparse, SparseExtensor

ga = VGA2D
context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()                                         # Scalar+
# Drum sample positions live in the plane as intrinsic antivectors.
Point = ga.gatype.antivector()                                      # Antivector-


# --- math -----------------------------------------------------------------------------
def lattice_tiles(tile_indices: np.ndarray, n_lattice: int) -> Point:
    """Row-major grid indices place triangle corners (`[..., 3] Point-`) on the `n_lattice` by `n_lattice` lattice."""
    offsets = np.arange(n_lattice)
    lattice = (mv.x.dual() * offsets[None, :] + mv.y.dual() * offsets[:, None]).reshape((-1,))  # [n_lattice * n_lattice] Point-
    return lattice[tile_indices]


def standing_waves(
    laplacian: SparseExtensor, extension: SparseExtensor, n_subdivisions: int, n_modes: int,
) -> tuple[Scalar, Scalar]:
    """Lowest `n_modes` interior standing waves (`[n_modes] Scalar+[V_interior]`) and squared frequencies (`[n_modes] Scalar+`) solve the Dirichlet eigenvalue problem."""
    # Zero-boundary extension preserves the interior inner product.
    measure = extension.adjoint()(extension)                            # (Scalar+ <- Scalar+)[V_interior <- V_interior]
    eigenvalues, modes = laplacian.eigh(measure, n_modes)               # [n_modes] Scalar+, [n_modes] Scalar+[V_interior]
    return eigenvalues, modes * n_subdivisions


def tile_mixing(contributors: np.ndarray, signs: np.ndarray, n_samples: int) -> SparseExtensor:
    """Signed three-piece sums map source triangle samples onto destination triangle samples (`(Scalar+ <- Scalar+)[n_tiles * n_samples <- n_tiles * n_samples]`)."""
    n_tiles = len(contributors)
    tile_indices = np.arange(n_tiles * n_samples).reshape(n_tiles, n_samples)
    tile_reading = Sparse.selection(tile_indices, n_tiles * n_samples)  # [n_tiles] Scalar+[n_samples <- n_tiles * n_samples]
    # Reading selects one triangle; its adjugate inserts a triangle into that position.
    signed_tiles = tile_reading[contributors] * signs[..., None]        # [n_tiles, 3] Scalar+[n_samples <- n_tiles * n_samples]
    return (tile_reading[:, None].adjugate() * signed_tiles).sum(axis=1).sum(axis=0) * Scalar


def tile_pieces(
    source_field: Scalar, source_restriction: SparseExtensor,
    contributors: np.ndarray, signs: np.ndarray, target_tile: int,
) -> Scalar:
    """Three signed source triangle fields and their sum (`[4] Scalar+[n_samples]`) form destination triangle `target_tile`."""
    n_tiles = len(contributors)
    n_samples = source_restriction.shape[0] // n_tiles
    tile_indices = np.arange(n_tiles * n_samples).reshape(n_tiles, n_samples)
    tile_reading = Sparse.selection(tile_indices, n_tiles * n_samples)  # [n_tiles] Scalar+[n_samples <- n_tiles * n_samples]
    source_tiles = tile_reading * source_restriction(source_field)      # [n_tiles] Scalar+[n_samples]
    contributions = source_tiles[contributors[target_tile]] * signs[target_tile]  # [3] Scalar+[n_samples]
    return concatenate((contributions, contributions.sum(axis=0)[None]), axis=0)


def vibration_states(
    waves: Scalar, eigenvalues: Scalar, mixed_modes: np.ndarray, mix_amplitudes: np.ndarray,
    transplant: SparseExtensor, source_extension: SparseExtensor, target_extension: SparseExtensor,
    n_frames: int,
) -> Iterator[tuple[Scalar, Scalar]]:
    """Yields paired full-drum displacement fields (`Scalar+[V_source]`, `Scalar+[V_target]`) over `n_frames` frames of one fundamental period."""
    n_mixed = len(mixed_modes)
    frequencies = eigenvalues[mixed_modes].square_root()                # [n_mixed] Scalar+
    period = 2 * np.pi / frequencies[0]                                 # [] Scalar+
    times = period * np.linspace(0, 1, n_frames, endpoint=False)[:, None]  # [n_frames, 1] Scalar+
    weights = mix_amplitudes * (frequencies * times).cos()              # [n_frames, n_mixed] Scalar+
    for frame_weights in weights:
        source_field = (waves[mixed_modes] * frame_weights).sum(axis=0)  # Scalar+[V_source]
        # Zero-boundary extension restores the fixed rim vertices for display.
        yield source_extension(source_field), target_extension(transplant(source_field))
