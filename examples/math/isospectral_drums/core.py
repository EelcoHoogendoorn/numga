"""Isospectral drums: vibration fields transplanted between two drums of different shapes with the same
frequencies, as sparse maps between fields on their lattices.

Each drum is cut from seven equal right triangles. Restriction maps read a field's triangles; a signed
mixing makes each destination triangle from three source pieces, and the adjoints of the destination's
restriction maps put them in place. The resulting transplant carries one drum's wave operator onto the
other's, so standing waves keep their frequencies.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga
from numga.sparse import SparseExtensor

context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()


# --- math -----------------------------------------------------------------------------
def tile_mixing(contributors: np.ndarray, signs: np.ndarray,
                n_samples: int) -> SparseExtensor:
    """Signed sums of triangle fields in their common local coordinates."""
    n_tiles = len(contributors)
    tile_indices = np.arange(n_tiles * n_samples).reshape(n_tiles, n_samples)
    read_tile = SparseExtensor.selection(context, tile_indices, n_tiles * n_samples) * Scalar  # [tiles] Scalar[samples] <- Scalar[tile samples]
    # A read selects one triangle; its adjoint inserts a triangle into that position.
    signed_tiles = read_tile[contributors] * signs[..., None]        # [tiles, terms] Scalar[samples] <- Scalar[tile samples]
    return read_tile[:, None].adjoint()(signed_tiles).sum(axis=1).sum(axis=0)
