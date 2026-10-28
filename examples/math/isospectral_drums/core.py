"""Isospectral drums: vibration fields transplanted between two drums of different shapes with the same
frequencies, as sparse maps between fields on their lattices.

Each drum is cut from seven equal right triangles. Restriction maps read a field's triangles; a signed
mixing makes each destination triangle from three source pieces, which the adjoint of reading that triangle
puts in place, and the destination's assembly welds the triangles into one field. The resulting transplant
carries one drum's wave operator onto the other's, so standing waves keep their frequencies.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D
from numga.sparse import SparseExtensor

ga = VGA2D
context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()                                         # Scalar+
# Places on a drum: antivectors of the plane, intrinsic.
Point = ga.gatype.antivector()                                      # Antivector-


# --- math -----------------------------------------------------------------------------
def tile_mixing(contributors: np.ndarray, signs: np.ndarray,
                n_samples: int) -> SparseExtensor:
    """Signed sums of triangle fields in their common local coordinates."""
    n_tiles = len(contributors)
    tile_indices = np.arange(n_tiles * n_samples).reshape(n_tiles, n_samples)
    read_tile = SparseExtensor.selection(context, tile_indices, n_tiles * n_samples) * Scalar  # [tiles] (Scalar+ <- Scalar+)[samples <- tile samples]
    # A read selects one triangle; its adjoint inserts a triangle into that position.
    signed_tiles = read_tile[contributors] * signs[..., None]        # [tiles, terms] (Scalar+ <- Scalar+)[samples <- tile samples]
    return read_tile[:, None].adjoint()(signed_tiles).sum(axis=1).sum(axis=0)
