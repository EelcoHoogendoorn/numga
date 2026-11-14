"""Quadratic self-products `x * T(x)` between a multivector `x` and a grade or sign transform `T(x)`
cancel off-diagonal blade pairs whenever `b_i * T(b_j) + b_j * T(b_i) == 0`, shrinking the static
output support below the general bilinear product `subspace * subspace`. Vectors under `v * v.reverse()`
land purely in grade `0` even though `vector * vector` spans grades `0` and `2`, and even elements
under `r * r.reverse()` land purely in grade `0` up to dimension three while reaching grades `0`
and `4` in dimension four.

Because real or complex scalar coordinates commute (`x_i * x_j == x_j * x_i`), off-diagonal terms
`(i, j)` and `(j, i)` combine with half their signed sum. Under an orthogonal diagonal metric,
`b_i * T(b_j)` and `b_j * T(b_i)` always land on the same canonical output mask `b_i ^ b_j` with
either matching signs, opposite signs, or zeros from a null contraction, so `(c_ij + c_ji) // 2`
remains an exact integer in `(-1, 0, +1)` with no rounding.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.subspace import SubSpace


def grade_transform_signs(algebra: Algebra, transform: str, masks: Iterable[int]) -> np.ndarray:
    """Named grade transforms (`"identity"`, `"reverse"`, `"clifford_conjugate"`, `"involute"`,
    `"scalar_negation"`, and `"pseudoscalar_negation"`) assign `+1` or `-1` to each canonical blade
    mask."""
    blades = np.asarray(tuple(masks), dtype=np.int64)
    signs = {
        "identity": lambda: np.ones_like(blades),
        "reverse": lambda: algebra.reverse_sign(blades),
        "clifford_conjugate": lambda: algebra.conjugate_sign(blades),
        "involute": lambda: algebra.involute_sign(blades),
        "scalar_negation": lambda: np.where(blades == 0, 1, -1),
        "pseudoscalar_negation": lambda: np.where((blades == 0) | (blades == algebra.pseudoscalar_mask), 1, -1),
    }[transform]()
    return np.asarray(signs, dtype=np.int8)


@lru_cache(maxsize=None)
def symmetric_product_table(subspace: SubSpace, transform: str = "reverse") -> tuple[np.ndarray, np.ndarray]:
    """Symmetrizing `x * T(x)` across the `n` blades of `subspace` yields `[n, n]` arrays of
    canonical output masks and exact integer pair coefficients `(c + c.T) // 2`."""
    algebra = subspace.algebra
    table = algebra.geometric_product_table(subspace.masks, subspace.masks)
    coefficients = table.coefficients * grade_transform_signs(algebra, transform, subspace.masks)
    return table.blades, (coefficients + coefficients.T) // 2


@lru_cache(maxsize=None)
def symmetric_product_support(subspace: SubSpace, transform: str = "reverse") -> frozenset[int]:
    """Nonzero pair coefficients determine the canonical blade masks reachable by `x * T(x)`."""
    blades, coefficients = symmetric_product_table(subspace, transform)
    return frozenset(blades[coefficients != 0].tolist())
