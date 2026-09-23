"""Coefficient-independent support of a value times a grade transform of itself."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.subspace import SubSpace


def grade_transform_sign(algebra: Algebra, transform: str, mask: int) -> int:
    """The grade signs used by the recursive inverse reductions."""

    return {
        "identity": lambda blade: 1,
        "reverse": algebra.reverse_sign,
        "clifford_conjugate": algebra.conjugate_sign,
        "scalar_negation": lambda blade: 1 if blade == 0 else -1,
        "pseudoscalar_negation": lambda blade: (
            1 if blade in (0, algebra.pseudoscalar_mask) else -1
        ),
        "involute": algebra.involute_sign,
    }[transform](mask)


@lru_cache(maxsize=None)
def symmetric_product_terms(
    space: SubSpace, transform: str = "reverse",
) -> tuple[tuple[int, int, int, int], ...]:
    """Return nonzero quadratic terms, with each unordered input pair once.

    Coefficients commute. Combining the two orders here proves cancellations
    without inspecting values or constructing an intermediate dense tensor.
    """
    if len(space) == 0:
        return ()

    algebra = space.algebra
    table = algebra.geometric_product_table(space.masks, space.masks)
    signs = np.array(
        [grade_transform_sign(algebra, transform, mask) for mask in space.masks],
        dtype=np.int8,
    )

    c = table.coefficients * signs[None, :]
    sym = c + c.T
    np.fill_diagonal(sym, np.diag(c))

    i_idx, j_idx = np.triu_indices(len(space))
    valid = sym[i_idx, j_idx] != 0
    valid_i = i_idx[valid]
    valid_j = j_idx[valid]
    valid_blades = table.blades[valid_i, valid_j]
    valid_coeffs = sym[valid_i, valid_j]

    return tuple(
        (int(blade), int(i), int(j), int(coeff))
        for blade, i, j, coeff in zip(valid_blades, valid_i, valid_j, valid_coeffs)
    )


@lru_cache(maxsize=None)
def symmetric_product_support(
    space: SubSpace, transform: str = "reverse",
) -> frozenset[int]:
    return frozenset(term[0] for term in symmetric_product_terms(space, transform))
