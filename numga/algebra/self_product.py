"""Coefficient-independent support of a value times a grade transform of itself."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.subspace import SubSpace


def grade_transform_signs(algebra: Algebra, transform: str, masks: Iterable[int]) -> np.ndarray:
    """The sign a grade transform gives each of the given blades `[n]`: the reverse, the Clifford conjugate
    and the grade involution by grade, the negations by whether a blade is the scalar or the pseudoscalar."""
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
def symmetric_product_table(space: SubSpace, transform: str = "reverse") -> tuple[np.ndarray, np.ndarray]:
    """A value times a grade transform of itself, `x * T(x)`, over every pair of the value's blades: the
    output blade of each pair `[n, n]`, and its coefficient `[n, n]`, symmetric in the pair.

    The coefficients of one value commute, so the two orders of a pair combine, and each stands for half
    their sum. Both orders of a basis product land on one blade, with equal or opposite signs, and a null
    basis vector makes both vanish: every half is a whole number.
    """
    algebra = space.algebra
    table = algebra.geometric_product_table(space.masks, space.masks)
    coefficients = table.coefficients * grade_transform_signs(algebra, transform, space.masks)
    return table.blades, (coefficients + coefficients.T) // 2


@lru_cache(maxsize=None)
def symmetric_product_support(space: SubSpace, transform: str = "reverse") -> frozenset[int]:
    """The blades a value of the space times a grade transform of itself can reach."""
    blades, coefficients = symmetric_product_table(space, transform)
    return frozenset(blades[coefficients != 0].tolist())
