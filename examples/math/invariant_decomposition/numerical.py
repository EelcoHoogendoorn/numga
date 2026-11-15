"""Companion-matrix eigenvalues isolate the NumPy cubic-root solve in a single typed helper."""

from __future__ import annotations

import numpy as np

from numga.extensor import Extensor


def cubic_roots(coefficients: Extensor) -> Extensor:
    """Companion-matrix eigenvalues give the ascending real roots (`[..., 3] Scalar+`) of `x**3 + c[0] * x**2 + c[1] * x + c[2]`."""
    companion = np.zeros(coefficients.shape + (3,))                            # [..., 3, 3]
    companion[..., 0, :] = -coefficients.to_array()
    companion[..., [1, 2], [0, 1]] = 1.0
    roots = np.sort(np.linalg.eigvals(companion).real, axis=-1)                # [..., 3]
    return coefficients.context.multivector.scalar(roots[..., None])           # [..., 3] Scalar+
