"""The roots of cubics: an excursion from numga to NumPy, scoped to one function that takes the cubics' coefficients
as scalars and returns their roots as scalars."""

from __future__ import annotations

import numpy as np

from numga.extensor import Extensor


def cubic_roots(coefficients: Extensor) -> Extensor:
    """The roots of the cubics `x**3 + c[0] * x**2 + c[1] * x + c[2]` for the scalar coefficients c along the last
    axis, ascending: the eigenvalues of their companion matrices; real here."""
    companion = np.zeros(coefficients.shape + (3,))
    companion[..., 0, :] = -coefficients.to_array()
    companion[..., [1, 2], [0, 1]] = 1.0
    roots = np.sort(np.linalg.eigvals(companion).real, axis=-1)                # [..., 3]
    return coefficients.context.multivector.scalar(roots[..., None])
