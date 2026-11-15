"""Coefficient-level operations act directly on an extensor's backing array while preserving its
algebraic `GAType`."""

from __future__ import annotations

import numpy as np

from numga.extensor import Extensor


@Extensor.real.register(lambda t: True)
def real(extensor: Extensor) -> Extensor:
    """Extracting the real part of every blade coefficient returns an extensor of the same `GAType`
    in a real-valued backend context, discarding imaginary round-off on real eigenpairs."""
    context = extensor.context
    dtype = np.real(np.zeros((), dtype=context.dtype)).dtype
    context = context.with_dtype(dtype)
    return Extensor._from_prepared_kernel(context, extensor.gatype, context.xp.real(extensor._kernel))
