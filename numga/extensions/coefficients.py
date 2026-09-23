"""Operations on the coefficients of any extensor, independent of its type."""

from __future__ import annotations

import numpy as np

from numga.extensor import Extensor


@Extensor.real.register(lambda t: True)
def real(value: Extensor) -> Extensor:
    """Real part of every coefficient, in a real context: for eigenpairs known to be real."""
    context = value.context
    dtype = np.real(np.zeros((), dtype=context.dtype)).dtype
    context = type(context)(value.algebra, dtype=dtype, execution=context.execution)
    return Extensor._from_prepared_kernel(context, value.gatype, context.xp.real(value._kernel))
