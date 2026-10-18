"""Shared plumbing for the linear-algebra extensions: type predicates and result wrapping.

Eigenvectors and singular vectors are nullary extensors: the last batch axis
enumerates modes. Spectral values are scalar extensors on the same mode axis.
Complex eigenpairs promote the result context; they are never truncated to
real coefficients. Numerical methods require a concrete NumPy/JAX context.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from numga.backend.context import binding_context
from numga.extensor import Extensor
from numga.gatype import GAType


def _result(value: Extensor, gatype: GAType, kernel: Any) -> Extensor:
    context = value.context
    dtype = np.result_type(context.dtype, context.kernel_dtype(kernel))
    if dtype != context.dtype:
        context = context.with_dtype(dtype)
    return Extensor._from_prepared_kernel(
        context, gatype, context.xp.asarray(kernel, dtype=dtype),
    )


def _scalars(value: Extensor, kernel: Any) -> Extensor:
    return _result(value, value.algebra.gatype.scalar(), kernel[..., None])


def _eigenpairs(value: Extensor, values: Any, vectors: Any) -> tuple[Extensor, Extensor]:
    vectors = _result(
        value, value.gatype.derive.axis(1),
        value.context.xp.swapaxes(vectors, -1, -2),
    )
    return _scalars(vectors, values), vectors


def _linear_system(value: Extensor, rhs: Extensor) -> tuple[Extensor, Extensor]:
    context = binding_context(value.context, (rhs.context,))
    return context.lower(value), context.lower(rhs).cast(value.axes[0])
