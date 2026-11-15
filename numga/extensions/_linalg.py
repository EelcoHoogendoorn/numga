"""Linear-algebra extensions share kernel wrapping, context promotion, and regressive pairing
plumbing across dense operators, bilinear forms, and sparse fields.

Spectral decompositions represent modes as nullary extensors whose trailing batch axis enumerates
modes, paired with scalar extensors on that same mode axis. When a real operator produces complex
eigenpairs, result wrapping promotes the backend context dtype rather than truncating imaginary
parts, while numerical solvers lower exact contexts onto concrete array backends.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from numga.backend.context import binding_context
from numga.extensor import Extensor
from numga.gatype import GAType, Orientation


@lru_cache(maxsize=None)
def _reading(first: GAType, second: GAType) -> Extensor:
    """Complementary nullary types pair via `&` with the extrinsic factor placed on the left when
    orientations differ, because swapping complementary grades in even dimensions introduces the
    grade involute; matching orientations keep the given argument order."""
    algebra = first.algebra
    scalar = algebra.subspace.scalar()
    if first.orientation is Orientation.extrinsic or second.orientation is Orientation.intrinsic:
        return (first & second).cast(scalar)
    swapped = (second & first).cast(scalar)                                    # Scalar <- (second, first)
    return algebra.operator.build(
        (scalar, first.output_subspace, second.output_subspace),
        (swapped.gatype.orientation, first.orientation, second.orientation),
        swapped._kernel.values.transpose(0, 2, 1),
    )


def _result(source: Extensor, gatype: GAType, kernel: Any) -> Extensor:
    context = source.context
    dtype = np.result_type(context.dtype, context.kernel_dtype(kernel))
    if dtype != context.dtype:
        context = context.with_dtype(dtype)
    return Extensor._from_prepared_kernel(
        context, gatype, context.xp.asarray(kernel, dtype=dtype),
    )


def _scalars(source: Extensor, kernel: Any) -> Extensor:
    return _result(source, source.algebra.gatype.scalar(), kernel[..., None])


def _eigenpairs(operator: Extensor, values: Any, vectors: Any) -> tuple[Extensor, Extensor]:
    vectors = _result(
        operator, operator.gatype.derive.axis(1),
        operator.context.xp.swapaxes(vectors, -1, -2),
    )
    return _scalars(vectors, values), vectors


def _linear_system(operator: Extensor, rhs: Extensor) -> tuple[Extensor, Extensor]:
    context = binding_context(operator.context, (rhs.context,))
    return context.lower(operator), context.lower(rhs).cast(operator.axes[0])
