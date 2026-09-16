"""Trace operations for square endomorphisms and scalars.

The trace is defined for square maps (operators with equal input and output
subspaces or isomorphic supports) and scalars.
"""

from __future__ import annotations

from typing import NoReturn

from numga.extensor import Extensor
from numga.gatype import GATypePattern


@Extensor.trace.register(lambda t: t.is_scalar)
def trace_scalar(value: Extensor) -> Extensor:
    """Trace of a scalar is the scalar itself."""
    return value


@Extensor.trace.register(
    lambda t: t.is_square_map and t.subspaces[0].same_support(t.subspaces[1])
)
def trace_square(value: Extensor) -> Extensor:
    """Trace of an arity-1 endomorphism on matching blade support.

    Casts the output subspace to the input subspace using numga's native
    AxisTransform relayout, then computes the contraction along the diagonal.
    """
    matched = value.cast(value.axes[1])
    kernel = matched.context.matrix_trace(matched._kernel)
    scalar_type = matched.algebra.gatype.scalar()
    return Extensor._from_prepared_kernel(
        matched.context,
        scalar_type,
        kernel,
    )


@Extensor.trace.register(GATypePattern.map())
def trace_nonsquare(value: Extensor) -> NoReturn:
    raise TypeError("trace requires an endomorphism with matching subspace support")
