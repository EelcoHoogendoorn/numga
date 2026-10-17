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


def _is_form(gatype: GAType) -> bool:
    return gatype.arity == 2 and gatype.output_subspace.same_support(gatype.algebra.subspace.scalar())


def _is_square_form(gatype: GAType) -> bool:
    """A form pairing one space with itself: the same blades and the same orientation in both slots."""
    return (_is_form(gatype) and gatype.subspaces[1].same_support(gatype.subspaces[2])
            and gatype.orientations[1] is gatype.orientations[2])


def _matching_forms(left: GAType, right: GAType) -> bool:
    return (_is_square_form(left) and _is_square_form(right)
            and left.subspaces[2].same_support(right.subspaces[2])
            and left.orientations == right.orientations)


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
    if rhs.gatype.orientation is not value.gatype.orientation:
        raise ValueError(
            f"cannot solve for an {rhs.gatype.orientation.value} right-hand side against a map whose "
            f"output takes an {value.gatype.orientation.value} orientation"
        )
    context = binding_context(value.context, (rhs.context,))
    return context.lower(value), context.lower(rhs).cast(value.axes[0])
