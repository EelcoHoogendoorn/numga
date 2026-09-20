"""Trace an output against an input slot, retaining the other open inputs."""

from __future__ import annotations

from typing import NoReturn
from functools import lru_cache

from numga.extensor import Extensor
from numga.gatype import GAType, GATypePattern


@Extensor.trace.register(lambda t: t.is_scalar)
def trace_scalar(value: Extensor) -> Extensor:
    """Trace of a scalar is the scalar itself."""
    return value


@Extensor.trace.register(
    lambda t: any(space.support_is_subset_of(t.output_subspace) for space in t.input_subspaces)
)
def trace_square(value: Extensor, *, slot: int = 0) -> Extensor:
    """Trace the output component matching an input slot, numbered from zero."""
    result_type = _trace_type(value.gatype, slot)
    matched = value.cast(value.input_subspaces[slot])
    kernel = matched.context.matrix_trace(
        matched._kernel, axis1=matched.ndim, axis2=matched.ndim + slot + 1,
        scalar_axis=matched.ndim,
    )
    return Extensor._from_prepared_kernel(
        matched.context,
        result_type,
        kernel,
    )


@lru_cache(maxsize=None)
def _trace_type(gatype: GAType, slot: int) -> GAType:
    """Resolve matching blade layouts and the remaining slots statically."""
    inputs = gatype.input_subspaces
    if not inputs[slot].support_is_subset_of(gatype.output_subspace):
        raise TypeError("trace requires an endomorphism with matching subspace support")
    return gatype.algebra.gatype(
        (gatype.algebra.subspace.scalar(),) + inputs[:slot] + inputs[slot + 1:],
    )


@Extensor.trace.register(GATypePattern.map())
def trace_nonsquare(value: Extensor) -> NoReturn:
    raise TypeError("trace requires an endomorphism with matching subspace support")
