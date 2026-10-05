"""Pair two slots of an extensor and sum over them, keeping its other slots open.

The output is slot 0 and the inputs are slots 1 to n, in the order of the kernel's axes. An output
pairs with an input of its own space without a metric, as a vector with its dual, and two inputs
pair without a metric when their spaces are complementary, through their regressive product: both
are `trace`. Two inputs of one space pair through the space's metric: that is `contract`. In a space
that is its own complement, as the bivectors of four dimensions, both apply, and the name chooses.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from numga.extensor import Extensor
from numga.gatype import GAType
from numga.operator.kernel import SymbolicKernel
from numga.subspace import SubSpace


def trace(value: Extensor, first: int = 0, second: int = 1) -> Extensor:
    """Pair two slots without a metric and sum over them.

    With the output as `first`, the output against an input of the same space: on `Space <- Space`
    the matrix trace. A slot spanning only part of the output is refused, since tracing it would
    choose a complement by blade label. With two inputs, inputs of complementary spaces through
    their regressive product: the sum of `value(e_k, e^k)` over the blades `e_k` of the first and
    the blades `e^k` of the second with `e_l & e^k` one for `l == k` and zero otherwise. In PGA,
    `(Plane & f(Point)).trace(1, 2) == f.trace()`.
    """
    if first == 0:
        return _trace_output(value, second - 1)
    inputs = value.input_subspaces
    return _paired(value, first - 1, second - 1, _incidence_reciprocal(inputs[first - 1], inputs[second - 1]))


def contract(value: Extensor, first: int = 1, second: int = 2) -> Extensor:
    """Pair two inputs of one space through its metric and sum over them.

    The metric is the slot's own, the inner product of a blade with the reverse of another, as for
    the spectra of forms: `V | V` on vectors. The result is the sum of `value(e_k, e^k)` over the
    space's blades `e_k` and their reciprocals `e^k` under that metric; it does not depend on the
    basis. A form's contraction is its trace with one slot raised by the metric, and contracting
    `Vector * f(Vector)` gives the vector derivative of a linear map `f`: its trace plus the bivector
    of its skew part, `f - f.adjoint() == Vector | curl`. A space with a null blade has no
    reciprocal blades, and is refused.
    """
    if first == 0:
        raise TypeError("an output pairs with an input of its own space without a metric: trace it")
    inputs = value.input_subspaces
    if not inputs[second - 1].same_support(inputs[first - 1]):
        raise TypeError(f"contract pairs inputs of one space; got {inputs[first - 1]} and {inputs[second - 1]}")
    return _paired(value, first - 1, second - 1, _metric_reciprocal(inputs[first - 1]))


def _paired(value: Extensor, first: int, second: int, reciprocal: Extensor) -> Extensor:
    """Feed the second input the reciprocal of each blade of the first, then sum the diagonal."""
    inputs = value.input_subspaces
    slots = [value.algebra.gatype(space) for space in inputs]
    slots[second] = reciprocal
    raised = value(*slots)
    offset = raised.ndim + 1
    kernel = raised.context.axes_trace(raised._kernel, offset + first, offset + second)
    remaining = tuple(space for slot, space in enumerate(inputs) if slot not in (first, second))
    return Extensor._from_prepared_kernel(
        raised.context, value.algebra.gatype((value.output_subspace,) + remaining), kernel,
    )


def _trace_output(value: Extensor, slot: int) -> Extensor:
    result_type = _trace_type(value.gatype, slot)
    matched = value.cast(value.input_subspaces[slot])
    kernel = matched.context.matrix_trace(
        matched._kernel, axis1=matched.ndim, axis2=matched.ndim + slot + 1,
        scalar_axis=matched.ndim,
    )
    return Extensor._from_prepared_kernel(matched.context, result_type, kernel)


@lru_cache(maxsize=None)
def _trace_type(gatype: GAType, slot: int) -> GAType:
    """Resolve matching blade layouts and the remaining slots statically."""
    inputs = gatype.input_subspaces
    if not inputs[slot].same_support(gatype.output_subspace):
        raise TypeError("trace pairs an output with an input of the same space")
    return gatype.algebra.gatype(
        (gatype.algebra.subspace.scalar(),) + inputs[:slot] + inputs[slot + 1:],
    )


@lru_cache(maxsize=None)
def _metric_reciprocal(space: SubSpace) -> Extensor:
    """The exact map sending each blade of the space to its reciprocal under the slot's metric."""
    from numga.extensions.forms import _default_metric
    metric = _default_metric(space)._kernel.values[0]                         # [blades, blades]
    if not metric.any(axis=0).all():
        raise TypeError(f"contract needs an invertible metric; {space} has a null blade")
    return space.algebra.operator.build((space, space), SymbolicKernel(metric).inverse())


@lru_cache(maxsize=None)
def _incidence_reciprocal(first: SubSpace, second: SubSpace) -> Extensor:
    """The exact map sending each blade of the first space to the blade of the second that it
    meets in one, and every other blade of the first in zero."""
    algebra = first.algebra
    incidence = algebra.operator.regressive(first, second)                     # Scalar <- (first, second)
    pairing = incidence.cast(algebra.subspace.scalar())._kernel.values[0]       # [first blades, second blades]
    if pairing.shape[0] != pairing.shape[1] or np.linalg.matrix_rank(pairing) < len(first):
        raise TypeError(f"trace pairs inputs of complementary spaces; {first} and {second} pair through the metric: contract them")
    return algebra.operator.build((second, first), SymbolicKernel(pairing).inverse())
