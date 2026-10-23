"""Trace and contraction reduce an extensor to a map of its remaining arguments.

Axes are numbered with the output as axis 0 and the inputs as axes 1 to n, in the order of the
kernel's axes. A axis carries an index unless it is a scalar output: a form `Scalar <- (V, V)` has
two, a map `V <- V` has two, and so the two axes may be left unnamed exactly when there are two.

`trace` pairs through the regressive product: two inputs of complementary spaces through `&`, and
the output, lifted to an input by meeting it with its complement, against an input of its own space.
`contract` pairs through the inner product: two inputs of one space through the space's metric, and
the output, lifted by `Space | value`, against an input of its own space. Either lift is a spelling
of the same thing: `f.trace() == (Antivector & f).trace(1, 2)` and `f.contract() ==
(Vector | f).contract(1, 2)`. On an output the pairing and its inverse cancel, so both give the
matrix trace, and need neither metric nor orientation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from numga.extensor import Extensor
from numga.gatype import GAType, Orientation
from numga.operator.kernel import SymbolicKernel


@Extensor.trace.register(lambda t: True)
def trace_unnamed(value: Extensor) -> Extensor:
    """The two axes that carry an index, paired through the regressive product."""
    return trace(value, *value.gatype.axis_roles.pair)


@Extensor.trace.overload(3).register(lambda t: True)
def trace(value: Extensor, first: int, second: int) -> Extensor:
    """Pair two axes through the regressive product and sum over them.

    Two inputs of complementary spaces pair through `&`: the sum of `value(e_k, e^k)` over the
    blades `e_k` of the first and the blades `e^k` of the second with `e_l & e^k` one for `l == k`
    and zero otherwise. The output pairs with an input of its own space, as if met with its
    complement first; on `Space <- Space` this is the matrix trace. In PGA,
    `(Plane & f(Point)).trace(1, 2) == f.trace()`.
    """
    first, second = value.gatype.axis_roles.paired(first, second)
    if first == 0:
        return _trace_output(value, second - 1)
    axes = value.gatype.derive
    reciprocal = _incidence_reciprocal(axes.axis(first), axes.axis(second))
    # summed through `&`, the pairing has the orientation of a regressive product of the two axes
    pairing = (value.gatype.orientations[first] * value.gatype.orientations[second]).flipped
    return _paired(value, first - 1, second - 1, reciprocal, pairing)


@Extensor.contract.register(lambda t: True)
def contract_unnamed(value: Extensor) -> Extensor:
    """The two axes that carry an index, paired through the inner product."""
    return contract(value, *value.gatype.axis_roles.pair)


@Extensor.contract.overload(3).register(lambda t: True)
def contract(value: Extensor, first: int, second: int) -> Extensor:
    """Pair two axes through the inner product and sum over them.

    Two inputs of one space pair through the space's own metric, the inner product of a blade with
    the reverse of another, as for the spectra of forms: `V | V` on vectors. The result is the sum
    of `value(e_k, e^k)` over the space's blades `e_k` and their reciprocals `e^k` under that metric;
    it does not depend on the basis, and a space with a null blade has no reciprocals, and is
    refused. The output pairs with an input of its own space, as if lifted by `Space | value` first;
    the metric and its inverse cancel, so this is the trace, for any metric. Contracting
    `Vector * f(Vector)` in axes 1 and 2 gives the vector derivative of a linear map `f`: its trace
    plus the bivector of its skew part, `f - f.adjoint() == Vector | curl`.
    """
    first, second = value.gatype.axis_roles.contracted(first, second)
    if first == 0:
        return _trace_output(value, second - 1)
    axes = value.gatype.derive
    reciprocal = _metric_reciprocal(axes.axis(first), axes.axis(second))
    # summed through `|`, the pairing has the orientation of a product of the two axes
    pairing = value.gatype.orientations[first] * value.gatype.orientations[second]
    return _paired(value, first - 1, second - 1, reciprocal, pairing)


def _paired(value: Extensor, first: int, second: int, reciprocal: Extensor, pairing: Orientation) -> Extensor:
    """Feed the second input the reciprocal of each blade of the first, then sum the diagonal. The
    output's orientation multiplies with that of the pairing the sum runs through."""
    gatype = value.gatype
    axes = [gatype.derive.axis(axis + 1) for axis in range(gatype.arity)]
    axes[second] = reciprocal
    raised = value(*axes)
    offset = raised.ndim + 1
    kernel = raised.context.axes_trace(raised._kernel, offset + first, offset + second)
    return Extensor._from_prepared_kernel(raised.context, gatype.derive.paired(first, second, pairing), kernel)


def _trace_output(value: Extensor, axis: int) -> Extensor:
    result_type = value.gatype.derive.traced(axis)
    matched = value.cast(value.input_subspaces[axis])
    kernel = matched.context.matrix_trace(
        matched._kernel, axis1=matched.ndim, axis2=matched.ndim + axis + 1,
        scalar_axis=matched.ndim,
    )
    return Extensor._from_prepared_kernel(matched.context, result_type, kernel)


@lru_cache(maxsize=None)
def _metric_reciprocal(first: GAType, second: GAType) -> Extensor:
    """The exact map sending each blade of the first axis's space to its reciprocal under the
    space's metric, as an element of the second axis."""
    space = first.output_subspace
    metric = first.metric.default._kernel.values[0]                         # [blades, blades]
    if not metric.any(axis=0).all():
        raise TypeError(f"contract needs an invertible metric; {space} has a null blade")
    return space.algebra.operator.build(
        (second.output_subspace, space), (second.orientation, first.orientation), SymbolicKernel(metric).inverse(),
    )


@lru_cache(maxsize=None)
def _incidence_reciprocal(first_axis: GAType, second_axis: GAType) -> Extensor:
    """The exact map sending each blade of the first space to the blade of the second that it
    meets in one, and every other blade of the first in zero."""
    first, second = first_axis.output_subspace, second_axis.output_subspace
    algebra = first.algebra
    incidence = algebra.operator.regressive(first_axis, second_axis)           # Scalar <- (first, second)
    pairing = incidence.cast(algebra.subspace.scalar())._kernel.values[0]       # [first blades, second blades]
    if pairing.shape[0] != pairing.shape[1] or np.linalg.matrix_rank(pairing) < len(first):
        raise TypeError(f"trace pairs inputs of complementary spaces; {first} and {second} pair through the metric: contract them")
    return algebra.operator.build(
        (second, first), (second_axis.orientation, first_axis.orientation), SymbolicKernel(pairing).inverse(),
    )
