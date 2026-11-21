"""Trace and contraction reduce a multilinear extensor across two axes, leaving an extensor over its
remaining arguments. Axes are numbered with the output as axis `0` and the inputs as axes `1` through
`arity`. Every non-scalar output and every input counts as an indexed axis, so both a unary map
`V <- V` and a bilinear form `Scalar <- (V, V)` carry exactly two indexed axes; when an extensor has
only two indexed axes, `first` and `second` may be omitted.

`trace` pairs through the metric-free regressive product `&`. Two inputs on complementary spaces pair
via `&` with the extrinsic factor placed first (`e_l & e^k`, or `e^k & e_l` when only the second is
extrinsic), making the trace of a bilinear form over oppositely oriented complementary spaces
independent of input order even in even dimensions where swapping `&` factors changes sign. Pairing
the output against an input on the same space yields the standard basis-independent matrix trace,
matching `(f & Antivector).trace(1, 2) == f.trace()`.

`contract` pairs through the space's canonical inner product. Two inputs on the same space pair via
`Space.scalar_product(Space.reverse())` (`V | V` on vectors), summing `operator(e_k, e^k)` over basis
blades and their metric reciprocals and refusing spaces that contain null blades. Pairing the output
against an input on the same space cancels the metric against its inverse and again yields the matrix
trace for any metric, matching `(Vector | f).contract(1, 2) == f.contract()` whenever `Vector`'s
metric is invertible. Trace and contraction are developed in `docs/conventions.md` and
`docs/tutorial/extensors/12-trace-and-contraction.md`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from numga.extensor import Extensor
from numga.extensions._linalg import _reading
from numga.gatype import GAType, Orientation
from numga.operator.kernel import SymbolicKernel
from numga.units import contract_unit, contract_unnamed_unit, trace_unit, trace_unnamed_unit

Extensor.trace.overload(1).set_unit(trace_unnamed_unit)
Extensor.trace.overload(3).set_unit(trace_unit)
Extensor.contract.overload(1).set_unit(contract_unnamed_unit)
Extensor.contract.overload(3).set_unit(contract_unit)


@Extensor.trace.register(lambda t: True)
def trace_unnamed(operator: Extensor) -> Extensor:
    """Omitting axis indices pairs the extensor's sole two indexed axes through the regressive
    product."""
    return trace(operator, *operator.gatype.axis_roles.pair)


@Extensor.trace.overload(3).register(lambda t: True)
def trace(operator: Extensor, first: int, second: int) -> Extensor:
    """Tracing axes `first` and `second` sums `operator` across dual complementary basis pairs via `&`
    (with the extrinsic factor first) when both axes are inputs, or computes the basis-independent
    matrix trace when pairing the output against a matching input (`(Plane & f(Point)).trace(1, 2) ==
    f.trace()` in `PGA`)."""
    first, second = operator.gatype.axis_roles.paired(first, second)
    if first == 0:
        return _trace_output(operator, second - 1)
    axes = operator.gatype.derive
    reciprocal = _incidence_reciprocal(axes.axis(first), axes.axis(second))
    # Summing through `&` multiplies the output orientation by the regressive pairing orientation.
    pairing = (operator.gatype.orientations[first] * operator.gatype.orientations[second]).flipped
    return _paired(operator, first - 1, second - 1, reciprocal, pairing)


@Extensor.contract.register(lambda t: True)
def contract_unnamed(operator: Extensor) -> Extensor:
    """Omitting axis indices pairs the extensor's sole two indexed axes through the inner product."""
    return contract(operator, *operator.gatype.axis_roles.pair)


@Extensor.contract.overload(3).register(lambda t: True)
def contract(operator: Extensor, first: int, second: int) -> Extensor:
    """Contracting axes `first` and `second` sums `operator(e_k, e^k)` over basis blades and their
    metric reciprocals under `Space.scalar_product(Space.reverse())` when both axes are inputs on a
    non-null space, or computes the matrix trace when pairing the output against an input; for example,
    `(Vector * f(Vector)).contract(1, 2)` yields the vector derivative (trace plus curl bivector) of
    `f`."""
    first, second = operator.gatype.axis_roles.contracted(first, second)
    if first == 0:
        return _trace_output(operator, second - 1)
    axes = operator.gatype.derive
    reciprocal = _metric_reciprocal(axes.axis(first), axes.axis(second))
    # Summing through `|` multiplies the output orientation by the inner-product orientation.
    pairing = operator.gatype.orientations[first] * operator.gatype.orientations[second]
    return _paired(operator, first - 1, second - 1, reciprocal, pairing)


def _paired(operator: Extensor, first_input: int, second_input: int, reciprocal: Extensor, pairing: Orientation) -> Extensor:
    # Substitute the reciprocal map into second_input so first_input and second_input align along a diagonal.
    gatype = operator.gatype
    axes = [gatype.derive.axis(axis + 1) for axis in range(gatype.arity)]
    axes[second_input] = reciprocal
    raised = operator(*axes)
    offset = raised.ndim + 1
    kernel = raised.context.axes_trace(raised._kernel, offset + first_input, offset + second_input)
    return Extensor._from_prepared_kernel(raised.context, gatype.derive.paired(first_input, second_input, pairing), kernel)


def _trace_output(operator: Extensor, input_axis: int) -> Extensor:
    result_type = operator.gatype.derive.traced(input_axis)
    matched = operator.cast(operator.input_subspaces[input_axis])
    kernel = matched.context.matrix_trace(
        matched._kernel, axis1=matched.ndim, axis2=matched.ndim + input_axis + 1,
        scalar_axis=matched.ndim,
    )
    return Extensor._from_prepared_kernel(matched.context, result_type, kernel)


@lru_cache(maxsize=None)
def _metric_reciprocal(first: GAType, second: GAType) -> Extensor:
    space = first.output_subspace
    metric = first.metric.default._kernel.values[0]                           # [blades, blades]
    if not metric.any(axis=0).all():
        raise TypeError(f"contract needs an invertible metric; {space} has a null blade")
    return space.algebra.operator.build(
        (second.output_subspace, space), (second.orientation, first.orientation), SymbolicKernel(metric).inverse(),
    )


@lru_cache(maxsize=None)
def _incidence_reciprocal(first_axis: GAType, second_axis: GAType) -> Extensor:
    first, second = first_axis.output_subspace, second_axis.output_subspace
    algebra = first.algebra
    pairing = _reading(first_axis, second_axis)._kernel.values[0]             # [first blades, second blades]
    if pairing.shape[0] != pairing.shape[1] or np.linalg.matrix_rank(pairing) < len(first):
        raise TypeError(f"trace pairs inputs of complementary spaces; {first} and {second} pair through the metric: contract them")
    return algebra.operator.build(
        (second, first), (second_axis.orientation, first_axis.orientation), SymbolicKernel(pairing).inverse(),
    )
