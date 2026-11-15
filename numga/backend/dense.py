"""Dense contraction executes a `BindingPlan` over any NumPy-compatible array namespace (`context.xp`).
Every concrete `Extensor` stores leading batch axes ahead of trailing structural axes
`(output, *inputs)`, so binding a subset of a map's input axes contracts each bound structural axis
against the output axis of its operand and splices any remaining input axes of a higher-arity operand
in place while broadcasting leading batch shapes on the left.

Multi-operand bindings decompose into sequential pairwise contractions whose static axis wiring is
cached per `(xp, plan, order)`. Rather than contracting axes in fixed positional order,
`ordered_binding` sorts bound axes at call time from smallest operand batch product to largest
(breaking ties from last bound axis to first), so small-batch operators such as per-camera motors
contract into the kernel before high-volume point batches expand the intermediate tensor.

Subspace embeddings, projections, and blade sign adjustments represented by non-exact `AxisTransform`
matrices apply to an operand's output axis immediately before contraction. Binary matrix-matrix and
matrix-vector contractions dispatch directly to `xp.matmul`, while general arities compile an
ellipsis `einsum` expression (`optimize=True` on traced and GPU backends, direct path-free evaluation
on `numpy` where path planning dominates small blade kernels). Kernel layout and dense contraction
strings are described in `docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache, partial
from math import prod
from string import ascii_letters
from typing import TYPE_CHECKING, Any

import numpy as np

from numga.binding import AxisTransform, AxisTransformKind, BindingPlan
from numga.gatype import GAType
from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.backend.context import Context
    from numga.extensor import Extensor


def execute_dense_bind(
    context: Context,
    target: Extensor,
    operands: Mapping[int, Extensor],
    plan: BindingPlan,
) -> Any:
    """Lowering `target` into `context` contracts all bound `operands` with left batch broadcasting."""
    kernels = {axis: operand._kernel for axis, operand in operands.items()}
    return ordered_binding(context.xp, plan)(context.lower(target)._kernel, kernels)


@lru_cache(maxsize=None)
def ordered_binding(xp: Any, plan: BindingPlan) -> Callable:
    """Pairwise contractions run from smallest operand batch product to largest, breaking ties from
    last bound axis to first."""
    structural = {binding.axis: len(binding.operand_gatype.subspaces) for binding in plan.bindings}
    last_first = tuple(binding.axis for binding in reversed(plan.bindings))
    orders = {}

    def batch(kernel: Any, axis: int) -> int:
        return prod(kernel.shape[:kernel.ndim - structural[axis]])

    def contract(target: Any, kernels: Mapping[int, Any]) -> Any:
        order = tuple(sorted(last_first, key=lambda axis: batch(kernels[axis], axis)))
        steps = orders.get(order)
        if steps is None:
            steps = orders[order] = binding_steps(xp, plan, order)
        for axis, apply in steps:
            target = apply(target, kernels[axis])
        return target
    return contract


@lru_cache(maxsize=None)
def binding_steps(
    xp: Any, plan: BindingPlan, order: tuple[int, ...],
) -> tuple:
    """Each step in `order` pairs an operand axis with its compiled contractor, tracking how
    higher-arity operands widen the intermediate structural axes."""
    bound = {binding.axis: binding for binding in plan.bindings}
    # Each target input axis initially spans one structural axis and widens to operand.arity when bound.
    widths = [1] * plan.target_gatype.arity
    current_axes = (plan.result_subspaces[0],) + plan.target_gatype.input_subspaces
    steps = []
    for axis in order:
        binding = bound[axis]
        target_axis = 1 + sum(widths[:axis])
        operand_type = binding.operand_gatype
        apply = _contractor(
            xp,
            current_axes,
            target_axis,
            (binding.transform.target,) + operand_type.input_subspaces,
        )
        if binding.transform.kind is not AxisTransformKind.EXACT:
            apply = _with_output_transform(xp, apply, operand_type, binding.transform)
        steps.append((axis, apply))
        current_axes = (
            current_axes[:target_axis]
            + operand_type.input_subspaces
            + current_axes[target_axis + 1 :]
        )
        widths[axis] = len(operand_type.input_subspaces)
    return tuple(steps)


def _with_output_transform(
    xp: Any,
    apply: Callable,
    gatype: GAType,
    transform: AxisTransform,
) -> Callable:
    def converted(target: Any, operand: Any) -> Any:
        return apply(target, transform_output(xp, operand, gatype, transform))

    return converted


def transform_output(
    xp: Any,
    kernel: Any,
    gatype: GAType,
    transform: AxisTransform,
) -> Any:
    """Non-identity `transform` matrices remap the operand's leading structural axis immediately after
    its batch axes."""
    if transform.kind is AxisTransformKind.EXACT:
        return kernel
    batch_ndim = kernel.ndim - len(gatype.subspaces)
    matrix = xp.asarray(transform.coordinate_matrix, dtype=kernel.dtype)
    transformed = xp.tensordot(matrix, kernel, axes=(1, batch_ndim))
    return xp.moveaxis(transformed, 0, batch_ndim)


def transform_axis(
    xp: Any,
    kernel: Any,
    structural_ndim: int,
    axis: int,
    transform: AxisTransform,
) -> Any:
    """Non-identity `transform` matrices remap structural `axis` in place while preserving leading
    batch axes."""
    if transform.kind is AxisTransformKind.EXACT:
        return kernel
    batch_ndim = kernel.ndim - structural_ndim
    physical_axis = batch_ndim + axis
    matrix = xp.asarray(transform.coordinate_matrix, dtype=kernel.dtype)
    transformed = xp.tensordot(matrix, kernel, axes=(1, physical_axis))
    return xp.moveaxis(transformed, 0, physical_axis)


@lru_cache(maxsize=None)
def _contractor(
    xp: Any,
    target_axes: tuple[SubSpace, ...],
    target_axis: int,
    operand_axes: tuple[SubSpace, ...],
) -> Callable:
    if len(target_axes) == 2 and target_axis == 1:
        if len(operand_axes) == 2:
            return xp.matmul
        if len(operand_axes) == 1:
            def apply_vector(matrix: Any, vector: Any) -> Any:
                return xp.matmul(matrix, vector[..., None])[..., 0]
            return apply_vector

    target_labels = ascii_letters[:len(target_axes)]
    operand_input_labels = ascii_letters[
        len(target_axes):len(target_axes) + len(operand_axes) - 1
    ]
    operand_labels = target_labels[target_axis] + operand_input_labels
    output_labels = (
        target_labels[:target_axis]
        + operand_input_labels
        + target_labels[target_axis + 1 :]
    )
    expression = f"...{target_labels},...{operand_labels}->...{output_labels}"
    if xp is not np:
        return partial(xp.einsum, expression, optimize=True)

    # Two operands admit a single contraction order, so NumPy's direct einsum avoids path-planning
    # overhead that outweighs contracting small blade kernels.
    return partial(np.einsum, expression)
