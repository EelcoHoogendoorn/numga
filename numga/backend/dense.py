"""Array-namespace-parametric dense execution of a ``BindingPlan``."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache, partial
from math import prod
from string import ascii_letters
from typing import TYPE_CHECKING, Any, Callable

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
    """Execute one deterministic atomic bind with left batch broadcasting."""

    kernels = {slot: operand._kernel for slot, operand in operands.items()}
    return ordered_binding(context.xp, plan)(context.lower(target)._kernel, kernels)


@lru_cache(maxsize=None)
def ordered_binding(xp: Any, plan: BindingPlan) -> Callable[[Any, Mapping[int, Any]], Any]:
    """Contract the operands, given by slot, smallest batch first.

    The order of pairwise contractions is chosen per call from the operands' batch sizes: an
    operand with a small batch, such as a motor per camera sandwiching a map per point and
    camera, contracts into the kernel before the large one does, so the large contraction
    meets a kernel that no longer carries the small operand's slots. Equal batches keep the
    order of the slots from last to first.
    """

    structural = {binding.slot: len(binding.operand_gatype.subspaces) for binding in plan.bindings}
    last_first = tuple(binding.slot for binding in reversed(plan.bindings))
    orders: dict[tuple[int, ...], tuple] = {}

    def batch(kernel: Any, slot: int) -> int:
        return prod(kernel.shape[:kernel.ndim - structural[slot]])

    def contract(target: Any, kernels: Mapping[int, Any]) -> Any:
        order = tuple(sorted(last_first, key=lambda slot: batch(kernels[slot], slot)))
        steps = orders.get(order)
        if steps is None:
            steps = orders[order] = binding_steps(xp, plan, order)
        for slot, apply in steps:
            target = apply(target, kernels[slot])
        return target
    return contract


@lru_cache(maxsize=None)
def binding_steps(
    xp: Any, plan: BindingPlan, order: tuple[int, ...],
) -> tuple[tuple[int, Callable[[Any, Any], Any]], ...]:
    """Resolve contractions and necessary coordinate conversions once, binding slots in order."""

    bound = {binding.slot: binding for binding in plan.bindings}
    widths = [1] * plan.target_gatype.arity                 # kernel axes each target slot spans so far
    current_axes = (plan.result_subspaces[0],) + plan.target_gatype.input_subspaces
    steps = []
    for slot in order:
        binding = bound[slot]
        target_axis = 1 + sum(widths[:slot])
        operand_type = binding.operand_gatype
        apply = _contractor(
            xp,
            current_axes,
            target_axis,
            (binding.transform.target,) + operand_type.input_subspaces,
        )
        if binding.transform.kind is not AxisTransformKind.EXACT:
            apply = _with_output_transform(xp, apply, operand_type, binding.transform)
        steps.append((slot, apply))
        current_axes = (
            current_axes[:target_axis]
            + operand_type.input_subspaces
            + current_axes[target_axis + 1 :]
        )
        widths[slot] = len(operand_type.input_subspaces)
    return tuple(steps)


def _with_output_transform(
    xp: Any,
    apply: Callable[[Any, Any], Any],
    gatype: GAType,
    transform: AxisTransform,
) -> Callable[[Any, Any], Any]:
    def converted(target: Any, operand: Any) -> Any:
        return apply(target, transform_output(xp, operand, gatype, transform))

    return converted


def transform_output(
    xp: Any,
    kernel: Any,
    gatype: GAType,
    transform: AxisTransform,
) -> Any:
    """Apply the planned output-axis coordinate map without mutation."""

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
    """Apply a coordinate transform to one trailing structural axis."""

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
) -> Callable[[Any, Any], Any]:
    """Select once from structural axes; batches belong to the array backend."""

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

    # NumPy searches for a contraction path on every call; the path depends on shapes only.
    paths: dict[tuple, list] = {}

    def contract(target: Any, operand: Any) -> Any:
        key = (target.shape, operand.shape)
        path = paths.get(key)
        if path is None:
            path = paths[key] = np.einsum_path(expression, target, operand, optimize="optimal")[0]
        return np.einsum(expression, target, operand, optimize=path)
    return contract
