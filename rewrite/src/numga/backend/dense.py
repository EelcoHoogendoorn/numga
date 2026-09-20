"""Array-namespace-parametric dense execution of a ``BindingPlan``."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache, partial
from string import ascii_letters
from typing import TYPE_CHECKING, Any, Callable

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

    current = context.lower(target)._kernel
    for slot, apply in binding_steps(context.xp, plan):
        current = apply(current, operands[slot]._kernel)
    return current


@lru_cache(maxsize=None)
def binding_steps(
    xp: Any, plan: BindingPlan,
) -> tuple[tuple[int, Callable[[Any, Any], Any]], ...]:
    """Resolve contractions and necessary coordinate conversions once."""

    current_axes = (plan.result_subspaces[0],) + plan.target_gatype.input_subspaces
    steps = []
    for binding in reversed(plan.bindings):
        target_axis = binding.slot + 1
        operand_type = binding.operand_gatype
        apply = _contractor(
            xp,
            current_axes,
            target_axis,
            (binding.transform.target,) + operand_type.input_subspaces,
        )
        if binding.transform.kind is not AxisTransformKind.EXACT:
            apply = _with_output_transform(xp, apply, operand_type, binding.transform)
        steps.append((binding.slot, apply))
        current_axes = (
            current_axes[:target_axis]
            + operand_type.input_subspaces
            + current_axes[target_axis + 1 :]
        )
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
    return partial(xp.einsum, expression, optimize=True)
