"""Unroll exact nonzero terms, as in the original sparse operators.

Only symbolic kernels supply sparsity. Computed maps use the dense executor;
numeric arrays are never inspected to decide which terms to execute.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from math import prod
from string import ascii_letters
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from numga.binding import BindingPlan
from numga.operator.kernel import SymbolicKernel

from .dense import execute_dense_bind

if TYPE_CHECKING:
    from numga.backend.context import Context
    from numga.extensor import Extensor


def execute_sparse_bind(
    context: Context, target: Extensor,
    operands: Mapping[int, Extensor], plan: BindingPlan,
) -> Any:
    if not target.context.is_exact:
        return execute_dense_bind(context, target, operands, plan)
    execute = _executor(context.xp, context.dtype, target._kernel, plan)
    return execute(tuple(operands[slot] for slot in plan.slots))


@lru_cache(maxsize=None)
def _executor(
    xp: Any, dtype: np.dtype, kernel: SymbolicKernel, plan: BindingPlan,
) -> Callable[[tuple[Extensor, ...]], Any]:
    """Group sparse terms and resolve axis/sign bookkeeping once."""

    retained = (0,) + tuple(
        slot + 1 for slot in range(plan.target_gatype.arity) if slot not in plan.slots
    )
    retained_shape = tuple(kernel.shape[axis] for axis in retained)
    groups = defaultdict(list)
    for coordinate, coefficient in np.ndenumerate(kernel.to_object_array()):
        if not coefficient:
            continue
        rows = []
        for binding in plan.bindings:
            required_row = coordinate[binding.slot + 1]
            row = binding.transform.target_from_source[required_row]
            if row is None:
                break
            coefficient *= (
                binding.transform.source.signs[row]
                * binding.transform.target.signs[required_row]
            )
            rows.append(row)
        else:
            groups[tuple(coordinate[axis] for axis in retained)].append(
                (dtype.type(float(coefficient)), tuple(rows))
            )
    terms_by_output = tuple(groups.items())

    free_shape = tuple(
        len(axis) for binding in plan.bindings
        for axis in binding.operand_gatype.input_subspaces
    )
    reshapes, labels, permutation = [], [], [len(free_shape)]
    cursor, retained_cursor = 0, len(free_shape) + 1
    bindings = {binding.slot: binding for binding in plan.bindings}
    for slot in range(plan.target_gatype.arity):
        binding = bindings.get(slot)
        if binding is None:
            permutation.append(retained_cursor)
            retained_cursor += 1
        else:
            width = binding.operand_gatype.arity
            reshapes.append(
                (1,) * cursor + free_shape[cursor:cursor + width]
                + (1,) * (len(free_shape) - cursor - width)
            )
            labels.append("..." + ascii_letters[cursor:cursor + width])
            permutation.extend(range(cursor, cursor + width))
            cursor += width
    expression = "," + ",".join(labels) + "->..." + ascii_letters[:len(free_shape)]
    row_indices = tuple((slice(None),) * binding.operand_gatype.arity for binding in plan.bindings)

    if xp is np:
        def term(
            coefficient: Any, arrays: tuple[Any, ...], shapes: tuple[tuple[int, ...], ...],
        ) -> Any:
            # No contracted labels remain: this is a fused broadcast product.
            return np.einsum(expression, coefficient, *arrays)

        def assign(array: Any, index: tuple[object, ...], value: Any) -> Any:
            array[index] = value
            return array
    else:
        def term(
            coefficient: Any, arrays: tuple[Any, ...], shapes: tuple[tuple[int, ...], ...],
        ) -> Any:
            return prod((array.reshape(shape) for array, shape in zip(arrays, shapes)), start=coefficient)

        def assign(array: Any, index: tuple[object, ...], value: Any) -> Any:
            return array.at[index].set(value)

    def execute(operands: tuple[Extensor, ...]) -> Any:
        batch_shape = np.broadcast_shapes(*(operand.shape for operand in operands))
        shapes = tuple(operand.shape + reshape for operand, reshape in zip(operands, reshapes))
        result = xp.zeros(batch_shape + free_shape + retained_shape, dtype=dtype)
        for coordinate, terms in terms_by_output:
            component = sum(term(coefficient, tuple(
                operand._kernel[(Ellipsis, row) + indices]
                for operand, row, indices in zip(operands, rows, row_indices)
            ), shapes) for coefficient, rows in terms)
            result = assign(result, (Ellipsis,) + coordinate, component)
        offset = len(batch_shape)
        return xp.transpose(
            result, tuple(range(offset)) + tuple(offset + axis for axis in permutation),
        )

    return execute
