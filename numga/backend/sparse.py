"""Sparse contraction (`execution="sparse"`) unrolls the nonzero entries of an exact `SymbolicKernel`
into straight-line scalar-weighted products of operand slices, skipping the structural zeros
pervasive in geometric, exterior, regressive, and sandwich product tables. Sparsity is determined
strictly from static `ExactContext` kernels; concrete runtime operators delegate immediately to
`execute_dense_bind` so numeric arrays are never inspected dynamically, keeping traced execution
graphs static under `jax.jit` and `torch.compile`.

For each `(xp, dtype, kernel, plan)` signature, `compile_sparse_bind` enumerates the nonzero
coordinates of `kernel`, resolves subspace row mappings and orientation signs from each
`AxisTransform` once, and groups surviving `(coefficient, operand_rows)` terms by their retained
structural coordinates. Input axes carried by higher-arity operands (`free_shape`) broadcast together
before a single final transpose restores the canonical `(batch..., output, *remaining_inputs)` order.

Term evaluation adapts to the target array namespace. On `numpy`, each multi-factor term evaluates
through a contraction-free broadcast `einsum` into a preallocated output array, whereas traced and
accelerator namespaces (`jax.numpy`, `TorchNamespace`) evaluate reshaped elementwise products and
`stack` every output component in one expression so compilers such as XLA fuse independent blade
polynomials without sequential buffer updates. Trade-offs between dense contraction and unrolled
sparse execution are measured in `docs/internals.md` (section 6).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import lru_cache
from math import prod
from string import ascii_letters
from typing import TYPE_CHECKING, Any

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
    """Exact operator kernels dispatch to cached straight-line term evaluators, while concrete targets
    delegate to `execute_dense_bind`."""
    if not target.context.is_exact:
        return execute_dense_bind(context, target, operands, plan)
    execute = compile_sparse_bind(context.xp, context.dtype, target._kernel, plan)
    return execute(tuple(operands[axis] for axis in plan.axes))


@lru_cache(maxsize=None)
def compile_sparse_bind(
    xp: Any, dtype: np.dtype, kernel: SymbolicKernel, plan: BindingPlan,
) -> Callable:
    """Enumerating the nonzero entries of `kernel` compiles a straight-line evaluator grouped by
    retained structural coordinates."""
    retained = (0,) + tuple(
        axis + 1 for axis in range(plan.target_gatype.arity) if axis not in plan.axes
    )
    retained_shape = tuple(kernel.shape[dimension] for dimension in retained)
    groups = defaultdict(list)
    for coordinate, coefficient in np.ndenumerate(kernel.values):
        if not coefficient:
            continue
        coefficient = float(coefficient)
        rows = []
        for binding in plan.bindings:
            required_row = coordinate[binding.axis + 1]
            row = binding.transform.target_from_source[required_row]
            if row is None:
                break
            coefficient *= (
                binding.transform.source.signs[row]
                * binding.transform.target.signs[required_row]
            )
            rows.append(row)
        else:
            groups[tuple(coordinate[dimension] for dimension in retained)].append(
                (dtype.type(coefficient), tuple(rows))
            )
    terms_by_output = tuple(groups.items())

    free_shape = tuple(
        len(dimension) for binding in plan.bindings
        for dimension in binding.operand_gatype.input_subspaces
    )
    reshapes, labels, permutation = [], [], [len(free_shape)]
    cursor, retained_cursor = 0, len(free_shape) + 1
    bindings = {binding.axis: binding for binding in plan.bindings}
    for axis in range(plan.target_gatype.arity):
        binding = bindings.get(axis)
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
            coefficient: Any, arrays: tuple[Any, ...], shapes: tuple,
        ) -> Any:
            # Repeating no labels across inputs turns np.einsum into a fused broadcast product.
            return np.einsum(expression, coefficient, *arrays)

        def assemble(
            operands: tuple[Extensor, ...], shapes: tuple, component_shape: tuple[int, ...],
        ) -> Any:
            result = np.zeros(component_shape + retained_shape, dtype=dtype)
            for coordinate, terms in terms_by_output:
                result[(Ellipsis,) + coordinate] = component(terms, operands, shapes)
            return result
    else:
        def term(
            coefficient: Any, arrays: tuple[Any, ...], shapes: tuple,
        ) -> Any:
            return prod((array.reshape(shape) for array, shape in zip(arrays, shapes)), start=coefficient)

        # Stacking independent components lets XLA fuse their evaluation without sequential in-place updates.
        ordered_terms = tuple(groups.get(coordinate, ()) for coordinate in np.ndindex(retained_shape))
        if ordered_terms:
            def assemble(
                operands: tuple[Extensor, ...], shapes: tuple, component_shape: tuple[int, ...],
            ) -> Any:
                return xp.stack(tuple(
                    xp.broadcast_to(component(terms, operands, shapes), component_shape)
                    for terms in ordered_terms
                ), axis=-1).reshape(component_shape + retained_shape)
        else:
            def assemble(
                operands: tuple[Extensor, ...], shapes: tuple, component_shape: tuple[int, ...],
            ) -> Any:
                return xp.zeros(component_shape + retained_shape, dtype=dtype)

    def component(
        terms: Any, operands: tuple[Extensor, ...], shapes: tuple,
    ) -> Any:
        return sum((term(coefficient, tuple(
            operand._kernel[(Ellipsis, row) + indices]
            for operand, row, indices in zip(operands, rows, row_indices)
        ), shapes) for coefficient, rows in terms), start=dtype.type(0))

    def execute(operands: tuple[Extensor, ...]) -> Any:
        batch_shape = np.broadcast_shapes(*(operand.shape for operand in operands))
        shapes = tuple(operand.shape + reshape for operand, reshape in zip(operands, reshapes))
        result = assemble(operands, shapes, batch_shape + free_shape)
        offset = len(batch_shape)
        return xp.transpose(
            result, tuple(range(offset)) + tuple(offset + dimension for dimension in permutation),
        )

    return execute
