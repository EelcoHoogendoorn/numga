"""Calling an arity-`k` `Extensor` on `k` operands resolves to a compiled contraction cached on the
binding `Context` by static algebraic types, backend context keys, and shared-operand identity
groups. Separating `BindingPlan` construction, output-subspace restriction, and contraction planning
from per-call evaluation lets repeated calls contract raw backend kernels directly and wrap the
prepared output without re-validating types. Unary map calls (`arity == 1`) use a flat cache key in
`unary_application` on the concrete operand or target context to avoid constructing nested signature
tuples on the hot path.

Compilation branches statically on field axes, exactness, and execution mode. Any call involving
field axes delegates to `bind_fields`, while purely symbolic exact calls preserve `target.bind`.
When an exact `SymbolicKernel` target evaluates on a concrete context in `"sparse"` execution mode,
`compile_sparse_bind` caches sparse index-and-sign plans per symbolic kernel; dense execution
specializes target materialization, output-blade slicing, and exact-operand lowering at compile
time, dispatching unary maps through `binding_steps` and higher-arity maps through `ordered_binding`.
Compiled call closures and dense versus sparse execution are described in `docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from numga.backend.dense import binding_steps, ordered_binding
from numga.backend.context import binding_context
from numga.binding import BindingPlan, TypeRules

from .fields import bind_fields

if TYPE_CHECKING:
    from numga.backend.context import Context
    from numga.gatype import GAType
    from numga.operator.kernel import SymbolicKernel
    from .extensor import Extensor


def unary_application(
    cls: type[Extensor], target_context: Context, target_type: GAType,
    operand_context: Context, operand_type: GAType,
) -> Callable:
    """Caching unary map evaluation on a flat key owned by the concrete operand or target context
    avoids building nested signature tuples on every call."""
    owner = operand_context if target_context.is_exact else target_context
    key = (cls, target_type, target_context.key, operand_context.key, operand_type)
    try:
        return owner._applications[key]
    except KeyError:
        execute = application(cls, target_context, target_type, ((operand_context, operand_type),))
        owner._applications[key] = execute
        return execute


def application(
    cls: type[Extensor], target_context: Context, target_type: GAType,
    signature: tuple,
    equality_groups: tuple = (),
) -> Callable:
    """Compiling full evaluation once per signature caches the contraction closure on the resolved
    binding context."""
    context = binding_context(target_context, tuple(c for c, _ in signature))
    key = (cls, target_type, target_context.key,
           tuple((c.key, t) for c, t in signature), equality_groups)
    try:
        return context._applications[key]
    except KeyError:
        execute = compile_application(cls, context, target_context, target_type, signature, equality_groups)
        context._applications[key] = execute
        return execute


def compile_application(
    cls: type[Extensor], context: Context, target_context: Context, target_type: GAType,
    signature: tuple, equality_groups: tuple,
) -> Callable:
    """Static target and operand types determine the contraction plan and kernel preparation once per signature."""
    if len(signature) != target_type.arity:
        raise ValueError(
            f"full application of arity-{target_type.arity} Extensor requires "
            f"{target_type.arity} operands, got {len(signature)}"
        )
    if target_type.has_fields or any(t.has_fields for _, t in signature):
        return lambda target, *operands: bind_fields(target, dict(enumerate(operands)))
    if context.is_exact or not signature:
        # Symbolic construction retains the exact binding implementation.
        return lambda target, *operands: target.bind(*operands)

    plan = BindingPlan.from_types(
        target_type, tuple((axis, t) for axis, (_, t) in enumerate(signature)),
        equality_groups,
    )
    result_type = TypeRules.bind(plan)
    wrap = cls._from_prepared_kernel
    xp, dtype = context.xp, context.dtype
    restrict = plan.result_subspaces[0] is not target_type.output_subspace
    indices = plan.output_indices if restrict else ()

    if target_context.is_exact and context.execution == "sparse":
        from numga.backend.sparse import compile_sparse_bind

        @lru_cache(maxsize=None)
        def sparse_executor(kernel: SymbolicKernel) -> Callable:
            if restrict:
                kernel = kernel.take(indices, axis=0)
            return compile_sparse_bind(xp, dtype, kernel, plan)

        exact_axes = tuple(i for i, (c, _) in enumerate(signature) if c.is_exact)
        if exact_axes:
            def execute(target: Extensor, *operands: Extensor) -> Extensor:
                prepared = list(operands)
                for axis in exact_axes:
                    prepared[axis] = context.lower(operands[axis])
                return wrap(context, result_type, sparse_executor(target._kernel)(tuple(prepared)))
        else:
            def execute(target: Extensor, *operands: Extensor) -> Extensor:
                return wrap(context, result_type, sparse_executor(target._kernel)(operands))
        return execute

    # Target slicing and exact-kernel materialization specialize statically at compile time.
    if target_context.is_exact:
        def prepare_target(kernel: SymbolicKernel) -> Any:
            if restrict:
                kernel = kernel.take(indices, axis=0)
            return context.materialize(kernel)
    elif restrict:
        index_array = tuple(indices)
        structural_ndim = len(target_type.subspaces)

        def prepare_target(kernel: Any) -> Any:
            return xp.take(kernel, xp.asarray(index_array), axis=kernel.ndim - structural_ndim)
    else:
        prepare_target = None

    exact_axes = tuple(axis for axis, (c, _) in enumerate(signature) if c.is_exact)
    if len(signature) > 1:
        contract_all = ordered_binding(xp, plan)

        def execute(target: Extensor, *operands: Extensor) -> Extensor:
            kernel = target._kernel if prepare_target is None else prepare_target(target._kernel)
            kernels = {axis: operand._kernel for axis, operand in enumerate(operands)}
            for axis in exact_axes:
                kernels[axis] = context.materialize(kernels[axis])
            return wrap(context, result_type, contract_all(kernel, kernels))
        return execute

    ((_, contract),) = binding_steps(xp, plan, (0,))
    if exact_axes:
        def lowered(left: Any, right: SymbolicKernel, contract: Callable = contract) -> Any:
            return contract(left, context.materialize(right))
        contract = lowered
    if prepare_target is None:
        def execute(target: Extensor, operand: Extensor) -> Extensor:
            return wrap(context, result_type, contract(target._kernel, operand._kernel))
    else:
        def execute(target: Extensor, operand: Extensor) -> Extensor:
            return wrap(context, result_type, contract(prepare_target(target._kernel), operand._kernel))
    return execute
