"""Compile full application once; warm calls only execute and wrap arrays."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

from numga.backend.dense import binding_steps
from numga.binding import BindingPlan, TypeRules

if TYPE_CHECKING:
    from numga.backend.context import Context
    from numga.gatype import GAType
    from numga.operator.kernel import SymbolicKernel
    from .extensor import Extensor


def unary_application(
    cls: type[Extensor], context: Context, gatype: GAType,
    operand_context: Context, operand_type: GAType,
) -> Callable[[Extensor, Extensor], Extensor]:
    # A flat key avoids constructing a nested signature on every unary call.
    owner = operand_context if context.is_exact else context
    key = (cls, gatype, context.key, operand_context.key, operand_type)
    try:
        return owner._applications[key]
    except KeyError:
        execute = application(cls, context, gatype, ((operand_context, operand_type),))
        owner._applications[key] = execute
        return execute


def application(
    cls: type[Extensor], target_context: Context, target_type: GAType,
    signature: tuple[tuple[Context, GAType], ...],
    equality_groups: tuple[tuple[int, ...], ...] = (),
) -> Callable[..., Extensor]:
    from numga.backend.context import binding_context

    context = next((c for c in (target_context,) + tuple(c for c, _ in signature)
                    if not c.is_exact), target_context)
    key = (cls, target_type, target_context.key,
           tuple((c.key, t) for c, t in signature), equality_groups)
    try:
        return context._applications[key]
    except KeyError:
        binding_context(target_context, tuple(c for c, _ in signature))
        execute = compile_application(cls, context, target_context, target_type, signature, equality_groups)
        context._applications[key] = execute
        return execute


def compile_application(
    cls: type[Extensor], context: Context, target_context: Context, target_type: GAType,
    signature: tuple[tuple[Context, GAType], ...], equality_groups: tuple[tuple[int, ...], ...],
) -> Callable[..., Extensor]:
    if len(signature) != target_type.arity:
        raise ValueError(
            f"full application of arity-{target_type.arity} Extensor requires "
            f"{target_type.arity} operands, got {len(signature)}"
        )
    if context.is_exact or not signature:
        # Symbolic construction retains the exact binding implementation.
        return lambda target, *operands: target.bind(*operands)

    plan = BindingPlan.from_types(
        target_type, tuple((slot, t) for slot, (_, t) in enumerate(signature)),
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
        def sparse_executor(kernel: SymbolicKernel) -> Callable[..., Any]:
            if restrict:
                kernel = kernel.take(indices, axis=0)
            return compile_sparse_bind(xp, dtype, kernel, plan)

        exact_slots = tuple(i for i, (c, _) in enumerate(signature) if c.is_exact)
        if exact_slots:
            def execute(target: Extensor, *operands: Extensor) -> Extensor:
                prepared = list(operands)
                for slot in exact_slots:
                    prepared[slot] = context.lower(operands[slot])
                return wrap(context, result_type, sparse_executor(target._kernel)(tuple(prepared)))
        else:
            def execute(target: Extensor, *operands: Extensor) -> Extensor:
                return wrap(context, result_type, sparse_executor(target._kernel)(operands))
        return execute

    steps = binding_steps(xp, plan)
    # Preparation is selected statically, not through context.lower per call.
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

    contractions = []
    for slot, contract in steps:
        if signature[slot][0].is_exact:
            def lowered(
                left: Any, right: SymbolicKernel,
                contract: Callable[[Any, Any], Any] = contract,
            ) -> Any:
                return contract(left, context.materialize(right))
            contract = lowered
        contractions.append((slot, contract))

    if len(contractions) == 1:
        _, contract = contractions[0]
        if prepare_target is None:
            def execute(target: Extensor, operand: Extensor) -> Extensor:
                return wrap(context, result_type, contract(target._kernel, operand._kernel))
        else:
            def execute(target: Extensor, operand: Extensor) -> Extensor:
                return wrap(context, result_type, contract(prepare_target(target._kernel), operand._kernel))
    else:
        def execute(target: Extensor, *operands: Extensor) -> Extensor:
            kernel = target._kernel
            if prepare_target is not None:
                kernel = prepare_target(kernel)
            for slot, contract in contractions:
                kernel = contract(kernel, operands[slot]._kernel)
            return wrap(context, result_type, kernel)
    return execute
