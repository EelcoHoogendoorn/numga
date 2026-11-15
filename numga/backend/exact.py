"""Every `Algebra` constructs its basis blades, identity maps, and bilinear product tables inside a
dedicated `ExactContext` backed by `SymbolicKernel`. Operator tables start in `int8` and stay integer
under composition, transposition, halving, and unimodular inversion (promoting to floating point only
when scaled by non-integer real numbers), while `dtype` reports `object` to distinguish unbatched
symbolic storage from fixed-width numeric array contexts.

Purely exact bindings contract in reverse axis order directly on `SymbolicKernel` instances,
splicing higher-arity operand input axes into the resulting structural layout. When an exact
operator meets a concrete array `Extensor`, `binding_context` selects the array context and
materializes the exact kernel into backend storage rather than executing in `ExactContext`; mutating
batch updates (`functional_set`, `functional_add`) are rejected because exact extensors have no
batch dimensions. Exact context semantics and cached table materialization are described in
`docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from numga.binding import AxisTransform, AxisTransformKind, BindingPlan
from numga.operator.kernel import SymbolicKernel

from .context import Context

if TYPE_CHECKING:
    from types import ModuleType

    from numga.algebra import Algebra
    from numga.extensor import Extensor


class ExactContext(Context):
    """Algebra-owned execution policy evaluates unbatched `SymbolicKernel` expressions in exact arithmetic."""

    __slots__ = ("_algebra",)

    def __init__(self, algebra: Algebra) -> None:
        object.__setattr__(self, "_algebra", algebra)
        super().__init__()

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ExactContext instances are immutable")

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(object)

    @property
    def key(self) -> tuple[object, ...]:
        return ("exact",)

    @property
    def xp(self) -> ModuleType:
        return np

    @property
    def is_exact(self) -> bool:
        return True

    def prepare_kernel(self, coefficients: object) -> SymbolicKernel:
        return coefficients if isinstance(coefficients, SymbolicKernel) else SymbolicKernel(coefficients)

    def reciprocal(self, kernel: SymbolicKernel) -> SymbolicKernel:
        return kernel.reciprocal()

    def scalar_kernel(self, scalar: Any) -> SymbolicKernel:
        return SymbolicKernel((scalar,))

    def matrix_inverse(self, kernel: SymbolicKernel) -> SymbolicKernel:
        return kernel.inverse()

    def matrix_trace(self, kernel: SymbolicKernel, *, axis1: int = -2, axis2: int = -1, scalar_axis: int = -1) -> SymbolicKernel:
        return kernel.trace(axis1, axis2).expand_dims(scalar_axis)

    def axes_trace(self, kernel: SymbolicKernel, axis1: int, axis2: int) -> SymbolicKernel:
        return kernel.trace(axis1, axis2)

    def solve(self, matrix: SymbolicKernel, rhs: SymbolicKernel) -> SymbolicKernel:
        return matrix.inverse().tensordot(rhs, axes=(1, 0))

    def first_last_input(self, kernel: SymbolicKernel, batch_ndim: int) -> SymbolicKernel:
        return kernel.first_last_input()

    def prepare_scalar(self, scalar: object) -> object:
        if isinstance(scalar, Real):
            return scalar
        raise TypeError(f"Extensor scalars must be real numbers; got {type(scalar).__name__}")

    def execute_bind(
        self,
        target: Extensor,
        operands: Mapping[int, Extensor],
        plan: BindingPlan,
    ) -> SymbolicKernel:
        kernel = target.kernel
        for binding in reversed(plan.bindings):
            operand = operands[binding.axis]
            transformed = _transform_output(operand.kernel, binding.transform)
            kernel = _contract(
                kernel,
                target_axis=binding.axis + 1,
                operand=transformed,
            )
        return kernel

    def transform_axis(
        self,
        kernel: SymbolicKernel,
        structural_ndim: int,
        axis: int,
        transform: AxisTransform,
    ) -> SymbolicKernel:
        return _transform_axis(kernel, axis, transform)

    def functional_set(
        self, kernel: SymbolicKernel, index: object, value: object
    ) -> NoReturn:
        raise TypeError("exact Extensors do not have mutable batch storage")

    def functional_add(
        self, kernel: SymbolicKernel, index: object, value: object
    ) -> NoReturn:
        raise TypeError("exact Extensors do not have mutable batch storage")

    def __repr__(self) -> str:
        return f"ExactContext(algebra={self.algebra!r})"


def _transform_output(
    kernel: SymbolicKernel,
    transform: AxisTransform,
) -> SymbolicKernel:
    return _transform_axis(kernel, 0, transform)


def _transform_axis(
    kernel: SymbolicKernel,
    axis: int,
    transform: AxisTransform,
) -> SymbolicKernel:
    if transform.kind is AxisTransformKind.EXACT:
        return kernel
    matrix = SymbolicKernel(np.asarray(transform.coordinate_matrix))
    return matrix.tensordot(kernel, axes=(1, axis)).moveaxis(0, axis)


def _contract(
    target: SymbolicKernel,
    target_axis: int,
    operand: SymbolicKernel,
) -> SymbolicKernel:
    contracted = target.tensordot(operand, axes=(target_axis, 0))
    target_remaining = target.ndim - 1
    operand_inputs = operand.ndim - 1
    # Tensordot appends remaining operand input axes at the end; permute splices them into target_axis.
    permutation = (
        tuple(range(target_axis))
        + tuple(range(target_remaining, target_remaining + operand_inputs))
        + tuple(range(target_axis, target_remaining))
    )
    if permutation != tuple(range(contracted.ndim)):
        contracted = contracted.transpose(permutation)
    return contracted
