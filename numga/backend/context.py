"""Coefficient storage, execution, and lowering policies for Extensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Number
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from numga.gatype import GAType
from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.binding import AxisTransform, BindingPlan
    from numga.extensor import Extensor
    from numga.multivector import MultivectorFactory
    from numga.gatype import GATypeFactory
    from numga.operator.kernel import SymbolicKernel
    from numga.subspace import SubSpaceFactory


class Context(ABC):
    """Immutable coefficient/execution policy used by ``Extensor``.

    The algebra-owned exact context constructs expressions; array contexts
    materialize and execute them. ``key`` identifies the storage/execution
    policy only. Algebra ownership is checked separately by identity, which
    keeps compatibility explicit while allowing tracing backends to reconstruct
    contexts from static pytree metadata.
    """

    __slots__ = ("_multivector_factory", "_execution", "_execute_bind", "_applications", "__weakref__")

    def __init__(self, execution: Literal["dense", "sparse"] = "dense") -> None:
        from numga.multivector import MultivectorFactory
        from .dense import execute_dense_bind
        from .sparse import execute_sparse_bind

        object.__setattr__(self, "_applications", {})
        object.__setattr__(self, "_execution", execution)
        object.__setattr__(self, "_execute_bind", {
            "dense": execute_dense_bind, "sparse": execute_sparse_bind,
        }[execution])

        object.__setattr__(
            self,
            "_multivector_factory",
            MultivectorFactory(self),
        )

    @property
    @abstractmethod
    def algebra(self) -> Algebra:
        """The exact algebra object owned by this context."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The effective backend coefficient dtype."""

    @property
    @abstractmethod
    def key(self) -> tuple[object, ...]:
        """Hashable, immutable context identity (excluding algebra)."""

    @property
    def is_exact(self) -> bool:
        """Whether this context stores exact, value-free coefficients."""

        return False

    @property
    def execution(self) -> str:
        return self._execution

    @property
    def multivector(self) -> MultivectorFactory:
        """Nullary-Extensor construction namespace for this context."""

        return self._multivector_factory

    @property
    def gatype(self) -> GATypeFactory:
        return self.algebra.gatype

    @property
    def subspace(self) -> SubSpaceFactory:
        return self.algebra.subspace

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Array namespace used by the dense reference executor."""

    @abstractmethod
    def prepare_kernel(self, value: object) -> Any:
        """Copy a public value into backend storage under this policy."""

    def materialize(self, kernel: SymbolicKernel) -> Any:
        """Backend storage holding the coefficients of an exact kernel."""

        return self.xp.asarray(kernel.materialize(self.dtype))

    def with_dtype(self, dtype: object) -> Context:
        """This storage policy with another coefficient dtype, as complex eigenpairs need."""

        return type(self)(self.algebra, dtype, execution=self.execution)

    def kernel_dtype(self, kernel: Any) -> np.dtype:
        """The NumPy dtype of backend storage."""

        return np.dtype(kernel.dtype)

    def reciprocal(self, kernel: Any) -> Any:
        return 1 / kernel

    def scalar_kernel(self, scalar: Any) -> Any:
        return self.xp.reshape(scalar, (1,))

    def matrix_inverse(self, kernel: Any) -> Any:
        return self.xp.linalg.inv(kernel)

    def matrix_trace(self, kernel: Any, *, axis1: int = -2, axis2: int = -1, scalar_axis: int = -1) -> Any:
        tr = self.xp.trace(kernel, axis1=axis1, axis2=axis2)
        return self.xp.expand_dims(tr, axis=scalar_axis)

    def solve(self, matrix: Any, rhs: Any) -> Any:
        """Solve with a vector RHS, broadcasting its leading batch axes."""

        rhs = self.xp.broadcast_to(rhs, matrix.shape[:-1])
        return self.xp.linalg.solve(matrix, rhs[..., None])[..., 0]

    def generalized_eigh(self, matrix: Any, metric: Any) -> tuple[Any, Any]:
        """Eigenpairs of a Hermitian pencil, reduced to a standard problem by the Cholesky
        factor of the positive-definite metric; eigenvectors are orthonormal in the metric."""

        xp = self.xp
        inverse = xp.linalg.inv(xp.linalg.cholesky(metric))
        adjoint = xp.conj(xp.swapaxes(inverse, -1, -2))
        values, vectors = xp.linalg.eigh(inverse @ matrix @ adjoint)
        return values, adjoint @ vectors

    def generalized_eigvalsh(self, matrix: Any, metric: Any) -> Any:
        return self.generalized_eigh(matrix, metric)[0]

    def generalized_eig(self, matrix: Any, metric: Any) -> tuple[Any, Any]:
        """Eigenpairs of a pencil, reduced to a standard problem by solving with the metric,
        which must be invertible."""

        return self.xp.linalg.eig(self.xp.linalg.solve(metric, matrix))

    def generalized_eigvals(self, matrix: Any, metric: Any) -> Any:
        return self.xp.linalg.eigvals(self.xp.linalg.solve(metric, matrix))

    def is_compatible_with(self, other: object) -> bool:
        return (
            isinstance(other, Context)
            and self.algebra is other.algebra
            and self.key == other.key
        )

    def extensor(
        self,
        gatype_or_subspace: GAType | SubSpace,
        coefficients: object,
    ) -> Extensor:
        from numga.extensor import Extensor

        if isinstance(gatype_or_subspace, GAType):
            gatype = gatype_or_subspace
        elif isinstance(gatype_or_subspace, SubSpace):
            gatype = self.algebra.gatype((gatype_or_subspace,))
        else:
            raise TypeError("extensor construction requires a GAType or SubSpace")
        return Extensor(self, gatype, coefficients)

    def lower(self, value: Extensor) -> Extensor:
        """Represent an Extensor in this coefficient/execution context."""

        if value.context is self:
            return value
        from numga.extensor import Extensor

        if value.algebra is not self.algebra:
            raise ValueError("Extensor and Context belong to different algebras")
        if value.context.is_compatible_with(self):
            return Extensor._from_prepared_kernel(self, value.gatype, value._kernel)
        if value.context.is_exact and not self.is_exact:
            numeric = self.materialize(value._kernel)
            return Extensor._from_prepared_kernel(self, value.gatype, numeric)
        raise ValueError("cannot lower an Extensor from an incompatible Context")

    def execute_bind(
        self,
        target: Extensor,
        operands: Mapping[int, Extensor],
        plan: BindingPlan,
    ) -> Any:
        return self._execute_bind(self, target, operands, plan)

    def transform_axis(
        self,
        kernel: Any,
        structural_ndim: int,
        axis: int,
        transform: AxisTransform,
    ) -> Any:
        from .dense import transform_axis

        return transform_axis(
            self.xp,
            kernel,
            structural_ndim,
            axis,
            transform,
        )

    def prepare_scalar(self, scalar: object) -> Any:
        """Coerce a scalar without silently changing its numeric kind."""

        if not isinstance(scalar, Number) and getattr(scalar, "ndim", None) != 0:
            raise TypeError(f"expected a scalar, got {type(scalar).__name__}")
        return self.prepare_kernel(scalar)

    @abstractmethod
    def functional_set(self, kernel: Any, index: object, value: object) -> Any:
        """Return storage with an indexed batch region replaced."""

    def functional_add(self, kernel: Any, index: object, value: object) -> Any:
        """Return storage with values added at an indexed batch region; repeated indices accumulate."""


def context_from_key(algebra: Algebra, key: tuple[object, ...]) -> Context:
    """Reconstruct an immutable context from tracing/static metadata."""

    if not isinstance(key, tuple) or not key:
        raise ValueError(f"invalid Context key {key!r}")
    backend = key[0]
    if backend == "numpy":
        from .numpy import NumpyContext

        return NumpyContext.from_key(algebra, key)
    if backend == "exact":
        from .exact import ExactContext

        context = ExactContext(algebra)
        if context.key != key:
            raise ValueError(f"Exact Context key is not canonical: {key!r}")
        return context
    if backend == "jax":
        from .jax import JaxContext

        return JaxContext.from_key(algebra, key)
    if backend == "torch":
        from .torch import TorchContext

        return TorchContext.from_key(algebra, key)
    raise ValueError(f"unknown Context backend key {backend!r}")


def binding_context(
    target: Context,
    operands: tuple[Context, ...],
) -> "Context":
    """Resolve context compatibility once per static context signature."""

    concrete = tuple(context for context in (target,) + operands if not context.is_exact)
    if not concrete:
        return target

    context = concrete[0]
    if any(not other.is_compatible_with(context) for other in concrete[1:]):
        raise ValueError("all concrete operands must use compatible Contexts")
    return context
