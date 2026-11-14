"""Every `Extensor` pairs a static `GAType` with an immutable `Context` that governs coefficient
storage, contraction execution, and backend-specific linear algebra. Each `Algebra` owns a single
`ExactContext` for value-free integer operator kernels such as product tables and basis maps,
whereas array contexts (`NumpyContext`, `JaxContext`, `TorchContext`) hold batched floating or
complex coefficients and contract operands either through dense tensor contractions
(`execution="dense"`) or through straight-line unrolled exact terms (`execution="sparse"`).

Context compatibility separates algebra ownership from storage policy. Two contexts are compatible
when they share the same `Algebra` object by identity and carry equal `key` tuples (`backend`,
`dtype`, optional device, and `execution`); keeping `algebra` out of `key` lets tracing runtimes
such as JAX store `key` as static pytree metadata and reconstruct the context via `context_from_key`.
When operands from distinct contexts meet in `binding_context` or `lower`, exact kernels materialize
into the target array context, and NumPy extensors act as host constants that promote into a paired
JAX or PyTorch context of the same algebra.

Cross-algebra inclusion (`include_from_algebra`) embeds extensors from another algebra whose active
generators match a subset of the target algebra's generators in name and metric square. Extrinsic
axes keep their blade names while intrinsic axes keep their unit-volume complements, preserving
joins of intrinsic elements, actions of extrinsic elements from the left, and linear map evaluation
across included inputs.

Kernel-level hooks (`prepare_kernel`, `prepare_scalar`, `functional_set`, `functional_add`, and
matrix or pencil solvers) isolate array-namespace differences behind a uniform functional interface
so `Extensor` methods and `numga.extensions` remain backend-agnostic. Stages, contexts, exact
materialization, and cross-algebra inclusion are described in `docs/internals.md` (section 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Number
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from numga.gatype import GAType, UnorientedSubspace
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
    """Subclasses bind one `Algebra` to a coefficient storage representation and contraction executor."""

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
            self.algebra._multivector_factory(self),
        )

    @property
    @abstractmethod
    def algebra(self) -> Algebra: ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Coefficient precision is reported as a canonical `np.dtype` across every backend."""

    @property
    @abstractmethod
    def key(self) -> tuple[object, ...]:
        """Hashable policy metadata identifies the backend, `dtype`, device, and execution mode while excluding `algebra`."""

    @property
    def is_exact(self) -> bool:
        """Exact contexts hold value-free symbolic kernels without batch dimensions; array contexts return `False`."""
        return False

    @property
    def execution(self) -> str:
        return self._execution

    @property
    def multivector(self) -> MultivectorFactory:
        """Attribute and call access constructs nullary extensors directly in this context."""
        return self._multivector_factory

    def include_from_algebra(self, extensor: Extensor) -> Extensor:
        """Generator-name inclusion embeds `extensor` into `self.algebra` whenever every generator active in
        `extensor.gatype` appears in `self.algebra` with the same metric square, as Euclidean `x+y+z+` lies
        inside projective `w0x+y+z+`. Unused ambient generators in the source algebra (such as time after
        slicing a spacetime event via `mv.t ^ (Point & time)`) do not participate, while every active
        component is preserved with signed coefficient reordering.

        Extrinsic axes keep their blade names (`context.include_from_algebra(space.x) == mv.x`, the plane
        through the origin) and preserve the geometric product, so a spatial rotor becomes the projective
        rotor about the origin. Intrinsic axes keep their unit-volume complements
        (`context.include_from_algebra(space.x.dual()) == mv.x.dual()`), mapping a spatial offset to the
        projective point at infinity (`origin + context.include_from_algebra(offset)` gives the finite point)
        and the spatial unit volume to the projective unit volume.

        Joins of intrinsic elements carry over unchanged, as do products and pairings with an extrinsic
        element on the left (`plane & point`, `rotor >> point`), whereas products with an intrinsic element
        on the left may change sign or vanish where a degenerate generator squares to zero. Linear maps and
        forms transport each axis by its own orientation so
        `context.include_from_algebra(f)(context.include_from_algebra(x)) == context.include_from_algebra(f(x))`."""
        from numga.multivector.subalgebra import include_from_algebra

        return include_from_algebra(self, extensor)

    @property
    def gatype(self) -> GATypeFactory:
        return self.algebra.gatype

    @property
    def subspace(self) -> SubSpaceFactory:
        return self.algebra.subspace

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Dense and sparse executors dispatch array operations through this NumPy-compatible namespace."""

    @abstractmethod
    def prepare_kernel(self, coefficients: object) -> Any:
        """Public coefficient data is validated for numeric kind compatibility and copied into backend storage."""

    def materialize(self, kernel: SymbolicKernel) -> Any:
        """Materializing an exact `SymbolicKernel` converts its entries into backend storage of
        `self.dtype`."""
        return self.xp.asarray(kernel.materialize(self.dtype))

    def with_dtype(self, dtype: object) -> Context:
        """Replacing `dtype` yields a sibling context on the same algebra, device, and execution mode."""
        return type(self)(self.algebra, dtype, execution=self.execution)

    def kernel_dtype(self, kernel: Any) -> np.dtype:
        """Backend storage reports its element type as the corresponding `np.dtype`."""
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

    def axes_trace(self, kernel: Any, axis1: int, axis2: int) -> Any:
        return self.xp.trace(kernel, axis1=axis1, axis2=axis2)

    def solve(self, matrix: Any, rhs: Any) -> Any:
        """Broadcasting leading batch axes of `rhs` to match `matrix` solves the linear system along
        the trailing vector axis."""
        rhs = self.xp.broadcast_to(rhs, matrix.shape[:-1])
        return self.xp.linalg.solve(matrix, rhs[..., None])[..., 0]

    def first_last_input(self, kernel: Any, batch_ndim: int) -> Any:
        """Dropping the length-1 scalar output axis and moving the final input axis first turns a scalar form
        kernel into a matrix-like layout."""
        return self.xp.moveaxis(kernel[(slice(None),) * batch_ndim + (0,)], -1, batch_ndim)

    def generalized_eigh(self, matrix: Any, metric: Any) -> tuple[Any, Any]:
        """Cholesky factorization of the positive-definite `metric` reduces a Hermitian pencil
        `matrix * v == lam * metric * v` to a standard problem with `metric`-orthonormal eigenvectors."""
        xp = self.xp
        inverse = xp.linalg.inv(xp.linalg.cholesky(metric))
        adjoint = xp.conj(xp.swapaxes(inverse, -1, -2))
        values, vectors = xp.linalg.eigh(inverse @ matrix @ adjoint)
        return values, adjoint @ vectors

    def generalized_eigvalsh(self, matrix: Any, metric: Any) -> Any:
        return self.generalized_eigh(matrix, metric)[0]

    def generalized_eig(self, matrix: Any, metric: Any) -> tuple[Any, Any]:
        """Solving `metric` against `matrix` reduces a general pencil with invertible `metric` to a standard
        eigenproblem."""
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
        gatype: GAType,
        coefficients: object,
    ) -> Extensor:
        from numga.extensor import Extensor

        if isinstance(gatype, SubSpace):
            raise UnorientedSubspace()
        if not isinstance(gatype, GAType):
            raise TypeError("extensor construction requires a GAType")
        return Extensor(self, gatype, coefficients)

    def lower(self, extensor: Extensor) -> Extensor:
        """Lowering an extensor into `self` materializes exact kernels directly and transfers
        concrete arrays from other backends through host memory."""
        if extensor.context is self:
            return extensor
        from numga.extensor import Extensor

        if extensor.algebra is not self.algebra:
            raise ValueError("Extensor and Context belong to different algebras")
        if extensor.context.is_compatible_with(self):
            return Extensor._from_prepared_kernel(self, extensor.gatype, extensor._kernel)
        if extensor.context.is_exact and not self.is_exact:
            numeric = self.materialize(extensor._kernel)
            return Extensor._from_prepared_kernel(self, extensor.gatype, numeric)
        if not extensor.context.is_exact and not self.is_exact:
            # Concrete array backends transfer coefficients through a host NumPy array.
            return Extensor(self, extensor.gatype, np.asarray(extensor.kernel))
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
        """Zero-dimensional scalars convert to backend storage while rejecting rank-nonzero arrays and
        incompatible numeric kinds."""
        if not isinstance(scalar, Number) and getattr(scalar, "ndim", None) != 0:
            raise TypeError(f"expected a scalar, got {type(scalar).__name__}")
        return self.prepare_kernel(scalar)

    @abstractmethod
    def functional_set(self, kernel: Any, index: object, value: object) -> Any:
        """Out-of-place batch indexing returns a new storage buffer with `index` replaced by `value`."""

    def functional_add(self, kernel: Any, index: object, value: object) -> Any:
        """Out-of-place scatter-addition returns a new storage buffer where repeated entries in `index` accumulate."""


def context_from_key(algebra: Algebra, key: tuple[object, ...]) -> Context:
    """Static `key` metadata paired with `algebra` reconstructs the matching immutable `Context` across
    tracing boundaries."""
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
) -> Context:
    """All-exact bindings stay in `target`, whereas concrete bindings select the shared array context and
    promote any paired NumPy or exact operands into it."""
    concrete = tuple(context for context in (target,) + operands if not context.is_exact)
    if not concrete:
        return target

    # NumPy operands act as host constants when paired with a JAX or PyTorch context of the same algebra.
    hosted = [context for context in concrete if context.key[0] != "numpy"]
    context = hosted[0] if hosted else concrete[0]
    if any(not other.is_compatible_with(context)
           and not (other.key[0] == "numpy" and other.algebra is context.algebra) for other in concrete):
        raise ValueError("all concrete operands must use compatible Contexts")
    return context
