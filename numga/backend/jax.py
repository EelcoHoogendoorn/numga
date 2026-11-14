"""Array execution on JAX pairs `jax.Array` coefficient storage with either dense contractions or
unrolled sparse-exact kernels, enabling 64-bit precision on import so CPU computations match
`NumpyContext` precision. Host coefficients passed to `prepare_kernel` evaluate under
`jax.ensure_compile_time_eval()`, allowing cached basis blades and constant operators to materialize
inside active `jit` or `vmap` traces without leaking tracers.

Registering `Extensor` as a JAX pytree node places `_kernel` as the sole dynamic child while storing
`(gatype, context.key)` as static auxiliary metadata. Keeping `Algebra` out of `context.key` and
recovering it from `gatype.algebra` on unflatten lets traced transformations (`jax.jit`, `jax.vmap`,
`jax.lax.scan`, `jax.grad`) preserve static geometric types and orientations with zero runtime
dispatch overhead.

Automatic differentiation through `derivative` wraps forward-mode (`jvp`) and reverse-mode (`vjp`)
differentiation directly in the geometric type system, appending the differentiated input's subspace
and orientation as a trailing linear input axis (`S <- (I..., T)`). A scalar function therefore
differentiates to the linear form `Scalar <- T` rather than an element of `T`, higher derivatives
nest into multilinear forms such as `Scalar <- (T, T)`, and each call selects reverse pullbacks or
forward pushforwards from the ratio of output to coupled input degrees of freedom. Tracing constant
tables under `jit` is illustrated in `docs/internals.md` (sections 2 and 4), and differentiating
functions of fields in `docs/fields.md` ("Derivatives with respect to a field").
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from numbers import Rational
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision on import so CPU runs match NumPy round-off in numerical checks.
jax.config.update("jax_enable_x64", True)

from numga.extensor import Extensor

from .context import Context, context_from_key

if TYPE_CHECKING:
    from types import ModuleType

    from numga.algebra import Algebra
    from numga.gatype import GAType


def _dtype(dtype: object) -> np.dtype:
    canonical = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
    if canonical.kind not in "fc":
        raise TypeError("JaxContext currently requires a real or complex floating dtype")
    return canonical


class JaxContext(Context):
    """Immutable policy stores coefficients as `jax.Array` buffers bound to one `Algebra` with dense
    or sparse-exact execution."""

    __slots__ = ("_algebra", "_dtype")

    def __init__(
        self, algebra: Algebra | str, dtype: object = np.float64,
        *, execution: Literal["dense", "sparse"] = "dense",
    ) -> None:
        from numga.algebra import Algebra

        object.__setattr__(self, "_algebra", Algebra(algebra) if isinstance(algebra, str) else algebra)
        object.__setattr__(self, "_dtype", _dtype(dtype))
        super().__init__(execution)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("JaxContext instances are immutable")

    @classmethod
    def from_key(cls, algebra: Algebra, key: tuple[object, ...]) -> JaxContext:
        if len(key) != 3 or key[0] != "jax":
            raise ValueError(f"invalid JAX Context key {key!r}")
        context = cls(algebra, key[1], execution=key[2])
        if context.key != key:
            raise ValueError(f"JAX Context key is not canonical: {key!r}")
        return context

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def key(self) -> tuple[object, ...]:
        return ("jax", self.dtype.str, self.execution)

    @property
    def xp(self) -> ModuleType:
        return jnp

    def prepare_kernel(self, coefficients: object) -> jax.Array:
        source_dtype = getattr(coefficients, "dtype", None)
        if source_dtype is None:
            source_dtype = np.asarray(coefficients).dtype
        source_dtype = np.dtype(source_dtype)
        if not np.can_cast(source_dtype, self.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot represent coefficients of dtype {source_dtype} in "
                f"Context dtype {self.dtype} without changing numeric kind"
            )
        if isinstance(coefficients, jax.core.Tracer):
            return jnp.asarray(coefficients, dtype=self.dtype)
        # Materializing non-tracer data under ensure_compile_time_eval lets cached basis blades build
        # inside an active transformation without a tracer escaping its trace.
        with jax.ensure_compile_time_eval():
            return jnp.asarray(coefficients, dtype=self.dtype)

    def prepare_scalar(self, scalar: object) -> jax.Array:
        if isinstance(scalar, Rational):
            scalar = float(Fraction(scalar))
        source_dtype = getattr(scalar, "dtype", None)
        if source_dtype is None:
            source_dtype = np.asarray(scalar).dtype
        source_dtype = np.dtype(source_dtype)
        if not np.can_cast(source_dtype, self.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot represent scalar dtype {source_dtype} in Context dtype "
                f"{self.dtype} without changing numeric kind"
            )
        result = jnp.asarray(scalar, dtype=self.dtype)
        if result.ndim != 0:
            raise TypeError(f"expected a scalar, got shape {result.shape}")
        return result

    def functional_set(
        self, kernel: jax.Array, index: object, value: object
    ) -> jax.Array:
        return kernel.at[index].set(value)

    def functional_add(
        self, kernel: jax.Array, index: object, value: object
    ) -> jax.Array:
        return kernel.at[index].add(value)

    def __repr__(self) -> str:
        return f"JaxContext(algebra={self.algebra!r}, dtype={self.dtype}, execution={self.execution!r})"


def derivative(function: Callable) -> Callable:
    """Differentiating `function` returns a callable that evaluates the linear tangent map `S <- (I..., T)`
    at any nullary value of type `T`. Binding a step of type `T` into the returned map yields the
    first-order change in `function(value)`; a scalar function produces the covector form `Scalar <- T`,
    and nested `derivative(derivative(function))` calls append further step axes to form multilinear
    maps such as `Scalar <- (T, T)`.

    Leading batch dimensions whose sizes match between `value` and `function(value)` act as independent
    copies, differentiating each output entry with respect to its corresponding input entry alone so
    cross terms between batch entries sum onto the diagonal. Any trailing batch dimensions of `value`
    beyond that shared prefix remain coupled and appear after the result's batch dimensions, giving a
    reduction over a trailing batch axis a derivative with respect to every entry along that axis.

    Field sites belong to `T`'s structural shape rather than its batch shape, so every site stays
    coupled to every other site and the appended step axis ranges over the same sites (producing
    `Scalar <- (T[n], T[n])` for the Hessian of a scalar energy on `T[n]`). Supplying any JAX pytree of
    nullary extensors returns a matching pytree of partial derivatives with all sibling leaves held fixed.
    """
    def at(value: Any) -> Any:
        if isinstance(value, Extensor):
            return along(function, value)
        leaves, structure = jax.tree_util.tree_flatten(value, is_leaf=lambda leaf: isinstance(leaf, Extensor))

        def holding(index: int) -> Callable:
            def varied(leaf: Extensor) -> Extensor:
                return function(jax.tree_util.tree_unflatten(structure, leaves[:index] + [leaf] + leaves[index + 1:]))
            return varied

        return jax.tree_util.tree_unflatten(structure, [along(holding(index), leaf) for index, leaf in enumerate(leaves)])

    def along(function: Callable, value: Extensor) -> Extensor:
        if value.arity:
            raise TypeError(f"derivatives are taken with respect to values, not to maps of type {value.gatype}")

        def result(kernel: jax.Array) -> jax.Array:
            return function(Extensor._from_prepared_kernel(value.context, value.gatype, kernel)).kernel

        image = function(value)
        value_batch, result_batch = value.shape, image.shape
        # Structural axes of the input comprise any field site axes followed by its blade axis.
        axis = value.gatype.structural_shape
        sites = len(axis) - 1
        image_sites, image_blades = len(image.gatype.site_shape), len(image.gatype.subspaces)
        # Leading batch dimensions of equal size act as independent elementwise copies.
        shared_value = set()
        for dimension, (value_size, result_size) in enumerate(zip(value_batch, result_batch)):
            if value_size != result_size:
                break
            shared_value.add(dimension)
        coupled = [dimension for dimension in range(len(value_batch)) if dimension not in shared_value]
        coupled_shape = tuple(value_batch[dimension] for dimension in coupled)

        # Past the shared leading batch prefix, compare coupled input and output coefficient counts per case.
        shared_shape = value_batch[:len(shared_value)]
        image_shape = image.kernel.shape[len(shared_value):]
        count = int(np.prod(coupled_shape + axis, dtype=int))
        image_count = int(np.prod(image_shape, dtype=int))
        if image_count < count:
            # Narrower outputs use reverse mode, pulling back one unit cotangent per output coefficient
            # broadcasted across the independent shared batch dimensions.
            basis = jnp.eye(image_count, dtype=value.kernel.dtype).reshape((image_count,) + image_shape)
            for dimension in range(len(shared_value)):
                basis = jnp.expand_dims(basis, 1 + dimension)
            cotangents = jnp.broadcast_to(basis, (image_count,) + image.kernel.shape)
            _, pullback = jax.vjp(result, value.kernel)
            pulled = jax.vmap(lambda cotangent: pullback(cotangent)[0])(cotangents)
            # Permuting moves [coupled..., value_axis] ahead of [shared..., result_coeffs] to match forward mode.
            pulled = pulled.reshape(image_shape + value_batch + axis)
            r, s = len(image_shape), len(shared_shape)
            changes = jnp.transpose(pulled, tuple(range(r + s, pulled.ndim)) + tuple(range(r, r + s)) + tuple(range(r)))
        else:
            # Wider outputs use forward mode, pushing forward one unit tangent per coupled input coefficient
            # broadcasted across the independent shared batch dimensions.
            basis = jnp.eye(count, dtype=value.kernel.dtype).reshape((count,) + coupled_shape + axis)
            for dimension in sorted(shared_value):
                basis = jnp.expand_dims(basis, 1 + dimension)
            tangents = jnp.broadcast_to(basis, (count,) + value_batch + axis)
            changes = jax.vmap(lambda tangent: jax.jvp(result, (value.kernel,), (tangent,))[1])(tangents)
        # Final transpose orders axes as [result_batch..., coupled..., result_sites..., value_sites, result_blades..., value_blades].
        changes = changes.reshape(coupled_shape + axis + image.kernel.shape)
        c, r = len(coupled_shape), len(result_batch)
        start = c + sites + 1 + r
        order = (
            tuple(range(c + sites + 1, start))
            + tuple(range(c))
            + tuple(range(start, start + image_sites))
            + tuple(range(c, c + sites))
            + tuple(range(start + image_sites, start + image_sites + image_blades))
            + (c + sites,)
        )
        step = tuple((len(image.gatype.subspaces), number) for _, number in value.gatype.fields)
        gatype = value.algebra.gatype.oriented(
            image.gatype.subspaces + (value.gatype.output_subspace,),
            image.gatype.orientations + (value.gatype.orientation,),
            fields=image.gatype.fields + step,
        )
        return Extensor._from_prepared_kernel(value.context, gatype, jnp.transpose(changes, order))

    return at


def _flatten_extensor(extensor: Extensor) -> tuple:
    if not isinstance(extensor.context, JaxContext):
        raise TypeError(
            "only Extensors in a JaxContext can cross a JAX transformation boundary"
        )
    return (extensor._kernel,), (extensor.gatype, extensor.context.key)


def _unflatten_extensor(
    metadata: tuple,
    children: tuple[Any, ...],
) -> Extensor:
    gatype, context_key = metadata
    context = context_from_key(gatype.algebra, context_key)
    return Extensor._from_prepared_kernel(context, gatype, children[0])


jax.tree_util.register_pytree_node(Extensor, _flatten_extensor, _unflatten_extensor)
