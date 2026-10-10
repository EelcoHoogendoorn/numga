"""JAX policy for the shared concrete Extensor runtime."""

from __future__ import annotations

from fractions import Fraction
from numbers import Rational
from typing import TYPE_CHECKING, Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from numga.extensor import Extensor

from .context import Context, context_from_key

if TYPE_CHECKING:
    from types import ModuleType

    from numga.algebra import Algebra
    from numga.gatype import GAType


def _dtype(value: object) -> np.dtype:
    dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(value)))
    if dtype.kind not in "fc":
        raise TypeError("JaxContext currently requires a real or complex floating dtype")
    return dtype


class JaxContext(Context):
    """JAX storage with dense or sparse-exact execution, bound to one algebra."""

    __slots__ = ("_algebra", "_dtype")

    def __init__(
        self, algebra: Algebra | str, dtype: object = np.float32,
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

    def prepare_kernel(self, value: object) -> jax.Array:
        source_dtype = getattr(value, "dtype", None)
        if source_dtype is None:
            source_dtype = np.asarray(value).dtype
        source_dtype = np.dtype(source_dtype)
        if not np.can_cast(source_dtype, self.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot represent coefficients of dtype {source_dtype} in "
                f"Context dtype {self.dtype} without changing numeric kind"
            )
        if isinstance(value, jax.core.Tracer):
            return jnp.asarray(value, dtype=self.dtype)
        # Host data is a constant even inside a transformation: built concretely, it may be cached, as
        # the factory caches basis blades, without a traced value escaping the trace.
        with jax.ensure_compile_time_eval():
            return jnp.asarray(value, dtype=self.dtype)

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


def derivative(function: Callable[[Any], Extensor]) -> Callable[[Any], Any]:
    """The derivative of a function of a value: at each value, the linear map from a step to the change.

    At a value of type `T`, the derivative of a function whose result has type `S <- (I...)` is an
    extensor of type `S <- (I..., T)`: the result's own slots, then one more for the step. Binding a
    step into it gives the first-order change of the result; for a scalar result it is the linear form
    `Scalar <- T`, the gradient as the form it is, never as an element of `T`. Derivatives nest: the
    derivative of a derivative takes one more slot again, so the second derivative of a scalar
    function is the bilinear form `Scalar <- (T, T)`.

    Leading batch axes hold independent cases. The result's leading batch axes, as far as they match the
    value's in size, are copies: each element of the result is differentiated with respect to the
    matching element of the value alone, and the axis appears once. The value's remaining batch axes are
    coupled: each element of the result is differentiated with respect to every element along them, and
    they appear after the result's batch axes, so a function summing over a trailing axis of its value
    has a derivative for every element along it. Copies are taken to be independent; a function that
    couples them has its cross terms summed into the diagonal.

    A field's sites lie inside its slot, not in its batch: they are always coupled, and the step's
    slot ranges over the same sites. The second derivative of a scalar function of a field `T[n]` is
    the coupled bilinear form `Scalar <- (T[n], T[n])`, every site against every other, and solving it
    against the gradient gives the coupled Newton step.

    The value may be a record of extensors, any JAX pytree of them, such as a registered dataclass: the
    derivative is the same record, of the derivatives with respect to each of its extensors, the others
    held, as `jax.grad` takes the gradient of a pytree.
    """
    def at(value: Any) -> Any:
        if isinstance(value, Extensor):
            return along(function, value)
        leaves, structure = jax.tree_util.tree_flatten(value, is_leaf=lambda leaf: isinstance(leaf, Extensor))

        def holding(index: int) -> Callable[[Extensor], Extensor]:
            def varied(leaf: Extensor) -> Extensor:
                return function(jax.tree_util.tree_unflatten(structure, leaves[:index] + [leaf] + leaves[index + 1:]))
            return varied

        return jax.tree_util.tree_unflatten(structure, [along(holding(index), leaf) for index, leaf in enumerate(leaves)])

    def along(function: Callable[[Extensor], Extensor], value: Extensor) -> Extensor:
        if value.arity:
            raise TypeError(f"derivatives are taken with respect to values, not to maps of type {value.gatype}")

        def result(kernel: jax.Array) -> jax.Array:
            return function(Extensor._from_prepared_kernel(value.context, value.gatype, kernel)).kernel

        image = function(value)
        value_batch, result_batch = value.shape, image.shape
        # The value's slot: its sites, if it is a field, and its blades.
        slot = value.gatype.structural_shape
        sites = len(slot) - 1
        image_sites, image_blades = len(image.gatype.site_shape), len(image.gatype.subspaces)
        # The leading axes the two share, as far as they match:
        shared_value = set()
        for axis, (value_size, result_size) in enumerate(zip(value_batch, result_batch)):
            if value_size != result_size:
                break
            shared_value.add(axis)
        coupled = [axis for axis in range(len(value_batch)) if axis not in shared_value]
        coupled_shape = tuple(value_batch[axis] for axis in coupled)

        # One tangent per coupled element and coefficient of the value, spread over every element of
        # the shared axes at once, which is valid because those elements are independent:
        count = int(np.prod(coupled_shape + slot, dtype=int))
        basis = jnp.eye(count, dtype=value.kernel.dtype).reshape((count,) + coupled_shape + slot)
        for axis in sorted(shared_value):
            basis = jnp.expand_dims(basis, 1 + axis)
        tangents = jnp.broadcast_to(basis, (count,) + value_batch + slot)
        changes = jax.vmap(lambda tangent: jax.jvp(result, (value.kernel,), (tangent,))[1])(tangents)
        # [coupled..., value sites, value blades, result batch..., result sites..., result blades...] to
        # [result batch..., coupled..., result sites..., value sites, result blades..., value blades]
        changes = changes.reshape(coupled_shape + slot + image.kernel.shape)
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
        gatype = value.algebra.gatype(
            image.gatype.subspaces + (value.gatype.output_subspace,), fields=image.gatype.fields + step,
        )
        return Extensor._from_prepared_kernel(value.context, gatype, jnp.transpose(changes, order))

    return at


def _flatten_extensor(
    value: Extensor,
) -> tuple[tuple[jax.Array], tuple[GAType, tuple[object, ...]]]:
    if not isinstance(value.context, JaxContext):
        raise TypeError(
            "only Extensors in a JaxContext can cross a JAX transformation boundary"
        )
    return (value._kernel,), (value.gatype, value.context.key)


def _unflatten_extensor(
    metadata: tuple[GAType, tuple[object, ...]],
    children: tuple[Any, ...],
) -> Extensor:
    gatype, context_key = metadata
    context = context_from_key(gatype.algebra, context_key)
    return Extensor._from_prepared_kernel(context, gatype, children[0])


jax.tree_util.register_pytree_node(Extensor, _flatten_extensor, _unflatten_extensor)
