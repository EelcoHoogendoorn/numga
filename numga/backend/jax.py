"""JAX policy for the shared concrete Extensor runtime."""

from __future__ import annotations

from fractions import Fraction
from numbers import Rational
from typing import TYPE_CHECKING, Any, Literal

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

    def __repr__(self) -> str:
        return f"JaxContext(algebra={self.algebra!r}, dtype={self.dtype}, execution={self.execution!r})"


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
