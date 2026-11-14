"""Scalar extensions connect nullary `Scalar` extensors (`arity == 0`, `is_scalar`) to the active
array backend (`xp`). Elementary transcendental functions (`sin`, `cos`, `sinh`, `arcsin`, and their
companions), `abs`, `clip`, and batch extrema (`min`, `max`) stay inside the algebra, aligning signed
scalar layouts first via `.cast(algebra.subspace.scalar())` and returning `Scalar` extensors with the
trailing blade axis intact.

Operations whose results leave the geometric algebra unpack directly to raw backend arrays matching
the extensor's batch `shape`. `to_array` strips the length-one scalar blade axis, while predicates
(`isnan`, `isfinite`, `isinf`), order comparisons (`<`, `<=`, `>`, `>=`), and index queries
(`argsort`, `argmax`, `argmin`) return backend boolean masks or integer index arrays directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from numga.extensor import Extensor


@Extensor.to_array.register(lambda g: g.is_scalar)
def to_array(value: Extensor) -> Any:
    """Extracting scalar coefficients drops the length-one blade axis to yield a backend array with
    the extensor's batch `shape`."""
    return value.cast(value.algebra.subspace.scalar()).kernel[..., 0]


@Extensor.argsort.register(lambda g: g.is_scalar)
def argsort(value: Extensor, *args: Any, **kwargs: Any) -> Any:
    """Sorting a scalar extensor returns backend integer indices along the requested batch axis."""
    return value.context.xp.argsort(value.to_array(), *args, **kwargs)


@Extensor.argmax.register(lambda g: g.is_scalar)
def argmax(value: Extensor, *args: Any, **kwargs: Any) -> Any:
    """Finding the maximum along batch axes returns the backend integer indices of maximal scalar
    entries."""
    return value.context.xp.argmax(value.to_array(), *args, **kwargs)


@Extensor.argmin.register(lambda g: g.is_scalar)
def argmin(value: Extensor, *args: Any, **kwargs: Any) -> Any:
    """Finding the minimum along batch axes returns the backend integer indices of minimal scalar
    entries."""
    return value.context.xp.argmin(value.to_array(), *args, **kwargs)


def _reduction(name: str) -> Callable:
    def apply(value: Extensor, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False) -> Extensor:
        value = value.cast(value.algebra.subspace.scalar())
        reduced = getattr(value.context.xp, name)(value.to_array(), axis=axis, keepdims=keepdims)
        return Extensor._from_prepared_kernel(value.context, value.gatype.derive.structural, reduced[..., None])
    apply.__name__ = name
    apply.__doc__ = f"Reducing along the selected batch axes returns the {name}imum as a scalar extensor."
    return apply


for _name in ("min", "max"):
    getattr(Extensor, _name).register(lambda g: g.is_scalar)(_reduction(_name))


def _function(name: str) -> Callable:
    def apply(value: Extensor) -> Extensor:
        value = value.cast(value.algebra.subspace.scalar())
        return Extensor._from_prepared_kernel(
            value.context, value.gatype.derive.structural,
            getattr(value.context.xp, name)(value._kernel),
        )
    apply.__name__ = name
    return apply


def _predicate(name: str) -> Callable:
    def apply(value: Extensor) -> Any:
        return getattr(value.context.xp, name)(value.to_array())
    apply.__name__ = name
    return apply


def _comparison(name: str) -> Callable:
    def apply(left: Extensor, right: Extensor) -> Any:
        return getattr(left.context.xp, name)(left.to_array(), right.to_array())
    apply.__name__ = name
    return apply


for _name in ("abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
              "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh"):
    getattr(Extensor, _name).register(lambda g: g.is_scalar)(_function(_name))

for _name in ("isnan", "isfinite", "isinf"):
    getattr(Extensor, _name).register(lambda g: g.is_scalar)(_predicate(_name))

for _name in ("less", "less_equal", "greater", "greater_equal"):
    getattr(Extensor, _name).register(lambda a, b: a.is_scalar and b.is_scalar)(_comparison(_name))


@Extensor.clip.register(lambda g: g.is_scalar)
def clip(value: Extensor, minimum: Any, maximum: Any) -> Extensor:
    """Clamping entries to `[minimum, maximum]` across batch axes returns a scalar extensor."""
    kernel = value.context.xp.clip(value.to_array(), minimum, maximum)
    return Extensor._from_prepared_kernel(
        value.context, value.algebra.gatype.scalar(), kernel[..., None],
    )
