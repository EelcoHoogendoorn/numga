"""Backend numerical functions on nullary scalars; predicates return batch masks."""

from typing import Any

from numga.extensor import Extensor


@Extensor.to_array.register(lambda g: g.is_scalar)
def to_array(value: Extensor) -> Any:
    """Read scalar values as a backend array with exactly the batch shape."""
    return value.cast(value.algebra.subspace.scalar()).kernel[..., 0]


@Extensor.argsort.register(lambda g: g.is_scalar)
def argsort(value: Extensor, *args: Any, **kwargs: Any) -> Any:
    """Return backend integer indices sorting scalar values over batch axes."""
    return value.context.xp.argsort(value.to_array(), *args, **kwargs)


@Extensor.argmax.register(lambda g: g.is_scalar)
def argmax(value: Extensor, *args: Any, **kwargs: Any) -> Any:
    """Return backend indices of maximal scalar values over batch axes."""
    return value.context.xp.argmax(value.to_array(), *args, **kwargs)


def _function(name):
    def apply(value: Extensor) -> Extensor:
        value = value.cast(value.algebra.subspace.scalar())
        return Extensor._from_prepared_kernel(
            value.context, value.gatype.structural,
            getattr(value.context.xp, name)(value._kernel),
        )
    apply.__name__ = name
    return apply


def _predicate(name):
    def apply(value: Extensor) -> Any:
        return getattr(value.context.xp, name)(value.to_array())
    apply.__name__ = name
    return apply


def _comparison(name):
    def apply(left: Extensor, right: Extensor) -> Any:
        return getattr(left.context.xp, name)(left.to_array(), right.to_array())
    apply.__name__ = name
    return apply


for _name in ("sin", "cos", "tan", "arcsin", "arccos", "arctan",
              "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh"):
    getattr(Extensor, _name).register(lambda g: g.is_scalar)(_function(_name))

for _name in ("isnan", "isfinite", "isinf"):
    getattr(Extensor, _name).register(lambda g: g.is_scalar)(_predicate(_name))

for _name in ("less", "less_equal", "greater", "greater_equal"):
    getattr(Extensor, _name).register(lambda a, b: a.is_scalar and b.is_scalar)(_comparison(_name))


@Extensor.clip.register(lambda g: g.is_scalar)
def clip(value: Extensor, minimum: Any, maximum: Any) -> Extensor:
    """Clip scalar values to numerical bounds, broadcasting over batch axes."""
    kernel = value.context.xp.clip(value.to_array(), minimum, maximum)
    return Extensor._from_prepared_kernel(
        value.context, value.algebra.gatype.scalar(), kernel[..., None],
    )
