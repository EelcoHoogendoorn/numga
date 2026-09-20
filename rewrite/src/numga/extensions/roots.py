"""Scalar, generalized Study, and unit or scaled rotor square roots.

These are the original real-algebra branch formulas. They select one root;
they do not search other blades for roots of negative scalars.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import GATypePattern, ReverseProductOne, Versor


@Extensor.square_root_denman_beavers.register(lambda t: t <= t.algebra.gatype.rotor())
def square_root_denman_beavers(value: Extensor, *, n: int = 30) -> Extensor:
    product = root = value
    for _ in range(n):
        reciprocal = product.inverse()
        root = root * (1 + reciprocal) / 2
        product = (product + reciprocal + 2) / 4
    return root.with_traits(ReverseProductOne, Versor)


@Extensor.geometric_mean.register(
    lambda a, b: a <= a.algebra.gatype.rotor() and b <= b.algebra.gatype.rotor()
)
def geometric_mean(left: Extensor, right: Extensor) -> Extensor:
    return (left + right).normalized()


@Extensor.square_root.register(lambda g: g.is_reoriented_scalar)
def reoriented_scalar_square_root(value: Extensor) -> Extensor:
    return value.select_subspace(value.subspace.canonical).square_root()


@Extensor.square_root.register(lambda g: g <= g.algebra.subspace.scalar())
def scalar_square_root(value: Extensor) -> Extensor:
    return Extensor._from_prepared_kernel(
        value.context, value.gatype.structural, value.context.xp.sqrt(value._kernel),
    )


@Extensor.square_root.register(lambda g: g <= g.algebra.gatype.rotor())
def rotor_square_root(value: Extensor) -> Extensor:
    """The unit even-versor root is the normalized bisector ``value + 1``."""

    return (value + 1).normalized().with_traits(ReverseProductOne, Versor)


@Extensor.square_root.register(
    lambda g: g <= g.algebra.subspace.even() and g.entails(Versor)
)
def scaled_rotor_square_root(value: Extensor) -> Extensor:
    """Root of a scaled rotor; requires a positive scalar reverse product."""

    scale = value.norm()
    unit = (value / scale).with_traits(ReverseProductOne, Versor)
    return (scale.square_root() * unit.square_root()).with_traits(Versor)


@Extensor.square_root.register(
    lambda g: g.is_study and g.nonscalar.squared.is_empty
)
def nilpotent_study_square_root(value: Extensor) -> Extensor:
    scalar = value.select[0]
    nonscalar = value.select_subspace(value.gatype.nonscalar.output_subspace)
    root = scalar.square_root()
    reciprocal = (2 * root).inverse()
    reciprocal = Extensor._from_prepared_kernel(
        value.context, reciprocal.gatype.structural,
        value.context.xp.nan_to_num(reciprocal._kernel, nan=0),
    )
    return root + nonscalar * reciprocal


@Extensor.square_root.register(lambda g: g.is_study)
def study_square_root(value: Extensor) -> Extensor:
    scalar = value.select[0]
    nonscalar = value.select_subspace(value.gatype.nonscalar.output_subspace)
    root = ((scalar + value.study_norm()) / 2).square_root()
    reciprocal = (2 * root).inverse()
    reciprocal = Extensor._from_prepared_kernel(
        value.context, reciprocal.gatype.structural,
        value.context.xp.nan_to_num(reciprocal._kernel, nan=0),
    )
    return root + nonscalar * reciprocal


@Extensor.inverse_square_root.register(lambda g: g.is_reoriented_scalar)
def reoriented_scalar_inverse_square_root(value: Extensor) -> Extensor:
    return value.select_subspace(value.subspace.canonical).inverse_square_root()


@Extensor.inverse_square_root.register(lambda g: g <= g.algebra.subspace.scalar())
def scalar_inverse_square_root(value: Extensor) -> Extensor:
    return Extensor._from_prepared_kernel(
        value.context, value.gatype.structural, value._kernel ** (-0.5),
    )


@Extensor.inverse_square_root.register(GATypePattern(arity=0))
def inverse_square_root(value: Extensor) -> Extensor:
    return value.square_root().inverse()
