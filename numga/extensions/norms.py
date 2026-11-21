"""Norms and normalization measure nullary extensors through two distinct quadratic products, while
recomputing blade coefficients even when an input carries a static unit-product trait.

`norm_squared`, `norm`, and `normalized` follow the symmetric reverse product `value * ~value`.
`antinorm` follows `value.dot(value)`, measuring an intrinsic value through its metric dual: on a
projective point, `norm` reads its homogeneous weight (so `normalized` sets weight one) whereas
`antinorm` reads its bulk (weight times Euclidean distance from the origin), and on a spatial
direction `(head - tail).antinorm()` measures Euclidean length across both vector and projective
algebras.

Normalization multiplies on the left by `(value * ~value).inverse_square_root()`, selecting the
fastest static path available: direct scalar reciprocal roots when `value * ~value` is structurally
scalar, closed-form Study square roots when `value * ~value` is a Study number (which also repairs
off-manifold grade-four drift on declared rotors), trait-certified scalar normalization in six or
more dimensions, and general principal square roots otherwise. Norm and antinorm conventions are
explained in `docs/conventions.md` and `docs/orientation.md`.
"""

from __future__ import annotations

from typing import NoReturn

from numga.extensor import Extensor
from numga.gatype import GATypePattern, ReverseProductScalar, ReverseProductZero
from numga.units import normalized_unit

from ._scalar_functions import register_scalar_function

Extensor.normalized.set_unit(normalized_unit)


@Extensor.study_norm_squared.register(lambda g: g.is_study)
def study_norm_squared(value: Extensor) -> Extensor:
    """For a Study number `a + n` whose non-scalar part `n` squares to a scalar, the squared Study
    norm equals the scalar `a * a - n * n`."""
    return value.algebra.operator.study_norm_squared(value.gatype)(value, value)


@Extensor.study_norm.register(lambda g: g.is_study)
def study_norm(value: Extensor) -> Extensor:
    """The Study norm is the scalar square root of `study_norm_squared()`."""
    return value.study_norm_squared().square_root()


@Extensor.norm_squared.register(GATypePattern(arity=0))
def norm_squared(value: Extensor) -> Extensor:
    """The squared norm evaluates the symmetric reverse product `value * ~value` directly from
    coefficients, ignoring any static unit-product trait."""
    return value.symmetric_reverse_product()


register_scalar_function(Extensor.norm, lambda context, kernel: context.xp.abs(kernel))


@Extensor.norm.register(GATypePattern(arity=0))
def norm(value: Extensor) -> Extensor:
    """The norm is the principal square root of `value.norm_squared()`, which may be a Study number
    or even multivector when `value * ~value` is non-scalar."""
    return value.norm_squared().square_root()


@Extensor.antinorm.register(GATypePattern(arity=0))
def antinorm(value: Extensor) -> Extensor:
    """The antinorm is the scalar square root of `value.dot(value)`, measuring an intrinsic element
    through its dual and returning `NaN` where `value.dot(value)` is negative."""
    return value.dot(value).square_root()


@Extensor.normalized.register(lambda g: g.entails(ReverseProductZero))
def null_normalized(value: Extensor) -> NoReturn:
    raise ValueError("cannot normalize a structurally zero reverse product")


@Extensor.normalized.register(
    lambda g: g.derive.structural.derive.symmetric_reverse.is_scalar
)
def scalar_normalized(value: Extensor) -> Extensor:
    """When `value * ~value` is structurally scalar, left-multiplying by its reciprocal square root
    scales `value` to unit reverse product."""
    correction = value.symmetric_reverse_product().inverse_square_root()
    return (correction * value).with_traits(*value.gatype.normalized_traits)


@Extensor.normalized.register(
    lambda g: g.derive.structural.derive.symmetric_reverse.is_study
)
def study_normalized(value: Extensor) -> Extensor:
    """When `value * ~value` is structurally a Study number, left-multiplying by the inverse Study
    square root removes both scalar scale error and grade-four Study drift."""
    product = value.algebra.operator.symmetric_reverse_product(value.gatype.derive.structural)
    correction = product(value, value).square_root().inverse()
    # Because product evaluates value * ~value, its inverse root multiplies on the left.
    return (correction * value).with_traits(*value.gatype.normalized_traits)


# Rotors in 6D+ whose reverse product is scalar only by trait certification fall back here after
# the structural Study case has handled 4D and 5D drift repair.
Extensor.normalized.register(lambda g: g.entails(ReverseProductScalar))(scalar_normalized)


@Extensor.normalized.register(GATypePattern(arity=0))
def general_normalized(value: Extensor) -> Extensor:
    """General normalization left-multiplies `value` by `(value * ~value).inverse_square_root()`
    whenever neither the scalar nor the Study closed form applies."""
    correction = value.symmetric_reverse_product().inverse_square_root()
    return (correction * value).with_traits(*value.gatype.normalized_traits)
