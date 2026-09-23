"""Study and reverse-product norms, using the original root formulas.

Explicit measurements and normalization always recompute their coefficients,
including when the input carries a unit-product assertion.
"""

from __future__ import annotations

from typing import NoReturn

from numga.extensor import Extensor
from numga.gatype import GATypePattern, ReverseProductScalar, ReverseProductZero


@Extensor.study_norm_squared.register(lambda g: g.is_study)
def study_norm_squared(value: Extensor) -> Extensor:
    """Measure ``a*a - n*n`` for the generalized Study number ``a + n``."""
    return value.algebra.operator.study_norm_squared(value.gatype)(value, value)


@Extensor.study_norm.register(lambda g: g.is_study)
def study_norm(value: Extensor) -> Extensor:
    return value.study_norm_squared().square_root()


@Extensor.norm_squared.register(GATypePattern(arity=0))
def norm_squared(value: Extensor) -> Extensor:
    """Measure coefficients even when the reverse product is declared unit."""
    return value.symmetric_reverse_product()


@Extensor.norm.register(lambda g: g.is_reoriented_scalar)
def reoriented_scalar_norm(value: Extensor) -> Extensor:
    return value.select_subspace(value.subspace.canonical).norm()


@Extensor.norm.register(lambda g: g <= g.algebra.subspace.scalar())
def scalar_norm(value: Extensor) -> Extensor:
    return Extensor._from_prepared_kernel(
        value.context, value.gatype.structural, value.context.xp.abs(value._kernel),
    )


@Extensor.norm.register(GATypePattern(arity=0))
def norm(value: Extensor) -> Extensor:
    """Square root of the measured reverse product; it need not be scalar."""
    return value.norm_squared().square_root()


@Extensor.normalized.register(lambda g: g.entails(ReverseProductZero))
def null_normalized(value: Extensor) -> NoReturn:
    raise ValueError("cannot normalize a structurally zero reverse product")


@Extensor.normalized.register(
    lambda g: g.structural.symmetric_reverse.is_scalar
)
@Extensor.normalized.register(ReverseProductScalar)
def scalar_normalized(value: Extensor) -> Extensor:
    """Use a scalar correction, including certified scalar products in 6D+."""
    correction = value.symmetric_reverse_product().inverse_square_root()
    return (correction * value).with_traits(*value.gatype.normalized_traits)


@Extensor.normalized.register(
    lambda g: g.structural.symmetric_reverse.is_study
)
def study_normalized(value: Extensor) -> Extensor:
    """Repair the complete Study product, including declared rotors' drift."""
    product = value.algebra.operator.symmetric_reverse_product(value.subspace)
    correction = product(value, value).square_root().inverse()
    # This correction comes from x * reverse(x), so its side matters.
    return (correction * value).with_traits(*value.gatype.normalized_traits)
