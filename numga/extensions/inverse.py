"""Recursive inverse formulas, selected by GAType facts.

Self-products reduce to scalars in at most three steps below six dimensions.
Traits can establish shorter paths on wider carriers. All inverses require
invertible input; the general matrix fallback is numerical only.
"""

from __future__ import annotations


from numga.extensor import Extensor
from numga.gatype import (
    CliffordConjugateProductOne,
    CoefficientOrthogonal,
    GATypePattern,
    ReverseProductOne,
    ReverseProductZero,
)


@Extensor.inverse.register(lambda t: t.arity == 0 and t.entails(ReverseProductZero))
def inverse_null(value: Extensor) -> Extensor:
    raise ZeroDivisionError("a statically null multivector has no inverse")


@Extensor.inverse.register(lambda t: t.entails(ReverseProductOne))
def inverse_unit_reverse(value: Extensor) -> Extensor:
    return value.reverse()


@Extensor.inverse.register(lambda t: t.entails(CliffordConjugateProductOne))
def inverse_unit_conjugate(value: Extensor) -> Extensor:
    return value.clifford_conjugate()


@Extensor.inverse.register(lambda t: t.entails(CoefficientOrthogonal))
def inverse_orthogonal(value: Extensor) -> Extensor:
    """A coefficient-orthogonal map is inverted by transposing its coefficients."""
    permutation = tuple(range(value.ndim)) + (value.ndim + 1, value.ndim)
    return Extensor._from_prepared_kernel(
        value.context, value.gatype.transposed, value.context.xp.transpose(value._kernel, permutation),
    )


@Extensor.inverse.register(lambda t: t.is_scalar)
def inverse_scalar(value: Extensor) -> Extensor:
    """Take the reciprocal directly, without squaring the coefficients."""

    gatype = value.gatype.structural.with_traits(*value.gatype.inverse_traits)
    return Extensor._from_prepared_kernel(
        value.context, gatype, value.context.reciprocal(value._kernel),
    )


def register_reductions(steps: int) -> None:
    """Prefer shorter reductions, then the order in which the transforms are listed."""

    @Extensor.inverse.register(
        lambda t: t.squared.reduces_to_scalar(steps - 1)
    )
    def inverse_squared(value: Extensor) -> Extensor:
        return (value / value.squared()).with_traits(*value.gatype.inverse_traits)

    @Extensor.inverse.register(
        lambda t: t.symmetric_reverse.reduces_to_scalar(steps - 1)
    )
    def inverse_reverse(value: Extensor) -> Extensor:
        return (value.reverse() / value.symmetric_reverse_product()).with_traits(
            *value.gatype.inverse_traits,
        )

    @Extensor.inverse.register(
        lambda t: t.symmetric_conjugate.reduces_to_scalar(steps - 1)
    )
    def inverse_conjugate(value: Extensor) -> Extensor:
        result = value.clifford_conjugate() / value.symmetric_conjugate_product()
        return result.with_traits(*value.gatype.inverse_traits)

    @Extensor.inverse.register(
        lambda t: t.symmetric_scalar_negation.reduces_to_scalar(steps - 1)
    )
    def inverse_scalar_negation(value: Extensor) -> Extensor:
        result = value.scalar_negation() / value.symmetric_scalar_negation_product()
        return result.with_traits(*value.gatype.inverse_traits)

    @Extensor.inverse.register(
        lambda t: t.symmetric_pseudoscalar_negation.reduces_to_scalar(steps - 1)
    )
    def inverse_pseudoscalar_negation(value: Extensor) -> Extensor:
        result = value.pseudoscalar_negation() / value.symmetric_pseudoscalar_negation_product()
        return result.with_traits(*value.gatype.inverse_traits)

    @Extensor.inverse.register(
        lambda t: t.symmetric_involute.reduces_to_scalar(steps - 1)
    )
    def inverse_involute(value: Extensor) -> Extensor:
        return (value.involute() / value.symmetric_involute_product()).with_traits(
            *value.gatype.inverse_traits,
        )


register_reductions(1)


@Extensor.inverse.register(
    lambda t: t.algebra.dimension == 5
    and t <= t.algebra.subspace.bivector()
)
def inverse_bivector_5d(value: Extensor) -> Extensor:
    """The apparent grade-four terms in the general inverse cancel exactly."""

    return value.bivector_product(-value.symmetric_reverse_product().inverse())


@Extensor.inverse.register(
    lambda t: t.algebra.dimension == 5
    and t <= t.algebra.subspace.trivector()
)
def inverse_trivector_5d(value: Extensor) -> Extensor:
    return value.trivector_product(value.squared().inverse())


register_reductions(2)
register_reductions(3)


@Extensor.inverse.register(GATypePattern(arity=0))
def inverse_geometric(value: Extensor) -> Extensor:
    """Solve left multiplication in the blade-generated subalgebra."""

    space = value.gatype.minimal_subalgebra.output_subspace
    left_multiply = (value * space).select_subspace(space)
    unit = value.algebra.operator.unit(space)
    coefficients = value.context.solve(left_multiply._kernel, value.context.lower(unit)._kernel)
    gatype = value.gatype.minimal_subalgebra.with_traits(*value.gatype.inverse_traits)
    return Extensor._from_prepared_kernel(
        value.context, gatype, coefficients,
    )


@Extensor.inverse_shirokov.register(GATypePattern(arity=0))
def inverse_shirokov(value: Extensor) -> Extensor:
    """Characteristic-polynomial inverse; ill-conditioned at high dimensions."""
    order = 2 ** ((value.algebra.dimension + 1) // 2)
    power = value
    adjugate = value.context.multivector.scalar()
    for k in range(1, order):
        adjugate = power - power.select[0] * (order / k)
        power = value * adjugate
    return adjugate / power.select[0]


@Extensor.inverse_factor.register(GATypePattern(arity=0))
def inverse_factor(value: Extensor) -> Extensor:
    """Hitzer's reverse/conjugate factor; inspect its scalar reduction statically."""
    return ~value * value.symmetric_reverse_product().clifford_conjugate()


@Extensor.inverse_hitzer.register(
    lambda t: t.arity == 0 and t.symmetric_reverse.symmetric_conjugate.is_scalar
)
def inverse_hitzer(value: Extensor) -> Extensor:
    factor = value.inverse_factor()
    return factor / value.scalar_product(factor)
