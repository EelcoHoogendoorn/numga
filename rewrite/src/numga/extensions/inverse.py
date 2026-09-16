"""The original recursive inverse formulas, selected by GAType facts.

Self-products reduce to scalars in at most three steps below six dimensions.
Traits can establish shorter paths on wider carriers. All inverses require
invertible input; the general matrix fallback is numerical only.
"""

from __future__ import annotations

from typing import NoReturn

from numga.extensor import Extensor
from numga.gatype import (
    CliffordConjugateProductOne,
    CoefficientOrthogonal,
    GATypePattern,
    ReverseProductOne,
)


@Extensor.inverse.register(lambda t: t <= t.algebra.subspace.empty())
def inverse_empty(value: Extensor) -> Extensor:
    """Use scalar-zero reciprocal semantics, preserving the batch shape."""

    return (value + 0).inverse()


@Extensor.inverse.register(lambda t: t.entails(ReverseProductOne))
def inverse_unit_reverse(value: Extensor) -> Extensor:
    return value.reverse()


@Extensor.inverse.register(lambda t: t.entails(CliffordConjugateProductOne))
def inverse_unit_conjugate(value: Extensor) -> Extensor:
    return value.clifford_conjugate()


@Extensor.inverse.register(lambda t: t.entails(CoefficientOrthogonal))
def inverse_orthogonal(value: Extensor) -> Extensor:
    return value.transpose()


@Extensor.inverse.register(lambda t: t.is_scalar)
def inverse_scalar(value: Extensor) -> Extensor:
    """Take the reciprocal directly, without squaring the coefficients."""

    gatype = value.gatype.structural.with_traits(*value.gatype.inverse_traits)
    return Extensor._from_prepared_kernel(
        value.context, gatype, value.context.reciprocal(value._kernel),
    )


def register_reductions(steps: int) -> None:
    """Prefer shorter reductions, then the original transform order."""

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


@Extensor.inverse.register(lambda t: t.is_square_map)
def inverse_linear(value: Extensor) -> Extensor:
    """Composition inverse, swapping input and output coefficient layouts."""

    return Extensor._from_prepared_kernel(
        value.context, value.gatype.transposed.structural,
        value.context.matrix_inverse(value._kernel),
    )


@Extensor.inverse.register(GATypePattern.map())
def inverse_nonsquare(value: Extensor) -> NoReturn:
    raise ValueError("a unary inverse requires equally sized input/output axes")
