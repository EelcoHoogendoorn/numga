"""Algebraic inverses dispatch on static `GAType` structure and traits, preferring closed-form
grade-transform reductions before falling back to subalgebra linear systems.

Unit traits invert without division (`ReverseProductOne` via `reverse()`,
`CliffordConjugateProductOne` via `clifford_conjugate()`, and `CoefficientOrthogonal` maps via
coefficient matrix transpose), while scalars take direct reciprocals without squaring. Below six
dimensions, chains of symmetric grade-transform self-products (`squared`, `symmetric_reverse`,
`symmetric_conjugate`, `symmetric_scalar_negation`, `symmetric_pseudoscalar_negation`, and
`symmetric_involute`) reduce any invertible multivector's denominator to a scalar in at most three
steps; shorter reductions are registered ahead of longer ones, with exact grade-cancelling
specializations for five-dimensional bivectors and trivectors.

When no static scalar reduction applies, general inversion solves left multiplication `value * x == 1`
inside the minimal blade-generated subalgebra on numerical backends. `inverse_shirokov` and
`inverse_hitzer` expose explicit characteristic-polynomial and Hitzer factor formulas for direct
comparison. Inversion dispatch and minimal-subalgebra fallback are described in `docs/internals.md`
and `docs/tutorial/geometric-algebra/13-inverses.md`.
"""

from __future__ import annotations

from typing import NoReturn

from numga.extensor import Extensor
from numga.gatype import (
    CliffordConjugateProductOne,
    CoefficientOrthogonal,
    GATypePattern,
    ReverseProductOne,
    ReverseProductZero,
)
from numga.units import invert_unit_law

Extensor.inverse.set_unit(invert_unit_law)


@Extensor.inverse.register(lambda t: t.arity == 0 and t.entails(ReverseProductZero))
def inverse_null(value: Extensor) -> NoReturn:
    raise ZeroDivisionError("a statically null multivector has no inverse")


@Extensor.inverse.register(lambda t: t.entails(ReverseProductOne))
def inverse_unit_reverse(value: Extensor) -> Extensor:
    return value.reverse()


@Extensor.inverse.register(lambda t: t.entails(CliffordConjugateProductOne))
def inverse_unit_conjugate(value: Extensor) -> Extensor:
    return value.clifford_conjugate()


@Extensor.inverse.register(lambda t: t.entails(CoefficientOrthogonal))
def inverse_orthogonal(operator: Extensor) -> Extensor:
    """A coefficient-orthogonal map inverts by transposing its output and input blade axes."""
    permutation = tuple(range(operator.ndim)) + (operator.ndim + 1, operator.ndim)
    return Extensor._from_prepared_kernel(
        operator.context, operator.gatype.derive.transposed, operator.context.xp.transpose(operator._kernel, permutation),
    )


@Extensor.inverse.register(lambda t: t.is_scalar)
def inverse_scalar(value: Extensor) -> Extensor:
    """Scalar inversion takes the direct reciprocal of the coefficient without squaring."""
    gatype = value.gatype.derive.structural.derive.with_traits(*value.gatype.inverse_traits)
    return Extensor._from_prepared_kernel(
        value.context, gatype, value.context.reciprocal(value._kernel),
    )


def register_reductions(steps: int) -> None:
    """Registering self-product reductions of length `steps` in transform-preference order ensures
    shorter chains always take precedence."""

    @Extensor.inverse.register(
        lambda t: t.derive.squared.reduces_to_scalar(steps - 1)
    )
    def inverse_squared(value: Extensor) -> Extensor:
        return (value / value.squared()).with_traits(*value.gatype.inverse_traits)

    @Extensor.inverse.register(
        lambda t: t.derive.symmetric_reverse.reduces_to_scalar(steps - 1)
    )
    def inverse_reverse(value: Extensor) -> Extensor:
        return (value.reverse() / value.symmetric_reverse_product()).with_traits(
            *value.gatype.inverse_traits,
        )

    @Extensor.inverse.register(
        lambda t: t.derive.symmetric_conjugate.reduces_to_scalar(steps - 1)
    )
    def inverse_conjugate(value: Extensor) -> Extensor:
        result = value.clifford_conjugate() / value.symmetric_conjugate_product()
        return result.with_traits(*value.gatype.inverse_traits)

    @Extensor.inverse.register(
        lambda t: t.derive.symmetric_scalar_negation.reduces_to_scalar(steps - 1)
    )
    def inverse_scalar_negation(value: Extensor) -> Extensor:
        result = value.scalar_negation() / value.symmetric_scalar_negation_product()
        return result.with_traits(*value.gatype.inverse_traits)

    @Extensor.inverse.register(
        lambda t: t.derive.symmetric_pseudoscalar_negation.reduces_to_scalar(steps - 1)
    )
    def inverse_pseudoscalar_negation(value: Extensor) -> Extensor:
        result = value.pseudoscalar_negation() / value.symmetric_pseudoscalar_negation_product()
        return result.with_traits(*value.gatype.inverse_traits)

    @Extensor.inverse.register(
        lambda t: t.derive.symmetric_involute.reduces_to_scalar(steps - 1)
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
def inverse_bivector_5d(bivector: Extensor) -> Extensor:
    """In five dimensions the grade-four part of `bivector * (-bivector.symmetric_reverse_product().inverse())`
    cancels identically, keeping the inverse strictly within the bivector subspace."""
    return bivector.bivector_product(-bivector.symmetric_reverse_product().inverse())


@Extensor.inverse.register(
    lambda t: t.algebra.dimension == 5
    and t <= t.algebra.subspace.trivector()
)
def inverse_trivector_5d(trivector: Extensor) -> Extensor:
    """In five dimensions a trivector's inverse stays strictly grade-three via `trivector_product`
    with `trivector.squared().inverse()`."""
    return trivector.trivector_product(trivector.squared().inverse())


register_reductions(2)
register_reductions(3)


@Extensor.inverse.register(GATypePattern(arity=0))
def inverse_geometric(value: Extensor) -> Extensor:
    """General multivector inversion solves `value * x == 1` as a linear system restricted to the
    minimal subalgebra generated by `value`'s blades."""
    subalgebra = value.gatype.derive.minimal_subalgebra
    space = subalgebra.output_subspace
    left_multiply = (value * subalgebra).select_subspace(space)
    unit = value.algebra.operator.unit(subalgebra)
    coefficients = value.context.solve(left_multiply._kernel, value.context.lower(unit)._kernel)
    # Because value * value.inverse() equals the extrinsic scalar 1, the inverse shares value's orientation.
    gatype = subalgebra.derive.with_orientations((value.gatype.orientation,)).derive.with_traits(*value.gatype.inverse_traits)
    return Extensor._from_prepared_kernel(
        value.context, gatype, coefficients,
    )


@Extensor.inverse_shirokov.register(GATypePattern(arity=0))
def inverse_shirokov(value: Extensor) -> Extensor:
    """Shirokov's Faddeev-LeVerrier recurrence evaluates the adjugate and determinant from traces of
    powers up to `2 ** ((dimension + 1) // 2)`, trading numerical conditioning in high dimensions
    for purely multiplicative steps."""
    order = 2 ** ((value.algebra.dimension + 1) // 2)
    power = value
    adjugate = value.context.multivector.scalar()
    for step in range(1, order):
        adjugate = power - power.select[0] * (order / step)
        power = value * adjugate
    return adjugate / power.select[0]


@Extensor.inverse_factor.register(GATypePattern(arity=0))
def inverse_factor(value: Extensor) -> Extensor:
    """Hitzer's cofactor `~value * (value * ~value).clifford_conjugate()` multiplies `value` into a
    scalar below six dimensions."""
    return ~value * value.symmetric_reverse_product().clifford_conjugate()


@Extensor.inverse_hitzer.register(
    lambda t: t.arity == 0 and t.derive.symmetric_reverse.derive.symmetric_conjugate.is_scalar
)
def inverse_hitzer(value: Extensor) -> Extensor:
    """Dividing `value.inverse_factor()` by `value.scalar_product(factor)` inverts `value` whenever
    the reverse-then-conjugate product reduces statically to a scalar."""
    factor = value.inverse_factor()
    return factor / value.scalar_product(factor)
