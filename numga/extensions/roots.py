"""Scalar, generalized Study, and unit or scaled rotor square roots.

The formulas are real-algebra branches: they select one root, and do not
search other blades for roots of negative scalars.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import GATypePattern, ReverseProductNonzero, ReverseProductOne, Versor

from ._scalar_functions import register_scalar_function


def _denman_beavers(value: Extensor, n: int) -> Extensor:
    """The Denman-Beavers iteration for the principal square root."""
    product = root = value
    for _ in range(n):
        reciprocal = product.inverse()
        root = root * (1 + reciprocal) / 2
        product = (product + reciprocal + 2) / 4
    return root


def _half_reciprocal(root: Extensor) -> Extensor:
    """The reciprocal of twice the root, zero where the root is zero."""
    reciprocal = (2 * root).inverse()
    return Extensor._from_prepared_kernel(
        root.context, reciprocal.gatype.derive.structural, root.context.xp.nan_to_num(reciprocal._kernel, nan=0),
    )


@Extensor.square_root_denman_beavers.register(lambda t: t.is_rotor)
def square_root_denman_beavers(value: Extensor, *, n: int = 30) -> Extensor:
    root = _denman_beavers(value, n)
    return root.with_traits(*root.gatype.normalized_traits)


@Extensor.geometric_mean.register(lambda a, b: a.is_rotor and b.is_rotor)
def geometric_mean(left: Extensor, right: Extensor) -> Extensor:
    return (left + right).normalized()


register_scalar_function(Extensor.square_root, lambda context, kernel: context.xp.sqrt(kernel))


@Extensor.square_root.register(
    lambda g: g.is_even and g.entails(ReverseProductOne)
)
def rotor_square_root(value: Extensor) -> Extensor:
    """The normalized bisector of an even element with unit reverse product.

    Its square is the input where the normalization exists. In six or more dimensions
    this root need not be a versor, even when the input is one.
    """

    return (value + 1).normalized()


@Extensor.square_root.register(
    lambda g: g.is_even and g.entails(Versor)
)
def scaled_rotor_square_root(value: Extensor) -> Extensor:
    """Root of a scaled rotor; requires a positive scalar reverse product."""

    scale = value.norm()
    unit = (value / scale).with_traits(ReverseProductOne, Versor)
    return (scale.square_root() * unit.square_root()).with_traits(ReverseProductNonzero)


@Extensor.square_root.register(
    lambda g: g.is_study and g.derive.nonscalar.derive.squared.is_empty
)
def nilpotent_study_square_root(value: Extensor) -> Extensor:
    scalar = value.select[0]
    nonscalar = value.select_subspace(value.gatype.derive.nonscalar.output_subspace)
    root = scalar.square_root()
    return root + nonscalar * _half_reciprocal(root)


@Extensor.square_root.register(lambda g: g.is_study)
def study_square_root(value: Extensor) -> Extensor:
    scalar = value.select[0]
    nonscalar = value.select_subspace(value.gatype.derive.nonscalar.output_subspace)
    root = ((scalar + value.study_norm()) / 2).square_root()
    return root + nonscalar * _half_reciprocal(root)


register_scalar_function(Extensor.inverse_square_root, lambda context, kernel: kernel ** (-0.5))


@Extensor.inverse_square_root.register(GATypePattern(arity=0))
def inverse_square_root(value: Extensor) -> Extensor:
    return value.square_root().inverse()


@Extensor.square_root.register(GATypePattern(arity=0))
def general_square_root(value: Extensor, *, n: int = 30) -> Extensor:
    """The principal square root of any multivector without one on the negative scalars, by the
    Denman–Beavers iteration. Tried after every closed form."""
    return _denman_beavers(value, n)
