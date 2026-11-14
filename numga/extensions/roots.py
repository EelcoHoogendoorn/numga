"""Principal square roots (`square_root`), inverse square roots (`inverse_square_root`), and rotor
midpoints (`geometric_mean`) dispatch from specialized algebraic closed forms down to general
multivector iteration. Scalars call `xp.sqrt` directly; unit even elements (`ReverseProductOne`)
bisect `rotor + 1` by normalization; scaled even versors factor out `norm()`; nilpotent and general
Study numbers (`scalar + nonscalar` whose nonscalar square is zero or scalar) evaluate the quadratic
root formula in closed form; and general multivectors fall back to the Denman-Beavers coupled Newton
iteration.

Every specialization selects the principal real-algebra branch and leaves negative scalars without
real roots rather than picking an arbitrary bivector plane whose square is `-1`. On even elements
carrying `ReverseProductOne`, `(rotor + 1).normalized()` squares back to `rotor` wherever `rotor + 1`
is normalizable, though in six or more dimensions the chord `rotor + 1` of a versor need not remain a
versor.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import GATypePattern, ReverseProductNonzero, ReverseProductOne, Versor

from ._scalar_functions import register_scalar_function


def _denman_beavers(value: Extensor, iterations: int) -> Extensor:
    product = root = value
    for _ in range(iterations):
        reciprocal = product.inverse()
        root = root * (1 + reciprocal) / 2
        product = (product + reciprocal + 2) / 4
    return root


def _half_reciprocal(root: Extensor) -> Extensor:
    reciprocal = (2 * root).inverse()
    # Zero out NaNs where the scalar root vanishes so zero Study numbers stay finite.
    return Extensor._from_prepared_kernel(
        root.context, reciprocal.gatype.derive.structural, root.context.xp.nan_to_num(reciprocal._kernel, nan=0),
    )


@Extensor.square_root_denman_beavers.register(lambda t: t.is_rotor)
def square_root_denman_beavers(rotor: Extensor, *, n: int = 30) -> Extensor:
    """Denman-Beavers iteration converges to the principal square root of a rotor while preserving
    its normalized traits."""
    root = _denman_beavers(rotor, n)
    return root.with_traits(*root.gatype.normalized_traits)


@Extensor.geometric_mean.register(lambda a, b: a.is_rotor and b.is_rotor)
def geometric_mean(left: Extensor, right: Extensor) -> Extensor:
    """Normalizing the chord `left + right` yields the midpoint rotor halfway between `left` and
    `right`."""
    return (left + right).normalized()


register_scalar_function(Extensor.square_root, lambda context, kernel: context.xp.sqrt(kernel))


@Extensor.square_root.register(
    lambda g: g.is_even and g.entails(ReverseProductOne)
)
def rotor_square_root(rotor: Extensor) -> Extensor:
    """Normalizing `rotor + 1` bisects an even element with unit reverse product wherever `rotor + 1`
    is normalizable; in six or more dimensions the result need not remain a versor."""
    return (rotor + 1).normalized()


@Extensor.square_root.register(
    lambda g: g.is_even and g.entails(Versor)
)
def scaled_rotor_square_root(rotor: Extensor) -> Extensor:
    """Factoring a scaled even versor into its positive scalar `norm()` and unit rotor combines their
    principal square roots."""
    scale = rotor.norm()
    unit = (rotor / scale).with_traits(ReverseProductOne, Versor)
    return (scale.square_root() * unit.square_root()).with_traits(ReverseProductNonzero)


@Extensor.square_root.register(
    lambda g: g.is_study and g.derive.nonscalar.derive.squared.is_empty
)
def nilpotent_study_square_root(value: Extensor) -> Extensor:
    """When the nonscalar part of a Study number squares to zero, linearizing around the scalar
    square root `r` gives `r + nonscalar / (2 * r)`."""
    scalar = value.select[0]
    nonscalar = value.select_subspace(value.gatype.derive.nonscalar.output_subspace)
    root = scalar.square_root()
    return root + nonscalar * _half_reciprocal(root)


@Extensor.square_root.register(lambda g: g.is_study)
def study_square_root(value: Extensor) -> Extensor:
    """The closed-form Study square root computes the scalar part from `(scalar + study_norm) / 2`
    and scales the nonscalar part by `1 / (2 * root)`."""
    scalar = value.select[0]
    nonscalar = value.select_subspace(value.gatype.derive.nonscalar.output_subspace)
    root = ((scalar + value.study_norm()) / 2).square_root()
    return root + nonscalar * _half_reciprocal(root)


register_scalar_function(Extensor.inverse_square_root, lambda context, kernel: kernel ** (-0.5))


@Extensor.inverse_square_root.register(GATypePattern(arity=0))
def inverse_square_root(value: Extensor) -> Extensor:
    """Inverting `value.square_root()` gives the principal inverse square root of a multivector."""
    return value.square_root().inverse()


@Extensor.square_root.register(GATypePattern(arity=0))
def general_square_root(value: Extensor, *, n: int = 30) -> Extensor:
    """Denman-Beavers iteration computes the principal square root of any multivector without
    negative real eigenvalues when no closed-form reduction applies."""
    return _denman_beavers(value, n)
