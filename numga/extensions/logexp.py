"""Exponentials, logarithms, and differential exponential maps evaluate by scaling and squaring with
whole-`GAType` dispatch across arbitrary dimensions, signatures, and array backends.

`exp` scales its generator down by `2**n`, evaluates an eighth-order Horner Taylor polynomial, and
squares `n` times; `log` extracts `n` principal square roots to bring an element near `1`, evaluates
the matching eighth-order series of `log(1 + y)` in `y = rotor - 1`, and scales back up by `2**n`. At
the default `n = 8`, series truncation stays below double-precision round-off for generators up to
magnitude ten, and `log` inverts `exp` step for step.

Because every step relies strictly on sums, geometric products, and principal square roots, these
implementations hold uniformly across degenerate, Lorentzian, and higher-dimensional Clifford
algebras. Nilpotent generators (`bivector.squared() == 0`, such as pure translations) short-circuit
statically to `1 + bivector`, while dimension-specific closed forms register ahead of these general
fallbacks through opt-in modules such as `numga.extensions.invariant_decomposition` and
`numga.extensions.optimized`.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import GATypePattern, ReverseProductOne, Versor

from ._scalar_functions import register_scalar_function


@Extensor.exp_linear.register(lambda t: t.is_bivector)
def exp_linear(bivector: Extensor) -> Extensor:
    """The first-order exponential `1 + bivector` leaves the result unnormalized."""
    return 1 + bivector


@Extensor.exp_linear_normalized.register(lambda t: t.is_bivector)
def exp_linear_normalized(bivector: Extensor) -> Extensor:
    """The normalized first-order exponential projects `1 + bivector` back onto the rotor manifold."""
    return (1 + bivector).normalized()


@Extensor.exp_cayley.register(lambda t: t.is_bivector)
@Extensor.exp_quadratic.register(lambda t: t.is_bivector)
def exp_quadratic(bivector: Extensor) -> Extensor:
    """The Cayley map `(1 + bivector / 2) / (1 - bivector / 2)` sends bivectors onto unit rotors as
    the exact algebraic inverse of `log_quadratic`."""
    half_step = 1 + bivector / 2
    return (half_step.squared() / half_step.symmetric_reverse_product()).with_traits(*half_step.gatype.normalized_traits)


@Extensor.log_linear.register(lambda t: t.is_even)
def log_linear(even: Extensor) -> Extensor:
    """The first-order logarithm restricts an even multivector to its grade-two bivector part."""
    return even.restrict[2]


@Extensor.log_linear_normalized.register(lambda t: t.is_rotor)
def log_linear_normalized(rotor: Extensor) -> Extensor:
    """The projective first-order logarithm divides `rotor`'s grade-two part by its reversal-fixed
    part, inverting `exp_linear_normalized` up to scalar scale."""
    denominator = rotor.restrict_subspace(rotor.gatype.derive.reverse_fixed_subspace)
    return rotor.bivector_product(denominator.inverse())


@Extensor.log_quadratic.register(lambda t: t.is_rotor)
def log_quadratic(rotor: Extensor) -> Extensor:
    """The Cayley inverse logarithm `2 * rotor.square_root().log_linear_normalized()` inverts
    `exp_quadratic`."""
    return rotor.square_root().log_linear_normalized() * 2


# TODO: review this port. It sums the series 2 artanh((rotor - 1) / (rotor + 1)), rather than a
# Pade approximant, and dispatches on rotors where legacy motor_log_pade accepted any even element.
@Extensor.log_pade.register(lambda t: t.is_rotor)
def log_pade(rotor: Extensor, *, n: int = 25) -> Extensor:
    """Truncating `2 * arctanh((rotor - 1) / (rotor + 1))` to `n` odd powers approximates the bivector
    logarithm near the identity."""
    cayley = (rotor - 1) / (rotor + 1)
    square = cayley.squared()
    power, result = cayley, cayley * 2
    for step in range(1, n):
        power = power * square
        result = result + power * 2 / (2 * step + 1)
    return result.restrict[2]


@Extensor.exp.register(lambda t: t.is_empty)
def empty_exp(empty: Extensor) -> Extensor:
    return (empty + 1).with_traits(ReverseProductOne, Versor)


@Extensor.log.register(lambda t: t.is_empty)
def empty_log(empty: Extensor) -> Extensor:
    return (empty + 0).log()


register_scalar_function(Extensor.exp, lambda context, kernel: context.xp.exp(kernel))
register_scalar_function(Extensor.log, lambda context, kernel: context.xp.log(kernel))


@Extensor.exp.register(lambda t: t.is_bivector and t.is_nilpotent)
def nilpotent_bivector_exp(bivector: Extensor, *, n: int = 8) -> Extensor:
    """When a bivector's `GAType` is statically nilpotent (such as a pure translation), its
    exponential equals `1 + bivector` identically."""
    return (bivector + 1).with_traits(ReverseProductOne, Versor)


@Extensor.exp.register(lambda t: t.is_nilpotent)
def nilpotent_exp(element: Extensor) -> Extensor:
    """When an element squares statically to zero (such as the PGA pseudoscalar), its exponential
    truncates exactly at `1 + element`."""
    return element + 1


@Extensor.exp_bisect.register(lambda t: t.is_bivector)
@Extensor.exp.register(lambda t: t.is_bivector)
def bivector_exp(bivector: Extensor, *, n: int = 8) -> Extensor:
    """Scaling `bivector` by `2**-n`, normalizing the eighth-order Taylor step, and squaring `n`
    times produces a normalized rotor that stays bounded on the Spin group even far outside the
    base series radius."""
    rotor = _taylor_exp(bivector / 2**n).normalized()
    for _ in range(n):
        rotor = rotor.squared()
    return rotor.normalized().with_traits(ReverseProductOne, Versor)


@Extensor.exp_derivative.register(lambda t: t.is_bivector)
def exp_derivative(bivector: Extensor, *, n: int = 8) -> Extensor:
    """The right-trivialized differential of `exp` at `bivector` returns the operator
    `Bivector <- Bivector` satisfying
    `(bivector + delta * h).exp() == bivector.exp() * (1 + bivector.exp_derivative()(delta) * h)`
    to first order in `h`.

    Because the operator equals the integral of `(bivector * s).exp() << Bivector` over `s` in
    `[0, 1]`, halving `bivector` splits that integral into two halves related by the half-turn
    `(bivector / 2).exp()`, evaluated by `n` halvings from the eighth-order series in `ad` followed
    by `n` doubling recurrences."""
    Bivector = bivector.algebra.gatype.bivector()
    small = bivector / 2**n
    # Commutator operator X -> small * X - X * small on Bivector.
    ad = small.commutator(Bivector) * 2                                # [...] Bivector <- Bivector
    # Horner evaluation of 1 - ad / 2! + ad^2 / 3! - ... up to _ORDER.
    mean = Bivector
    for order in range(_ORDER, 1, -1):
        mean = Bivector - ad(mean) / order                             # [...] Bivector <- Bivector
    turn = _exp_series(small)
    for _ in range(n):
        mean = ((turn << Bivector) + Bivector)(mean) / 2
        turn = turn.squared().with_traits(ReverseProductOne, Versor)
    return mean


_ORDER = 8


def _exp_series(bivector: Extensor) -> Extensor:
    return _taylor_exp(bivector).with_traits(ReverseProductOne, Versor)


def _log_series(rotor: Extensor) -> Extensor:
    logarithm = _taylor_log(rotor)
    return logarithm.restrict_subspace(
        logarithm.subspace.difference(logarithm.gatype.derive.reverse_fixed_subspace),
    )


def _taylor_exp(element: Extensor) -> Extensor:
    result = 1 + element / _ORDER
    for order in range(_ORDER - 1, 0, -1):
        result = 1 + element * result / order
    return result


def _taylor_log(element: Extensor) -> Extensor:
    offset = element - 1
    result = offset / _ORDER
    for order in range(_ORDER - 1, 0, -1):
        result = offset * (1 / order - result)
    return result


@Extensor.log.register(
    lambda t: t.is_rotor and t.is_scalar_bivector and t.derive.nonscalar.is_nilpotent
)
def translator_log(translator: Extensor, *, n: int = 8) -> Extensor:
    """When a rotor's bivector part is statically nilpotent, its logarithm is simply `translator.restrict[2]`."""
    return translator.restrict[2]


@Extensor.log.register(
    lambda t: t <= t.algebra.subspace.even() and t.entails(ReverseProductOne)
)
def unit_versor_log(rotor: Extensor, *, n: int = 8) -> Extensor:
    """Extracting `n` principal square roots brings a unit even versor near `1`, where the
    reversal-antisymmetric part of the eighth-order Taylor logarithm scaled by `2**n` recovers its
    generator.

    Below six dimensions the reversal-antisymmetric subspace is purely grade two, whereas in six or
    more dimensions the principal Clifford logarithm can also carry grade-six components when plane
    angles cross `pi`."""
    for _ in range(n):
        rotor = rotor.square_root()
    return _log_series(rotor) * 2**n


@Extensor.log.register(
    lambda t: t <= t.algebra.subspace.even() and t.entails(Versor)
)
def versor_log(versor: Extensor, *, n: int = 8) -> Extensor:
    """Separating `scale = versor.norm()` from `unit = versor / scale` returns `scale.log() + unit.log(n=n)`
    for even versors with positive scalar reverse product."""
    scale = versor.norm()
    unit = (versor / scale).with_traits(ReverseProductOne, Versor)
    return scale.log() + unit.log(n=n)


@Extensor.exp.register(GATypePattern(arity=0))
def general_exp(value: Extensor, *, n: int = 8) -> Extensor:
    """The general multivector exponential scales `value` down by `2**n`, evaluates the Taylor
    series, and squares `n` times inside the subalgebra generated by `value`."""
    result = _taylor_exp(value / 2**n)
    for _ in range(n):
        result = result.squared()
    return result


@Extensor.log.register(GATypePattern(arity=0))
def general_log(value: Extensor, *, n: int = 8) -> Extensor:
    """The general multivector logarithm takes `n` principal square roots and scales the eighth-order
    Taylor logarithm around `1` by `2**n`, inverting `general_exp` step for step wherever the
    principal roots exist."""
    for _ in range(n):
        value = value.square_root()
    return _taylor_log(value) * 2**n
