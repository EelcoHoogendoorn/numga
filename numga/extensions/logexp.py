"""Exp and log by scaling and squaring, with whole-GAType dispatch.

exp takes the eighth-order Taylor step of the generator scaled down by 2**n and squares it n
times; log takes n square roots and the eighth-order series of the logarithm around the identity,
the inverse step, scaled back up by 2**n. With n = 8 the truncation stays below round-off for
generators up to a size of about ten, and more steps only add round-off. n is the caller's.

These are the reference implementations, kept conceptually simple: built from sums and products
only, they hold for every signature and every backend, and the log undoes the exp step for step.
Closed forms, special cases and backend-specific versions belong in opt-in modules with their own
register(), such as `invariant_decomposition` and `optimized`.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import GATypePattern, ReverseProductOne, Versor


@Extensor.exp_linear.register(lambda t: t <= t.algebra.gatype.bivector())
def exp_linear(b: Extensor) -> Extensor:
    """First-order exponential; the result is not normalized."""
    return 1 + b


@Extensor.exp_linear_normalized.register(lambda t: t <= t.algebra.gatype.bivector())
def exp_linear_normalized(b: Extensor) -> Extensor:
    return (1 + b).normalized()


@Extensor.exp_cayley.register(lambda t: t <= t.algebra.gatype.bivector())
@Extensor.exp_quadratic.register(lambda t: t <= t.algebra.gatype.bivector())
def exp_quadratic(b: Extensor) -> Extensor:
    """Cayley exponential approximation, inverse of log_quadratic."""
    r = 1 + b / 2
    return (r.squared() / r.symmetric_reverse_product()).with_traits(*r.gatype.normalized_traits)


@Extensor.log_linear.register(lambda t: t <= t.algebra.gatype.even())
def log_linear(m: Extensor) -> Extensor:
    return m.restrict[2]


@Extensor.log_linear_normalized.register(lambda t: t <= t.algebra.gatype.rotor())
def log_linear_normalized(m: Extensor) -> Extensor:
    denominator = m.restrict_subspace(m.gatype.derive.reverse_fixed_subspace)
    return m.bivector_product(denominator.inverse())


@Extensor.log_quadratic.register(lambda t: t <= t.algebra.gatype.rotor())
def log_quadratic(m: Extensor) -> Extensor:
    return m.square_root().log_linear_normalized() * 2


# TODO: review this port. It sums the series 2 artanh((m-1)/(m+1)), not a Pade approximant, and
#  dispatches on rotors where the legacy motor_log_pade took any even multivector.
@Extensor.log_pade.register(lambda t: t <= t.algebra.gatype.rotor())
def log_pade(m: Extensor, *, n: int = 25) -> Extensor:
    """Odd series in `(m - 1) / (m + 1)`; converges near identity."""
    f = (m - 1) / (m + 1)
    square = f.squared()
    power, result = f, f * 2
    for k in range(1, n):
        power = power * square
        result = result + power * 2 / (2*k + 1)
    return result.restrict[2]


@Extensor.exp.register(lambda t: t <= t.algebra.subspace.empty())
def empty_exp(z: Extensor) -> Extensor:
    return (z + 1).with_traits(ReverseProductOne, Versor)


@Extensor.log.register(lambda t: t <= t.algebra.subspace.empty())
def empty_log(z: Extensor) -> Extensor:
    return (z + 0).log()


@Extensor.exp.register(lambda t: t.is_reoriented_scalar)
def reoriented_scalar_exp(s: Extensor) -> Extensor:
    """exp of a scalar stored against the basis element -1, as a signed layout can store it: the
    value is read back against +1 first, so exp acts on the scalar and not on its negation."""

    return s.select_subspace(s.subspace.canonical).exp()


@Extensor.log.register(lambda t: t.is_reoriented_scalar)
def reoriented_scalar_log(s: Extensor) -> Extensor:
    """log of a scalar stored against the basis element -1, read back against +1 first."""

    return s.select_subspace(s.subspace.canonical).log()


@Extensor.exp.register(lambda t: t <= t.algebra.subspace.scalar())
def scalar_exp(s: Extensor) -> Extensor:
    return Extensor._from_prepared_kernel(s.context, s.gatype.derive.structural, s.context.xp.exp(s.kernel))


@Extensor.log.register(lambda t: t <= t.algebra.subspace.scalar())
def scalar_log(s: Extensor) -> Extensor:
    return Extensor._from_prepared_kernel(s.context, s.gatype.derive.structural, s.context.xp.log(s.kernel))


@Extensor.exp.register(
    lambda t: t <= t.algebra.subspace.bivector()
    and t.derive.squared.is_empty
)
def nilpotent_bivector_exp(b: Extensor, *, n: int = 8) -> Extensor:
    """`b.exp() == 1 + b` when `b` squares to zero by its type, as for a translation."""

    return (b + 1).with_traits(ReverseProductOne, Versor)


@Extensor.exp_bisect.register(lambda t: t <= t.algebra.gatype.bivector())
@Extensor.exp.register(lambda t: t <= t.algebra.subspace.bivector())
def bivector_exp(b: Extensor, *, n: int = 8) -> Extensor:
    """The Taylor step of the generator scaled down by 2**n, followed by n squarings, the step and
    the result each normalized. Within the step's range the normalizations change nothing above
    round-off; beyond it the result stays a rotor, at the wrong angle, where the bare series and
    its squarings grow without bound."""

    m = _taylor_exp(b / 2**n).normalized()
    for _ in range(n):
        m = m.squared()
    return m.normalized().with_traits(ReverseProductOne, Versor)


@Extensor.exp_derivative.register(lambda t: t <= t.algebra.subspace.bivector())
def exp_derivative(b: Extensor, *, n: int = 8) -> Extensor:
    """The derivative of exp at `b`, carried back to the identity: the map `Bivector <- Bivector`
    with `(b + db * h).exp() == b.exp() * (1 + b.exp_derivative()(db) * h)` to first order in `h`.

    It is the mean of the turns `(b * s).exp() << Bivector` over `s` from 0 to 1. Halving `b`
    splits that mean in two, the second half the first turned by `(b / 2).exp()`, so the mean at
    `b` is `((b / 2).exp() << Bivector + Bivector)(mean at b / 2) / 2`. n halvings bring `b` down
    to where the mean is its series in the commutator with the open type, `ad`,
    `Bivector - ad / 2 + ad(ad) / 6 - ...` to the order of exp's step; the turns of the halved
    generators are the squarings of its exponential, as in exp itself. A boost's turns grow like
    the exponential of twice its size, and the round-off with them."""

    Bivector = b.algebra.gatype.bivector()
    small = b / 2**n
    # the commutator product with the generator, small * X - X * small
    ad = small.commutator(Bivector) * 2                                # [...] Bivector <- Bivector
    # the series 1 - ad / 2! + ad(ad) / 3! - ..., by Horner: 1 - ad / 2 (1 - ad / 3 (1 - ...))
    mean = Bivector
    for k in range(_ORDER, 1, -1):
        mean = Bivector - ad(mean) / k                                 # [...] Bivector <- Bivector
    turn = _exp_series(small)
    for _ in range(n):
        mean = ((turn << Bivector) + Bivector)(mean) / 2
        turn = turn.squared().with_traits(ReverseProductOne, Versor)
    return mean


# The order of the series steps of exp and log.
_ORDER = 8


def _exp_series(x: Extensor) -> Extensor:
    """exp of a small bivector by its Taylor series, a rotor."""
    return _taylor_exp(x).with_traits(ReverseProductOne, Versor)


def _log_series(m: Extensor) -> Extensor:
    """The bivector log of a rotor near the identity."""
    return _taylor_log(m).restrict[2]


def _taylor_exp(x: Extensor) -> Extensor:
    """exp of a small multivector by its Taylor series, by Horner: 1 + x (1 + x / 2 (1 + x / 3 (...)))."""
    result = 1 + x / _ORDER
    for k in range(_ORDER - 1, 0, -1):
        result = 1 + x * result / k
    return result


def _taylor_log(m: Extensor) -> Extensor:
    """log of a multivector near one by the series of log(1 + y) in y = m - 1, by Horner:
    y (1 - y (1 / 2 - y (1 / 3 - ...)))."""
    y = m - 1
    result = y / _ORDER
    for k in range(_ORDER - 1, 0, -1):
        result = y * (1 / k - result)
    return result


@Extensor.exp.register(lambda t: t.derive.squared.is_empty)
def nilpotent_exp(x: Extensor) -> Extensor:
    """`x.exp() == 1 + x` when `x` squares to zero by its type, as the pseudoscalar of PGA does."""

    return x + 1


@Extensor.exp.register(lambda t: t.derive.squared.is_scalar)
def scalar_square_exp(x: Extensor) -> Extensor:
    """`x.exp() == even + x * odd`, for scalars `even` and `odd`, when `x` squares to a scalar `s`,
    as a pseudoscalar or a single blade does. With `r = xp.sqrt(xp.abs(s))` they are `xp.cosh(r)`
    and `xp.sinh(r) / r` for `s > 0`, `xp.cos(r)` and `xp.sin(r) / r` for `s < 0`, and 1 and 1 for
    `s == 0`. No versor trait is asserted: the exponential of a multiple of the pseudoscalar in four
    dimensions is not a versor."""

    xp = x.context.xp
    square = x.squared().kernel[..., 0]                                # [...] the scalar s
    root = xp.sqrt(xp.abs(square))
    safe = xp.where(root > 0, root, 1)
    even = xp.where(square > 0, xp.cosh(root), xp.cos(root))
    odd = xp.where(root > 0, xp.where(square > 0, xp.sinh(root), xp.sin(root)) / safe, 1)
    scalar = x.context.multivector.scalar
    return scalar(even[..., None]) + x * scalar(odd[..., None])


@Extensor.log.register(
    lambda t: t <= t.algebra.gatype.rotor()
    and t.is_scalar_bivector
    and t.derive.nonscalar.derive.squared.is_empty
)
def translator_log(m: Extensor, *, n: int = 8) -> Extensor:
    return m.restrict[2]


@Extensor.log.register(lambda t: t <= t.algebra.gatype.rotor())
def unit_versor_log(m: Extensor, *, n: int = 8) -> Extensor:
    """Unit motor log: halve by n square roots first, then the series of the logarithm, scaled back.

    Each square-root step normalizes m + 1 as part of that root's formula; the supplied motor
    is not normalized. The domain is that of the scalar and Study roots: exactly -1 would need
    a separate branch.
    """

    for _ in range(n):
        m = m.square_root()
    return _log_series(m) * 2**n


@Extensor.log.register(
    lambda t: t <= t.algebra.subspace.even() and t.entails(Versor)
)
def versor_log(m: Extensor, *, n: int = 8) -> Extensor:
    """Retain log-scale; requires a positive scalar reverse product."""

    scale = m.norm()
    unit = (m / scale).with_traits(ReverseProductOne, Versor)
    return scale.log() + unit.log(n=n)


@Extensor.exp.register(GATypePattern(arity=0))
def general_exp(x: Extensor, *, n: int = 8) -> Extensor:
    """Any multivector: the Taylor step of `x` scaled down by 2**n, followed by n squarings. Its
    powers stay in the subalgebra it generates, so the result's type closes on its own. Tried
    after every closed form and special case."""

    m = _taylor_exp(x / 2**n)
    for _ in range(n):
        m = m.squared()
    return m


@Extensor.log.register(GATypePattern(arity=0))
def general_log(m: Extensor, *, n: int = 8) -> Extensor:
    """Any multivector with a principal square root: n square roots bring it near one, then the
    series of the logarithm, scaled back up by 2**n, the inverse of `general_exp` step for step."""

    for _ in range(n):
        m = m.square_root()
    return _taylor_log(m) * 2**n
