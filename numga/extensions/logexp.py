"""Original quadratic/bisection exp and log, with whole-GAType dispatch.

The fixed bisection count is caller-controlled. These are the original
real-branch formulas, not an unscaled logarithm series around identity.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import ReverseProductOne, Versor


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
    denominator = m.restrict_subspace(m.gatype.reverse_fixed_subspace)
    return m.bivector_product(denominator.inverse())


@Extensor.log_quadratic.register(lambda t: t <= t.algebra.gatype.rotor())
def log_quadratic(m: Extensor) -> Extensor:
    return m.square_root().log_linear_normalized() * 2


@Extensor.log_pade.register(lambda t: t <= t.algebra.gatype.rotor())
def log_pade(m: Extensor, *, n: int = 25) -> Extensor:
    """Odd series in (m-1)/(m+1); converges near identity."""
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
    return s.select_subspace(s.subspace.canonical).exp()


@Extensor.log.register(lambda t: t.is_reoriented_scalar)
def reoriented_scalar_log(s: Extensor) -> Extensor:
    return s.select_subspace(s.subspace.canonical).log()


@Extensor.exp.register(lambda t: t <= t.algebra.subspace.scalar())
def scalar_exp(s: Extensor) -> Extensor:
    return Extensor._from_prepared_kernel(s.context, s.gatype.structural, s.context.xp.exp(s.kernel))


@Extensor.log.register(lambda t: t <= t.algebra.subspace.scalar())
def scalar_log(s: Extensor) -> Extensor:
    return Extensor._from_prepared_kernel(s.context, s.gatype.structural, s.context.xp.log(s.kernel))


@Extensor.exp.register(
    lambda t: t <= t.algebra.subspace.bivector()
    and t.squared.is_empty
)
def nilpotent_bivector_exp(b: Extensor, *, n: int = 15) -> Extensor:
    """The original translator shortcut: exp(B) = 1+B when B²=0."""

    return (b + 1).with_traits(ReverseProductOne, Versor)


@Extensor.exp_bisect.register(lambda t: t <= t.algebra.gatype.bivector())
@Extensor.exp.register(lambda t: t <= t.algebra.subspace.bivector())
def bivector_exp(b: Extensor, *, n: int = 15) -> Extensor:
    """Quadratic exp of the scaled generator, followed by n squarings."""

    r = 1 + b / 2**(n + 1)
    m = r.squared() / r.symmetric_reverse_product()
    for _ in range(n):
        m = m.squared()
    return m.with_traits(ReverseProductOne, Versor)


@Extensor.log.register(
    lambda t: t <= t.algebra.gatype.rotor()
    and t.is_scalar_bivector
    and t.nonscalar.squared.is_empty
)
def translator_log(m: Extensor, *, n: int = 15) -> Extensor:
    return m.restrict[2]


@Extensor.log.register(lambda t: t <= t.algebra.gatype.rotor())
def unit_versor_log(m: Extensor, *, n: int = 15) -> Extensor:
    """Original motor log: halve first, then apply the quadratic inverse.

    Each square-root step normalizes m+1 as part of that root's formula;
    it does not defensively normalize the supplied motor. The original
    scalar/Study root domain applies; exactly -1 needs a separate branch.
    """

    for _ in range(n + 1):
        m = m.square_root()
    denominator = m.restrict_subspace(m.gatype.reverse_fixed_subspace)
    return m.bivector_product(denominator.inverse()) * 2**(n + 1)


@Extensor.log.register(
    lambda t: t <= t.algebra.subspace.even() and t.entails(Versor)
)
def versor_log(m: Extensor, *, n: int = 15) -> Extensor:
    """Retain log-scale; requires a positive scalar reverse product."""

    scale = m.norm()
    unit = (m / scale).with_traits(ReverseProductOne, Versor)
    return scale.log() + unit.log(n=n)
