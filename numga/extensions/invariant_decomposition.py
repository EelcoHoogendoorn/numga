"""Opt-in closed-form exp and log below six dimensions by invariant decomposition; call register().

After Roelfs and De Keninck, "Normalization, Square Roots, and the Exponential and Logarithmic
Maps in Geometric Algebras of Less than 6D" (arXiv:2206.07496), equation numbers theirs.

Up to three dimensions a bivector squares to a scalar, and exp and log are Euler's formula and
its inverse. In four and five dimensions the square `b.squared()` of a bivector `b` is a Study
number: a scalar plus the 4-vector `b ^ b`, whose square is a scalar, with Study norm
`b.squared().study_norm()` (eq. 18). `b.decompose_invariant()` splits `b` into two commuting
simple bivectors, `b_plus = (1 + b.squared().scalar_negation() / b.squared().study_norm()) * b / 2`
and `b_minus` with the opposite sign (eqs. 33-35). They square to the scalars `(dot + norm) / 2`
and `(dot - norm) / 2`, with `dot` the scalar part of `b.squared()` and `norm` its Study norm,
and `b.exp() == b_plus.exp() * b_minus.exp()` (eq. 37), each factor by Euler's formula.

The logarithm works backwards (section 7): the bivector part `r.restrict[2]` of a rotor `r`
splits the same way, the squares of its parts give the two angles by `xp.arctan2` against the
scalar part of `r` (`xp.arctanh` for a boost), and `r.log()` is
`r.restrict[2] * (r.restrict[4] * beta + alpha)`, with `alpha` and `r.restrict[4] * beta` from
eqs. 43-44. Of the two factorizations of `r` into commuting simple rotors, which differ by
negating both factors, the one whose first factor has a nonnegative scalar part is taken, so
that both angles follow from the scalar part of `r` in any algebra.

Static structure selects simpler paths: up to three dimensions Euler's formula; where the
4-vectors square to zero, as in PGA, one part is always null and the decomposition collapses to a
closed form without branches. The decomposition has complex parts only in algebras with at least
two positive and two negative directions, which the paper leaves open; those keep the generic
methods.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import ReverseProductOne, Versor


# --- scalar functions --------------------------------------------------------------------------
# Each takes the backend namespace and an array of real arguments. At the removable singularity
# of a closed form a Taylor series takes over, and the division there is kept off zero.

_SMALL = 1e-3


def _polynomial(x, coefficients):
    result = 0
    for coefficient in reversed(coefficients):
        result = result * x + coefficient
    return result


_C_SERIES = (1, 1 / 2, 1 / 24, 1 / 720, 1 / 40320)
_S_SERIES = (1, 1 / 6, 1 / 120, 1 / 5040, 1 / 362880)
_H_SERIES = (1, -1 / 3, 2 / 15, -2 / 35, 8 / 315)          # in powers of p - 1


def cosine(xp, z):
    """`xp.cosh(xp.sqrt(z))`, which is `xp.cos(xp.sqrt(-z))` for `z < 0`."""
    root = xp.sqrt(xp.abs(z))
    return xp.where(xp.abs(z) < _SMALL, _polynomial(z, _C_SERIES), xp.where(z < 0, xp.cos(root), xp.cosh(root)))


def sine(xp, z):
    """`xp.sinh(xp.sqrt(z)) / xp.sqrt(z)`, which is `xp.sin(xp.sqrt(-z)) / xp.sqrt(-z)` for
    `z < 0`."""
    small = xp.abs(z) < _SMALL
    root = xp.sqrt(xp.abs(z))
    return xp.where(small, _polynomial(z, _S_SERIES),
                    xp.where(z < 0, xp.sin(root), xp.sinh(root)) / xp.where(small, 1, root))


def angle(xp, p):
    """`xp.arccosh(p) / xp.sqrt(p**2 - 1)`, which is `xp.arccos(p) / xp.sqrt(1 - p**2)` for
    `p < 1`."""
    small = xp.abs(p - 1) < _SMALL
    root = xp.where(small, 1, xp.sqrt(xp.abs(1 - p * p)))
    # for p >= 1, xp.arccosh(p) == xp.log(p + xp.sqrt(p**2 - 1))
    return xp.where(small, _polynomial(p - 1, _H_SERIES),
                    xp.where(p < 1, xp.arctan2(root, p), xp.log(xp.abs(p) + root)) / root)


def sine_derivative(xp, z):
    """The derivative of `sine` in `z`, `(cosine(xp, z) - sine(xp, z)) / (2 * z)`, which tends to
    1/6 as `z` goes to zero."""
    small = xp.abs(z) < _SMALL
    return xp.where(small, _polynomial(z, (1 / 6, 1 / 60, 1 / 1680, 1 / 90720)),
                    (cosine(xp, z) - sine(xp, z)) / (2 * xp.where(small, 1, z)))


# --- the decomposition ------------------------------------------------------------------------

def _scalar(value: Extensor):
    return value.select_subspace(value.algebra.subspace.scalar())._kernel[..., 0]


def _square(b: Extensor):
    """The scalar part `dot` of `b.squared()`, its 4-vector part `wedge`, which is `b ^ b`, and its
    Study norm `norm` (eq. 18); the parts of `b` square to `(dot + norm) / 2` and
    `(dot - norm) / 2`."""
    xp = b.context.xp
    square = b.squared()
    dot = _scalar(square)
    wedge = square.restrict_subspace(b.algebra.subspace.k_vector(4))
    return dot, wedge, xp.sqrt(dot * dot - _scalar(wedge * wedge))


def _where(condition, chosen: Extensor, otherwise: Extensor) -> Extensor:
    """chosen where condition holds, otherwise elsewhere, in otherwise's type."""
    xp, space = otherwise.context.xp, otherwise.output_subspace
    kernel = xp.where(condition[..., None], chosen.select_subspace(space)._kernel, otherwise._kernel)
    return Extensor._from_prepared_kernel(otherwise.context, otherwise.gatype, kernel)


def exp_simple(b: Extensor) -> Extensor:
    """`b.exp()` by Euler's formula, `cosine(xp, a) + b * sine(xp, a)` with `a` the scalar
    `b.squared()`, for `b` squaring to a scalar."""
    xp = b.context.xp
    square = _scalar(b.squared())
    return (b * sine(xp, square) + cosine(xp, square)).with_traits(ReverseProductOne, Versor)


def exp_null_wedge(b: Extensor) -> Extensor:
    """`b.exp()` where `b ^ b` squares to zero, as in PGA: one part is null and the other squares
    to the scalar part `a` of `b.squared()`, and with `wedge = b ^ b` the product of their
    exponentials is `cosine(xp, a) + b * sine(xp, a) + b * wedge * sine_derivative(xp, a)
    + wedge * sine(xp, a) / 2`."""
    xp, algebra = b.context.xp, b.algebra
    square = b.squared()
    a = _scalar(square)
    wedge = square.restrict_subspace(algebra.subspace.k_vector(4))
    return (
        b * sine(xp, a) + (b * wedge).restrict_subspace(algebra.subspace.bivector()) * sine_derivative(xp, a)
        + wedge * (sine(xp, a) / 2) + cosine(xp, a)
    ).with_traits(ReverseProductOne, Versor)


def exp_decomposed(b: Extensor) -> Extensor:
    """`b.exp() == b_plus.exp() * b_minus.exp()` (eq. 37), with
    `b_plus, b_minus = b.decompose_invariant()` and each factor by Euler's formula."""
    xp = b.context.xp
    dot, wedge, norm = _square(b)
    b_plus, b_minus = b.decompose_invariant()
    plus, minus = (dot + norm) / 2, (dot - norm) / 2
    rotor = (b_plus * sine(xp, plus) + cosine(xp, plus)) * (b_minus * sine(xp, minus) + cosine(xp, minus))
    # Where norm == 0 the parts are undefined, but both square to dot / 2 and multiply to wedge / 2.
    c, s = cosine(xp, dot / 2), sine(xp, dot / 2)
    equal = b * (c * s) + wedge * (s * s / 2) + c * c
    return _where(norm == 0, equal, rotor).with_traits(ReverseProductOne, Versor)


def _angle_square(xp, part_square, scalar):
    """The square of a simple part of the logarithm, from the square `part_square` of the matching
    part of the rotor's bivector part and the rotor's scalar part `scalar`: a rotation angle by
    `xp.arctan2`, a boost rapidity by `xp.arctanh`, zero for a null part."""
    rotation = xp.arctan2(xp.sqrt(xp.clip(-part_square, 0, None)), scalar)
    boost = xp.arctanh(xp.sqrt(xp.clip(part_square, 0, None)) / scalar)
    return xp.where(part_square < 0, -rotation * rotation, boost * boost)


def log_simple(r: Extensor) -> Extensor:
    """`r.log()` for rotors of simple bivectors: `r.restrict[2] * angle(xp, c)`, with `c` the
    scalar part of `r`, is the bivector part times the angle whose cosine is `c`, over its sine."""
    xp = r.context.xp
    return r.restrict_subspace(r.algebra.subspace.bivector()) * angle(xp, _scalar(r))


def log_null_wedge(r: Extensor) -> Extensor:
    """`r.log()` where the 4-vectors square to zero, as in PGA. With `bivector = r.restrict[2]`,
    `quadvector = r.restrict[4]` and `s = sine(xp, a)`, the scalar part of `bivector.squared()` is
    `s**2 * a` and the scalar part of `r` is `cosine(xp, a)`, which together give `a`. The
    logarithm `b` has `b ^ b == 2 * quadvector / s` and is
    `bivector * (1 / s - 2 * sine_derivative(xp, a) * quadvector / s**3)`."""
    xp, algebra = r.context.xp, r.algebra
    scalar = _scalar(r)
    bivector = r.restrict_subspace(algebra.subspace.bivector())
    quadvector = r.restrict_subspace(algebra.subspace.k_vector(4))
    a = _angle_square(xp, _scalar(bivector.squared()), scalar)
    s = sine(xp, a)
    correction = (bivector * quadvector).restrict_subspace(algebra.subspace.bivector())
    return bivector * (1 / s) + correction * (-2 * sine_derivative(xp, a) / s**3)


def log_decomposed(r: Extensor) -> Extensor:
    """`r.log()` as `(bivector * (quadvector * beta + alpha)).restrict[2]`, with
    `bivector = r.restrict[2]`, `quadvector = r.restrict[4]`, and `alpha` and `quadvector * beta`
    from eqs. 43-44."""
    xp, algebra = r.context.xp, r.algebra
    scalar = _scalar(r)
    bivector = r.restrict_subspace(algebra.subspace.bivector())
    quadvector = r.restrict_subspace(algebra.subspace.k_vector(4))
    # The parts of bivector square to (dot + n) / 2 and (dot - n) / 2, with n the Study norm of
    # bivector.squared(), whose 4-vector part bivector ^ bivector is quadvector * (2 * scalar).
    dot, _, n = _square(bivector)
    sigma_plus, sigma_minus = (dot + n) / 2, (dot - n) / 2
    # scalar is the product of the scalar parts of the two factors of r, and negating both factors
    # leaves r unchanged. Taking the plus factor with a nonnegative scalar part gives minus by
    # xp.arctan2 against scalar, as in the paper, and plus the same way against xp.abs(scalar).
    plus, minus = _angle_square(xp, sigma_plus, xp.abs(scalar)), _angle_square(xp, sigma_minus, scalar)
    c_plus, c_minus, s_plus, s_minus = cosine(xp, plus), cosine(xp, minus), sine(xp, plus), sine(xp, minus)
    # alpha and beta are eqs. 43-44 with n as their norm; the two parts of the logarithm multiply
    # to quadvector / (s_plus * s_minus).
    safe = xp.where(n == 0, 1, n)
    alpha = (plus * s_plus * c_minus - minus * s_minus * c_plus) / safe
    beta = (s_plus * c_minus - s_minus * c_plus) / (safe * s_plus * s_minus)
    logarithm = (bivector * (quadvector * beta + alpha)).restrict_subspace(algebra.subspace.bivector())
    # Where n == 0, r is 1 + bivector up to a null part, and its logarithm is bivector.
    return _where(n == 0, bivector, logarithm)


def _null_wedge(t) -> bool:
    """The 4-vector part of the square of type t squares to zero."""
    algebra = t.algebra
    wedge = t.squared.output_subspace.intersection(algebra.subspace.k_vector(4))
    return algebra.gatype(wedge).squared.is_empty


def _decomposable(algebra) -> bool:
    """Four or five dimensions, without the two positive and two negative directions that let the
    parts of a bivector be complex."""
    signature = algebra.signature
    return algebra.dimension in (4, 5) and min(signature.count(1), signature.count(-1)) < 2


def register() -> None:
    """Put exp and log below six dimensions ahead of the generic methods they replace."""
    from numga.extensions.logexp import bivector_exp, unit_versor_log

    # exp tries its predicates in order: special cases (empty, scalar, nilpotent), then the
    # generic bivector_exp. Inserting at bivector_exp's position puts these between, most specific
    # first: a scalar square, then a null 4-vector part, then the general decomposition.
    before_exp = Extensor.exp.position_of(bivector_exp)
    for predicate, implementation in (
        (lambda t: t <= t.algebra.subspace.bivector() and _decomposable(t.algebra), exp_decomposed),
        (lambda t: t <= t.algebra.subspace.bivector() and _decomposable(t.algebra) and _null_wedge(t), exp_null_wedge),
        (lambda t: t <= t.algebra.subspace.bivector() and t.squared.is_scalar, exp_simple),
    ):
        Extensor.exp.register(predicate, position=before_exp)(implementation)
    before_log = Extensor.log.position_of(unit_versor_log)
    for predicate, implementation in (
        (lambda t: t <= t.algebra.gatype.rotor() and _decomposable(t.algebra), log_decomposed),
        (lambda t: t <= t.algebra.gatype.rotor() and _decomposable(t.algebra)
         and _null_wedge(t.algebra.gatype.bivector()), log_null_wedge),
        (lambda t: t <= t.algebra.gatype.rotor() and t.algebra.dimension < 4, log_simple),
    ):
        Extensor.log.register(predicate, position=before_log)(implementation)
