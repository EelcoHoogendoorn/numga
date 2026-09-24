"""Opt-in closed-form exp and log below six dimensions by invariant decomposition; call register().

After Roelfs and De Keninck, "Normalization, Square Roots, and the Exponential and Logarithmic
Maps in Geometric Algebras of Less than 6D" (arXiv:2206.07496), equation numbers theirs.

Up to three dimensions a bivector squares to a scalar, and exp and log are Euler's formula and
its inverse. In four and five dimensions a bivector B squares to a Study number
B**2 = B.B + B^B, a scalar plus a 4-vector whose square is a scalar, with norm ||B**2|| = sqrt((B.B)**2 - (B^B)**2) (eq. 18). It
splits into two commuting simple bivectors b+- = P+-(B) B with the projectors
P+- = (1 +- (B.B - B^B) / ||B**2||) / 2 (eqs. 33-35), of squares
lambda+- = (B.B +- ||B**2||) / 2, and exp(B) = exp(b+) exp(b-) (eq. 37), each factor Euler's
formula c(b) + s(b).

The logarithm works backwards (section 7): the bivector part <R>_2 splits the same way into
parts of norms sigma+-, from which the angles theta+- follow by atan2 against <R> (atanh for a
boost), and B = (alpha + beta I) <R>_2 with alpha and beta I from eqs. 43-44. Of the two
factorizations R = R+ R- = (-R+)(-R-), the one with c(b+) >= 0 is taken, so that both angles
follow from <R> in any algebra.

Static structure selects simpler paths: up to three dimensions Euler's formula; where the
4-vectors square to zero, as in PGA, one part is always null and the decomposition collapses to a
closed form without branches. The decomposition has complex parts only in R(2,2) and the algebras
containing it, which the paper leaves open; those keep the generic methods.
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
_H_SERIES = (1, -1 / 3, 2 / 15, -2 / 35, 8 / 315)          # in x = p - 1


def cosine(xp, z):
    """C(z) = cosh(sqrt z), which is cos(sqrt(-z)) for z < 0."""
    root = xp.sqrt(xp.abs(z))
    return xp.where(xp.abs(z) < _SMALL, _polynomial(z, _C_SERIES), xp.where(z < 0, xp.cos(root), xp.cosh(root)))


def sine(xp, z):
    """S(z) = sinh(sqrt z) / sqrt z, which is sin(sqrt(-z)) / sqrt(-z) for z < 0."""
    small = xp.abs(z) < _SMALL
    root = xp.sqrt(xp.abs(z))
    return xp.where(small, _polynomial(z, _S_SERIES),
                    xp.where(z < 0, xp.sin(root), xp.sinh(root)) / xp.where(small, 1, root))


def angle(xp, p):
    """H(p) = acosh(p) / sqrt(p**2 - 1), which is acos(p) / sqrt(1 - p**2) for p < 1."""
    small = xp.abs(p - 1) < _SMALL
    root = xp.where(small, 1, xp.sqrt(xp.abs(1 - p * p)))
    # acosh(p) = log(p + sqrt(p**2 - 1)) for p >= 1
    return xp.where(small, _polynomial(p - 1, _H_SERIES),
                    xp.where(p < 1, xp.arctan2(root, p), xp.log(xp.abs(p) + root)) / root)


def sine_derivative(xp, z):
    """S'(z) = (C(z) - S(z)) / (2 z), which tends to 1/6 at z = 0."""
    small = xp.abs(z) < _SMALL
    return xp.where(small, _polynomial(z, (1 / 6, 1 / 60, 1 / 1680, 1 / 90720)),
                    (cosine(xp, z) - sine(xp, z)) / (2 * xp.where(small, 1, z)))


# --- the decomposition ------------------------------------------------------------------------

def _scalar(value: Extensor):
    return value.select_subspace(value.algebra.subspace.scalar())._kernel[..., 0]


def _square(b: Extensor):
    """B.B, the 4-vector B^B, and ||B**2|| = sqrt((B.B)**2 - (B^B)**2) (eq. 18); the parts of B
    square to (B.B +- ||B**2||) / 2."""
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
    """exp(B) = c(B) + s(B), Euler's formula, for B squaring to a scalar."""
    xp = b.context.xp
    square = _scalar(b.squared())
    return (b * sine(xp, square) + cosine(xp, square)).with_traits(ReverseProductOne, Versor)


def exp_null_wedge(b: Extensor) -> Extensor:
    """exp(B) where B^B squares to zero, as in PGA: one part is null and the other squares to
    a = B.B, and the product of their exponentials is C(a) + S(a) B + S'(a) B (B^B) + S(a) B^B / 2."""
    xp, algebra = b.context.xp, b.algebra
    square = b.squared()
    a = _scalar(square)
    wedge = square.restrict_subspace(algebra.subspace.k_vector(4))
    return (
        b * sine(xp, a) + (b * wedge).restrict_subspace(algebra.subspace.bivector()) * sine_derivative(xp, a)
        + wedge * (sine(xp, a) / 2) + cosine(xp, a)
    ).with_traits(ReverseProductOne, Versor)


def exp_decomposed(b: Extensor) -> Extensor:
    """exp(B) = [c(b+) + s(b+)] [c(b-) + s(b-)] (eq. 37), with b+- from decompose_invariant."""
    xp = b.context.xp
    dot, wedge, norm = _square(b)
    b_plus, b_minus = b.decompose_invariant()
    plus, minus = (dot + norm) / 2, (dot - norm) / 2
    rotor = (b_plus * sine(xp, plus) + cosine(xp, plus)) * (b_minus * sine(xp, minus) + cosine(xp, minus))
    # ||B**2|| = 0 leaves the parts undefined but squaring alike: lambda = B.B / 2, b+ b- = B^B / 2.
    c, s = cosine(xp, dot / 2), sine(xp, dot / 2)
    equal = b * (c * s) + wedge * (s * s / 2) + c * c
    return _where(norm == 0, equal, rotor).with_traits(ReverseProductOne, Versor)


def _angle_square(xp, part_square, scalar):
    """The square lambda = b**2 of a simple part of B, from the square of the matching part of
    <R>_2 and <R>: a rotation angle by atan2, a boost rapidity by atanh, zero for a null part."""
    rotation = xp.arctan2(xp.sqrt(xp.clip(-part_square, 0, None)), scalar)
    boost = xp.arctanh(xp.sqrt(xp.clip(part_square, 0, None)) / scalar)
    return xp.where(part_square < 0, -rotation * rotation, boost * boost)


def log_simple(r: Extensor) -> Extensor:
    """log(R) = <R>_2 theta / sin(theta), with cos(theta) = <R>, for rotors of simple bivectors."""
    xp = r.context.xp
    return r.restrict_subspace(r.algebra.subspace.bivector()) * angle(xp, _scalar(r))


def log_null_wedge(r: Extensor) -> Extensor:
    """log(R) where the 4-vectors square to zero, as in PGA: <<R>_2**2> = S(a)**2 a gives a against
    <R> = C(a), B^B = 2 <R>_4 / S(a), and B = <R>_2 (1 / S(a) - 2 S'(a) <R>_4 / S(a)**3)."""
    xp, algebra = r.context.xp, r.algebra
    scalar = _scalar(r)
    bivector = r.restrict_subspace(algebra.subspace.bivector())
    quadvector = r.restrict_subspace(algebra.subspace.k_vector(4))
    a = _angle_square(xp, _scalar(bivector.squared()), scalar)
    s = sine(xp, a)
    correction = (bivector * quadvector).restrict_subspace(algebra.subspace.bivector())
    return bivector * (1 / s) + correction * (-2 * sine_derivative(xp, a) / s**3)


def log_decomposed(r: Extensor) -> Extensor:
    """log(R) = (alpha + beta I) <R>_2, with alpha and beta I from eqs. 43-44."""
    xp, algebra = r.context.xp, r.algebra
    scalar = _scalar(r)
    bivector = r.restrict_subspace(algebra.subspace.bivector())
    quadvector = r.restrict_subspace(algebra.subspace.k_vector(4))
    # The parts of <R>_2 square to (<R>_2 . <R>_2 +- n) / 2, with
    # n = ||<R>_2**2|| = sqrt((<R>_2 . <R>_2)**2 - 4 <R>**2 <R>_4**2), as <R>_2 ^ <R>_2 = 2 <R> <R>_4.
    dot, _, n = _square(bivector)
    sigma_plus, sigma_minus = (dot + n) / 2, (dot - n) / 2
    # <R> = c(b+) c(b-), and R = R+ R- = (-R+)(-R-): taking c(b+) >= 0 makes theta- the paper's
    # atan2(sigma-, <R>), and theta+ the same against |<R>|.
    plus, minus = _angle_square(xp, sigma_plus, xp.abs(scalar)), _angle_square(xp, sigma_minus, scalar)
    c_plus, c_minus, s_plus, s_minus = cosine(xp, plus), cosine(xp, minus), sine(xp, plus), sine(xp, minus)
    # n is ||s**2(B)|| of eqs. 43-44, and b+ b- = <R>_4 / (s+ s-).
    safe = xp.where(n == 0, 1, n)
    alpha = (plus * s_plus * c_minus - minus * s_minus * c_plus) / safe
    beta = (s_plus * c_minus - s_minus * c_plus) / (safe * s_plus * s_minus)
    logarithm = (bivector * (quadvector * beta + alpha)).restrict_subspace(algebra.subspace.bivector())
    # n = 0: R = 1 + <R>_2 up to a null part, whose logarithm is its bivector part.
    return _where(n == 0, bivector, logarithm)


def _null_wedge(t) -> bool:
    """The 4-vector part of the square of type t squares to zero."""
    algebra = t.algebra
    wedge = t.squared.output_subspace.intersection(algebra.subspace.k_vector(4))
    return algebra.gatype(wedge).squared.is_empty


def _decomposable(algebra) -> bool:
    """Four or five dimensions, without the two directions of each sign that let the parts of a
    bivector be complex, as in R(2,2)."""
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
