"""Opt-in closed-form exponentials and logarithms below six dimensions decompose bivectors and
rotors into commuting simple blades, following Roelfs and De Keninck, "Normalization, Square Roots,
and the Exponential and Logarithmic Maps in Geometric Algebras of Less than 6D" (arXiv:2206.07496);
equation numbers below refer to that paper. Calling `register()` inserts these closed forms ahead of
the generic scaling-and-squaring implementations in `numga.extensions.logexp`.

Up to three dimensions every bivector squares to a scalar, so `exp` and `log` reduce directly to
Euler's formula and its inverse. In four and five dimensions `bivector.squared()` is a Study number
(a scalar `dot` plus the grade-four wedge `bivector ^ bivector` whose square is scalar) with Study
norm `norm = bivector.squared().study_norm()` (eq. 18). `bivector.decompose_invariant()` splits
`bivector` into two commuting simple bivectors `b_plus` and `b_minus` squaring to `(dot + norm) / 2`
and `(dot - norm) / 2` (eqs. 33-35), so `bivector.exp() == b_plus.exp() * b_minus.exp()` (eq. 37)
with each factor evaluated by Euler's formula.

The logarithm inverts that product on rotors (section 7). Splitting the bivector part `rotor.restrict[2]`
recovers the two simple squares, `arctan2` and `arctanh` against the scalar part of `rotor` give the
two angles, and `rotor.log()` recombines `(rotor.restrict[2] * (rotor.restrict[4] * beta + alpha)).restrict[2]`
from eqs. 43-44. Selecting the factorization whose `plus` factor has a nonnegative scalar part fixes
the sign ambiguity of negating both simple rotor factors across every signature.

Static subspace structure selects tighter closed forms automatically: single-square blades use
Euler's formula directly, and algebras whose grade-four blades square to zero (such as 3D PGA) have
one null part so the exponential and logarithm collapse to branch-free rational expressions. Algebras
with at least two positive and two negative basis squares admit complex invariant parts outside the
paper's real formulas and retain the generic `logexp` methods.
"""

from __future__ import annotations

from typing import Any

from numga.algebra import Algebra
from numga.extensor import Extensor
from numga.gatype import ReverseProductOne, Versor


# --- scalar functions --------------------------------------------------------------------------
# Removable singularities at zero or one switch to fourth-order Taylor polynomials while keeping
# denominators away from zero in inactive `where` branches.

_SMALL = 1e-3


def _polynomial(argument: Any, coefficients: tuple[float, ...]) -> Any:
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * argument + coefficient
    return result


_C_SERIES = (1.0, 1 / 2, 1 / 24, 1 / 720, 1 / 40320)
_S_SERIES = (1.0, 1 / 6, 1 / 120, 1 / 5040, 1 / 362880)
_H_SERIES = (1.0, -1 / 3, 2 / 15, -2 / 35, 8 / 315)          # in powers of cos_angle - 1


def scalar_square_exp(element: Extensor) -> Extensor:
    """When `element.squared()` is a scalar `s`, `element.exp()` equals `even + element * odd` with
    hyperbolic coefficients for `s > 0`, circular coefficients for `s < 0`, and `1 + element` for
    `s == 0`, without asserting a `Versor` trait (since 4D pseudoscalar exponentials are non-versors)."""
    xp = element.context.xp
    square = element.squared().kernel[..., 0]                          # [...]
    root = xp.sqrt(xp.abs(square))
    safe = xp.where(root > 0, root, 1)
    even = xp.where(square > 0, xp.cosh(root), xp.cos(root))
    odd = xp.where(root > 0, xp.where(square > 0, xp.sinh(root), xp.sin(root)) / safe, 1)
    scalar = element.context.multivector.scalar
    return scalar(even[..., None]) + element * scalar(odd[..., None])


def cosine(xp: Any, square: Any) -> Any:
    """For `square >= 0` this series evaluates `cosh(sqrt(square))`, and for `square < 0` it
    evaluates `cos(sqrt(-square))`."""
    root = xp.sqrt(xp.abs(square))
    return xp.where(xp.abs(square) < _SMALL, _polynomial(square, _C_SERIES), xp.where(square < 0, xp.cos(root), xp.cosh(root)))


def sine(xp: Any, square: Any) -> Any:
    """For `square > 0` this series evaluates `sinh(sqrt(square)) / sqrt(square)`, approaching `1`
    at `0` and `sin(sqrt(-square)) / sqrt(-square)` for `square < 0`."""
    small = xp.abs(square) < _SMALL
    root = xp.sqrt(xp.abs(square))
    return xp.where(small, _polynomial(square, _S_SERIES),
                    xp.where(square < 0, xp.sin(root), xp.sinh(root)) / xp.where(small, 1, root))


def angle(xp: Any, cos_angle: Any) -> Any:
    """For `cos_angle > 1` this ratio evaluates `arccosh(cos_angle) / sqrt(cos_angle**2 - 1)`,
    approaching `1` at `1` and `arccos(cos_angle) / sqrt(1 - cos_angle**2)` for `cos_angle < 1`."""
    small = xp.abs(cos_angle - 1) < _SMALL
    root = xp.where(small, 1, xp.sqrt(xp.abs(1 - cos_angle * cos_angle)))
    # For cos_angle >= 1, arccosh(cos_angle) == log(cos_angle + sqrt(cos_angle**2 - 1)).
    return xp.where(small, _polynomial(cos_angle - 1, _H_SERIES),
                    xp.where(cos_angle < 1, xp.arctan2(root, cos_angle), xp.log(xp.abs(cos_angle) + root)) / root)


def sine_derivative(xp: Any, square: Any) -> Any:
    """The derivative `(cosine(square) - sine(square)) / (2 * square)` of `sine` with respect to
    `square` tends smoothly to `1 / 6` as `square` approaches zero."""
    small = xp.abs(square) < _SMALL
    return xp.where(small, _polynomial(square, (1 / 6, 1 / 60, 1 / 1680, 1 / 90720)),
                    (cosine(xp, square) - sine(xp, square)) / (2 * xp.where(small, 1, square)))


# --- the decomposition ------------------------------------------------------------------------

def _scalar(value: Extensor) -> Any:
    return value.select_subspace(value.algebra.subspace.scalar())._kernel[..., 0]


def _square(bivector: Extensor) -> tuple[Any, Extensor, Any]:
    xp = bivector.context.xp
    square = bivector.squared()
    dot = _scalar(square)
    wedge = square.restrict_subspace(bivector.algebra.subspace.k_vector(4))
    return dot, wedge, xp.sqrt(dot * dot - _scalar(wedge * wedge))


def _where(condition: Any, when_true: Extensor, when_false: Extensor) -> Extensor:
    xp, space = when_false.context.xp, when_false.output_subspace
    kernel = xp.where(condition[..., None], when_true.select_subspace(space)._kernel, when_false._kernel)
    return Extensor._from_prepared_kernel(when_false.context, when_false.gatype, kernel)


def exp_simple(bivector: Extensor) -> Extensor:
    """When `bivector.squared()` is a scalar `s`, Euler's formula `cosine(s) + bivector * sine(s)`
    gives the rotor directly."""
    xp = bivector.context.xp
    square = _scalar(bivector.squared())
    return (bivector * sine(xp, square) + cosine(xp, square)).with_traits(ReverseProductOne, Versor)


def exp_null_wedge(bivector: Extensor) -> Extensor:
    """When `wedge = bivector ^ bivector` squares to zero, one simple part is null and the other
    squares to the scalar part `s` of `bivector.squared()`, collapsing their product exponential to
    a branch-free expression in `bivector`, `wedge`, and `(bivector * wedge).restrict[2]`."""
    xp, algebra = bivector.context.xp, bivector.algebra
    square = bivector.squared()
    scalar_square = _scalar(square)
    wedge = square.restrict_subspace(algebra.subspace.k_vector(4))
    return (
        bivector * sine(xp, scalar_square)
        + (bivector * wedge).restrict_subspace(algebra.subspace.bivector()) * sine_derivative(xp, scalar_square)
        + wedge * (sine(xp, scalar_square) / 2)
        + cosine(xp, scalar_square)
    ).with_traits(ReverseProductOne, Versor)


def exp_decomposed(bivector: Extensor) -> Extensor:
    """Decomposing `bivector` into commuting simple parts `(b_plus, b_minus)` evaluates
    `bivector.exp() == b_plus.exp() * b_minus.exp()` (eq. 37), substituting the closed isoclinic
    limit when the Study norm of `bivector.squared()` vanishes."""
    xp = bivector.context.xp
    dot, wedge, norm = _square(bivector)
    b_plus, b_minus = bivector.decompose_invariant()
    plus, minus = (dot + norm) / 2, (dot - norm) / 2
    rotor = (b_plus * sine(xp, plus) + cosine(xp, plus)) * (b_minus * sine(xp, minus) + cosine(xp, minus))
    # Where norm == 0 the individual parts are undefined, but both square to dot / 2 and multiply to wedge / 2.
    c, s = cosine(xp, dot / 2), sine(xp, dot / 2)
    equal = bivector * (c * s) + wedge * (s * s / 2) + c * c
    return _where(norm == 0, equal, rotor).with_traits(ReverseProductOne, Versor)


def _angle_square(xp: Any, part_square: Any, cos_product: Any) -> Any:
    rotation = xp.arctan2(xp.sqrt(xp.clip(-part_square, 0, None)), cos_product)
    boost = xp.arctanh(xp.sqrt(xp.clip(part_square, 0, None)) / cos_product)
    return xp.where(part_square < 0, -rotation * rotation, boost * boost)


def log_simple(rotor: Extensor) -> Extensor:
    """For a rotor of a simple bivector, scaling `rotor.restrict[2]` by the inverse-sine ratio of
    its scalar part recovers the bivector generator."""
    xp = rotor.context.xp
    return rotor.restrict_subspace(rotor.algebra.subspace.bivector()) * angle(xp, _scalar(rotor))


def log_null_wedge(rotor: Extensor) -> Extensor:
    """When grade-four blades square to zero, the scalar part of `rotor` and the scalar square of
    `rotor.restrict[2]` determine the non-null square, yielding a branch-free rational expression in
    `bivector` and `(bivector * quadvector).restrict[2]`."""
    xp, algebra = rotor.context.xp, rotor.algebra
    scalar = _scalar(rotor)
    bivector = rotor.restrict_subspace(algebra.subspace.bivector())
    quadvector = rotor.restrict_subspace(algebra.subspace.k_vector(4))
    scalar_square = _angle_square(xp, _scalar(bivector.squared()), scalar)
    s = sine(xp, scalar_square)
    correction = (bivector * quadvector).restrict_subspace(algebra.subspace.bivector())
    return bivector * (1 / s) + correction * (-2 * sine_derivative(xp, scalar_square) / s**3)


def log_decomposed(rotor: Extensor) -> Extensor:
    """The invariant-decomposition logarithm reconstructs `(bivector * (quadvector * beta + alpha)).restrict[2]`
    from eqs. 43-44 on four- and five-dimensional rotors, falling back to the half-angle formula when
    the two simple parts are isoclinic."""
    xp, algebra = rotor.context.xp, rotor.algebra
    scalar = _scalar(rotor)
    bivector = rotor.restrict_subspace(algebra.subspace.bivector())
    quadvector = rotor.restrict_subspace(algebra.subspace.k_vector(4))
    # The parts of bivector square to (dot + norm) / 2 and (dot - norm) / 2, where norm is the Study
    # norm of bivector.squared() and bivector ^ bivector equals quadvector * (2 * scalar).
    dot, _, norm = _square(bivector)
    sigma_plus, sigma_minus = (dot + norm) / 2, (dot - norm) / 2
    # Because negating both simple rotor factors leaves rotor unchanged, choosing the plus factor with
    # nonnegative scalar part determines minus against scalar and plus against abs(scalar).
    plus, minus = _angle_square(xp, sigma_plus, xp.abs(scalar)), _angle_square(xp, sigma_minus, scalar)
    c_plus, c_minus, s_plus, s_minus = cosine(xp, plus), cosine(xp, minus), sine(xp, plus), sine(xp, minus)
    safe = xp.where(norm == 0, 1, norm)
    alpha = (plus * s_plus * c_minus - minus * s_minus * c_plus) / safe
    beta = (s_plus * c_minus - s_minus * c_plus) / (safe * s_plus * s_minus)
    logarithm = (bivector * (quadvector * beta + alpha)).restrict_subspace(algebra.subspace.bivector())
    # When norm == 0 both parts turn through the same angle t with scalar == cos(t)**2, so the
    # logarithm equals bivector * (2 * t / sin(2 * t)) evaluated at cos(2 * t) == 2 * scalar - 1.
    return _where(norm == 0, bivector * angle(xp, 2 * scalar - 1), logarithm)


def _decomposable(algebra: Algebra) -> bool:
    signature = algebra.signature
    return algebra.dimension in (4, 5) and min(signature.count(1), signature.count(-1)) < 2


def register() -> None:
    """Registering invariant-decomposition `exp` and `log` rules places them immediately ahead of the
    generic scaling-and-squaring methods in four and five dimensions."""
    from numga.extensions.logexp import bivector_exp, unit_versor_log

    Extensor.exp.register(lambda t: t.derive.squared.is_scalar)(scalar_square_exp)

    # Inserting at bivector_exp's index places closed forms after nilpotent/scalar shortcuts and
    # before scaling-and-squaring, ordered from most specific (scalar square) to general Study square.
    before_exp = Extensor.exp.position_of(bivector_exp)
    for predicate, implementation in (
        (lambda t: t <= t.algebra.subspace.bivector() and _decomposable(t.algebra), exp_decomposed),
        (lambda t: t <= t.algebra.subspace.bivector() and _decomposable(t.algebra) and t.has_null_wedge, exp_null_wedge),
        (lambda t: t <= t.algebra.subspace.bivector() and t.derive.squared.is_scalar, exp_simple),
    ):
        Extensor.exp.register(predicate, position=before_exp)(implementation)
    before_log = Extensor.log.position_of(unit_versor_log)
    for predicate, implementation in (
        (lambda t: t <= t.algebra.gatype.rotor() and _decomposable(t.algebra), log_decomposed),
        (lambda t: t <= t.algebra.gatype.rotor() and _decomposable(t.algebra)
         and t.algebra.gatype.bivector().has_null_wedge, log_null_wedge),
        (lambda t: t <= t.algebra.gatype.rotor() and t.algebra.dimension < 4, log_simple),
    ):
        Extensor.log.register(predicate, position=before_log)(implementation)
