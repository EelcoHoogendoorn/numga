"""Bivector invariant decompositions and rigid-motion motor factorizations exploit Study-number
arithmetic below six dimensions.

When a bivector's symmetric reverse product or square is a Study number (a scalar plus a grade-four
part squaring to a scalar), closed-form Study square roots and norms separate the bivector without
iterative eigensolvers. `decompose_polar` factors a bivector into a unit screw `direction` and a
commuting Study `scale` (`bivector == direction * scale`), while `decompose_invariant` splits a
bivector into two commuting simple blades `(b_plus, b_minus)` whose squares are scalars (Roelfs and
De Keninck, arXiv:2206.07496, eqs. 33-35).

Motor factorizations separate rotations from translations relative to an origin point. In a
degenerate algebra with a canonical origin, `motor_rotor` selects the nondegenerate blades fixing
that origin and `motor_translator` extracts the remaining translation `motor * ~motor.motor_rotor()`.
For an arbitrary `origin` antivector, `motor_split` computes the half-displacement translator carrying
`origin` to `motor >> origin` and factors `motor == translation * rotation` around that point.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import ReverseProductOne, Versor


@Extensor.decompose_polar.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.derive.symmetric_reverse.is_study
)
def decompose_polar(bivector: Extensor) -> tuple[Extensor, Extensor]:
    """Polar bivector factorization returns `(direction, scale)` with `bivector == direction * scale`
    and `scale == (bivector * ~bivector).square_root()`; zero and null bivectors return a zero
    `direction` rather than dividing by zero."""
    scale = bivector.symmetric_reverse_product().square_root()
    xp = bivector.context.xp
    inverse = scale.inverse().map_kernel(lambda kernel: xp.nan_to_num(kernel, nan=0, posinf=0, neginf=0))
    return bivector.bivector_product(inverse), scale


@Extensor.decompose_invariant.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.derive.squared.is_scalar
)
def decompose_simple(bivector: Extensor) -> tuple[Extensor, Extensor]:
    return bivector, bivector.context.multivector.empty().broadcast_to(bivector.shape)


@Extensor.decompose_invariant.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.derive.squared.is_study
)
def decompose_bisimple(bivector: Extensor) -> tuple[Extensor, Extensor]:
    """The invariant decomposition splits `bivector` into commuting simple bivectors `(b_plus, b_minus)`
    ordered so `b_plus` carries the larger scalar square `(dot + norm) / 2`, where `dot` is the
    scalar part of `bivector.squared()` and `norm` is its Study norm."""
    squared = bivector.squared()
    split = -squared.scalar_negation() / (squared.study_norm() * 2)
    return ((1 - split * 2).bivector_product(bivector) / 2,
            (1 + split * 2).bivector_product(bivector) / 2)


@Extensor.motor_rotor.register(lambda t: t <= t.algebra.gatype.rotor())
def motor_rotor(motor: Extensor) -> Extensor:
    """Selecting the nondegenerate blades of `motor` isolates the rotation fixing the algebra's
    canonical origin."""
    return motor.select_subspace(motor.subspace.nondegenerate()).with_traits(ReverseProductOne, Versor)


@Extensor.motor_translator.register(lambda t: t <= t.algebra.gatype.rotor())
def motor_translator(motor: Extensor) -> Extensor:
    """Factoring out `motor.motor_rotor()` leaves the pure translation `motor * ~motor.motor_rotor()`
    supported on the scalar and null-generator bivectors."""
    translation = motor * ~motor.motor_rotor()
    return translation.restrict_subspace(motor.algebra.subspace.translator()).with_traits(ReverseProductOne, Versor)


@Extensor.motor_split.register(
    lambda m, o: m <= m.algebra.gatype.rotor()
    and o <= o.algebra.gatype.antivector()
    and m.algebra.signature.count(0) == 1
    and o.output_subspace.same_support(o.algebra.subspace.antivector().nondegenerate())
)
def split_canonical_euclidean(motor: Extensor, origin: Extensor) -> tuple[Extensor, Extensor]:
    return motor.motor_translator(), motor.motor_rotor()


@Extensor.motor_split.register(
    lambda m, o: m <= m.algebra.gatype.rotor() and o <= o.algebra.gatype.antivector()
)
def motor_split(motor: Extensor, origin: Extensor) -> tuple[Extensor, Extensor]:
    """Factoring `motor` around `origin` returns `(translation, rotation)` with
    `motor == translation * rotation`, where `rotation` fixes `origin` and `translation` carries
    `origin` to `motor >> origin`."""
    displacement = ((motor >> origin) / origin).with_traits(ReverseProductOne, Versor)
    translation = displacement.square_root()
    return translation, ~translation * motor
