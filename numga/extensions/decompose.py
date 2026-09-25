"""Invariant bivector decompositions and motor factorization."""

from numga.extensor import Extensor
from numga.gatype import ReverseProductOne, Versor


@Extensor.decompose_polar.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.symmetric_reverse.is_study
)
def decompose_polar(b: Extensor) -> tuple[Extensor, Extensor]:
    """`b == direction * scale`, with `scale` the Study square root of `b * ~b`. A bivector with
    no scale, zero or null, has no direction: it comes back as zero rather than as a division by
    zero."""
    scale = b.symmetric_reverse_product().square_root()
    xp = b.context.xp
    inverse = scale.inverse().map_kernel(lambda kernel: xp.nan_to_num(kernel, nan=0, posinf=0, neginf=0))
    return b.bivector_product(inverse), scale


@Extensor.decompose_invariant.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.squared.is_scalar
)
def decompose_simple(b: Extensor) -> tuple[Extensor, Extensor]:
    return b, b.context.multivector.empty().broadcast_to(b.shape)


@Extensor.decompose_invariant.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.squared.is_study
)
def decompose_bisimple(b: Extensor) -> tuple[Extensor, Extensor]:
    """The commuting simple parts `b_plus, b_minus` of `b`, with
    `b_plus = (1 + b.squared().scalar_negation() / b.squared().study_norm()) * b / 2` and `b_minus`
    with the opposite sign (Roelfs and De Keninck, eqs. 33-35): `b_plus` has the larger square,
    `(dot + norm) / 2`, with `dot` the scalar part of `b.squared()` and `norm` its Study norm."""
    squared = b.squared()
    split = -squared.scalar_negation() / (squared.study_norm() * 2)
    return ((1 - split * 2).bivector_product(b) / 2,
            (1 + split * 2).bivector_product(b) / 2)


@Extensor.motor_rotor.register(lambda t: t <= t.algebra.gatype.rotor())
def motor_rotor(motor: Extensor) -> Extensor:
    """Rotation fixing the canonical origin of a degenerate algebra: the motor's blades free of the
    null generator."""
    return motor.select_subspace(motor.subspace.nondegenerate()).with_traits(ReverseProductOne, Versor)


@Extensor.motor_translator.register(lambda t: t <= t.algebra.gatype.rotor())
def motor_translator(motor: Extensor) -> Extensor:
    """The translation left after the rotation: `motor * ~motor.motor_rotor()`, which lives on the
    scalar and the bivectors containing the null generator."""
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
    """Move the supplied origin along its bisector, then rotate around it."""
    displacement = ((motor >> origin) / origin).with_traits(ReverseProductOne, Versor)
    translation = displacement.square_root()
    return translation, ~translation * motor
