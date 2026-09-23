"""Invariant bivector decompositions and motor factorization."""

from numga.extensor import Extensor
from numga.gatype import ReverseProductOne, Versor


@Extensor.decompose_polar.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.symmetric_reverse.is_study
)
def decompose_polar(b: Extensor) -> tuple[Extensor, Extensor]:
    scale = b.symmetric_reverse_product().square_root()
    return b.bivector_product(scale.inverse()), scale


@Extensor.decompose_invariant.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.squared.is_scalar
)
def decompose_simple(b: Extensor) -> tuple[Extensor, Extensor]:
    return b, b.context.multivector.empty().broadcast_to(b.shape)


@Extensor.decompose_invariant.register(
    lambda t: t <= t.algebra.gatype.bivector() and t.squared.is_study
)
def decompose_bisimple(b: Extensor) -> tuple[Extensor, Extensor]:
    squared = b.squared()
    split = -squared.scalar_negation() / (squared.study_norm() * 2)
    return ((1 + split * 2).bivector_product(b) / 2,
            (1 - split * 2).bivector_product(b) / 2)


@Extensor.motor_rotor.register(lambda t: t <= t.algebra.gatype.rotor())
def motor_rotor(motor: Extensor) -> Extensor:
    """Rotation fixing the canonical origin of a degenerate algebra."""
    bulk = motor.subspace.restrict(
        mask for mask in motor.subspace.masks if not mask & motor.algebra.degenerate_mask
    )
    return motor.select_subspace(bulk).with_traits(ReverseProductOne, Versor)


@Extensor.motor_translator.register(lambda t: t <= t.algebra.gatype.rotor())
def motor_translator(motor: Extensor) -> Extensor:
    return motor * ~motor.motor_rotor()


@Extensor.motor_split.register(
    lambda m, o: m <= m.algebra.gatype.rotor()
    and o <= o.algebra.gatype.antivector()
    and m.algebra.signature.count(0) == 1
    and o.output_subspace.masks == ((m.algebra.blade_count - 1) ^ m.algebra.degenerate_mask,)
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
