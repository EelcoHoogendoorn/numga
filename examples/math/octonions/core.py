"""Octonion multiplication and the rotations that preserve it.

The scalar and seven vectors of Cl(0, 7) carry the eight octonion components.
A fixed trivector supplies their multiplication; the bivectors that commute with
that trivector generate its automorphisms, the exceptional group G₂.
"""

from numga import Algebra, NumpyContext, stack

ga = Algebra("a-b-c-d-e-f-g-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Octonion = ga.gatype.from_blades("1 a b c d e f g")
Bivector = ga.gatype.bivector()
Trivector = ga.gatype.trivector()
Rotor = ga.gatype.rotor()

# Each oriented triple supplies a quaternion multiplication cycle.
cycles = stack([mv.abc, mv.ade, mv.agf, mv.bdf, mv.beg, mv.cdg, mv.cfe])
orientation = cycles.sum(axis=0)
# Projection is part of this product: discard the Clifford product's other grades.
multiply = ((Octonion * Octonion) * (1 - orientation)).cast(Octonion)


# --- math -----------------------------------------------------------------------------
def norm_squared(value: Octonion) -> Scalar:
    """Positive squared length, pairing the value with its grade involution."""
    return value.involute().scalar_product(value)


def associator(first: Octonion, second: Octonion, third: Octonion) -> Octonion:
    """The difference between the two bracketings of three factors."""
    return multiply(multiply(first, second), third) - multiply(first, multiply(second, third))


def symmetry_generators(tolerance: float) -> Bivector:
    """Infinitesimal rotations preserving the multiplication, as a batch of bivectors."""
    # Clifford multiplication and grade projection already commute with rotations.
    # Only the fixed trivector can change, so its stationary directions are the symmetries.
    constraint = Bivector.commutator(orientation)                              # [] Trivector <- Bivector
    _, singular_values, directions = constraint.svd()
    return directions[singular_values < tolerance]                            # [generators] Bivector


def derivation_defect(generator: Bivector, first: Octonion, second: Octonion) -> Octonion:
    """Failure of an infinitesimal rotation to obey the octonion product rule."""
    return (generator.commutator(multiply(first, second))
            - multiply(generator.commutator(first), second)
            - multiply(first, generator.commutator(second)))


def multiplication_defect(rotor: Rotor, first: Octonion, second: Octonion) -> Octonion:
    """Rotate a product, or multiply the rotated factors: their difference."""
    return (rotor >> multiply(first, second)) - multiply(rotor >> first, rotor >> second)
