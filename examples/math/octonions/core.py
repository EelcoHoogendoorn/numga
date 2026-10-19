"""Octonions: the eight-component numbers whose product keeps lengths but not brackets, and the rotations
that preserve it.

The real numbers, the complex numbers, the quaternions and the octonions are the only number systems in which
the length of a product is the product of the lengths. The octonions pay for their eight components with
associativity: in a product of three, the brackets matter. The rotations of their seven imaginary directions
that keep the product intact form the exceptional group G2.

An octonion is modelled as the scalar plus a vector of Cl(0, 7), and the seven multiplication triples as one
trivector, `orientation`; the product is a single map from pairs of octonions to octonions, the geometric
product times `1 - orientation`, keeping the scalar and vector parts. The symmetries are the bivectors whose
commutator with that trivector vanishes. The same octonions are every multivector of space, Cl(3, 0): the even
part a quaternion and the odd part the pseudoscalar times a second, multiplied by doubling the quaternions;
with one direction squaring to minus one they are the split octonions.

In index notation the imaginary units multiply as e_i e_j = -delta_ij + epsilon_ijk e_k, with epsilon totally
antisymmetric and one on each of the seven triples, and G2 is the group of rotations that keep epsilon.
"""

from numga import Algebra, NumpyContext, stack
from numga.algebras import VGA3D as space

ga = Algebra("a-b-c-d-e-f-g-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Octonion = ga.gatype.from_blades("1 a b c d e f g")
Bivector = ga.gatype.bivector()
Trivector = ga.gatype.trivector()
Rotor = ga.gatype.rotor()
# An octonion as a whole multivector of space, Cl(3, 0).
SpaceOctonion = space.gatype.from_blades("1 x y z yz zx xy xyz")

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


def space_multiply(first: SpaceOctonion, second: SpaceOctonion) -> SpaceOctonion:
    """The octonion product of two multivectors of space, by doubling the quaternions: the even parts multiply
    as quaternions, and the odd parts, a second quaternion apiece, enter reversed and in swapped order."""
    # The even parts are quaternions; the odd parts, the pseudoscalar times a second quaternion apiece.
    first_even, first_odd = first.restrict.even(), first.restrict.odd()
    second_even, second_odd = second.restrict.even(), second.restrict.odd()
    return (first_even * second_even - second_odd.reverse() * first_odd
            + second_odd * first_even + first_odd * second_even.reverse())


def multiplication_defect(rotor: Rotor, first: Octonion, second: Octonion) -> Octonion:
    """Rotate a product, or multiply the rotated factors: their difference."""
    return (rotor >> multiply(first, second)) - multiply(rotor >> first, rotor >> second)
