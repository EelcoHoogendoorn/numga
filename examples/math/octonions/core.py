"""Octonions: the eight-component numbers whose product keeps lengths but not brackets, and the rotations
that preserve it.

The real numbers, the complex numbers, the quaternions and the octonions are the only number systems in which
the length of a product is the product of the lengths. The octonions pay for their eight components with
associativity: in a product of three, the brackets matter. The rotations of their seven units that square to
minus one, which keep the product intact, form the exceptional group G2.

An octonion is modelled as the scalar plus a vector of Cl(0, 7), and the seven multiplication triples as one
trivector, `orientation`; the product is a single map from pairs of octonions to octonions, the geometric
product times `1 - orientation`, keeping the scalar and vector parts. The squared length is the product with
the conjugate, `value.symmetric_conjugate_product()`, and the associator a map on triples of octonions. The
symmetries are the bivectors whose commutator with that trivector vanishes. The same octonions are every
multivector of space, Cl(3, 0): the even parts, a scalar and a bivector that scale and turn, multiply by the
geometric product, and the odd parts, a vector and a trivector, enter reversed and in swapped order; with one
basis vector squaring to minus one they are the split octonions.

In index notation the imaginary units multiply as e_i e_j = -delta_ij + epsilon_ijk e_k, with epsilon totally
antisymmetric and one on each of the seven triples, and G2 is the group of rotations that keep epsilon. In the
notation of the Cayley-Dickson construction an octonion is a pair of quaternions, multiplied as
(p, q)(r, s) = (p r - conj(s) q, s p + q conj(r)); the first of each pair is the even part of space here, and
the second the odd part.
"""

from numga import Algebra, NumpyContext, stack
from numga.algebras import VGA3D as space

ga = Algebra("a-b-c-d-e-f-g-")
mv = NumpyContext(ga).multivector
Octonion = ga.gatype.from_blades("1 a b c d e f g")                 # Scalar+ + Vector+
Bivector = ga.gatype.bivector()                                     # Bivector+
Trivector = ga.gatype.trivector()                                   # Trivector+
# An octonion as a whole multivector of space, Cl(3, 0).
SpaceOctonion = space.gatype.from_blades("1 x y z yz zx xy xyz")    # Full+

# Each oriented triple closes under the product, `a` times `b` giving `c`; with the scalar, it multiplies as
# the even part of space does.
cycles = stack([mv.abc, mv.ade, mv.agf, mv.bdf, mv.beg, mv.cdg, mv.cfe])   # [triples] Trivector+
orientation = cycles.sum(axis=0)                                           # [] Trivector+
# Projection is part of this product: discard the Clifford product's other grades.
multiply = ((Octonion * Octonion) * (1 - orientation)).cast(Octonion)      # [] Octonion+ <- (Octonion+, Octonion+)
# The difference between the two bracketings of three factors.
associator = (multiply(multiply(Octonion, Octonion), Octonion)
              - multiply(Octonion, multiply(Octonion, Octonion)))  # [] Octonion+ <- (Octonion+, Octonion+, Octonion+)


# --- math -----------------------------------------------------------------------------
def symmetry_generators(tolerance: float) -> Bivector:
    """Infinitesimal rotations preserving the multiplication, as a batch of bivectors."""
    # Clifford multiplication and grade projection already commute with rotations.
    # Only the fixed trivector can change, so the bivectors that leave it unchanged are the symmetries.
    constraint = Bivector.commutator(orientation)                              # [] Trivector+ <- Bivector+
    _, singular_values, singular_bivectors = constraint.svd()
    return singular_bivectors[singular_values < tolerance]                     # [generators] Bivector+


def space_multiply(first: SpaceOctonion, second: SpaceOctonion) -> SpaceOctonion:
    """The octonion product of two multivectors of space, doubling the even part: the even parts multiply by
    the geometric product, and the odd parts enter reversed and in swapped order."""
    # The even part of each factor, a scalar and a bivector, scales and turns; its odd part, a vector and a
    # trivector, scales, turns and reflects, and is the second half of the pair.
    first_even, first_odd = first.restrict.even(), first.restrict.odd()
    second_even, second_odd = second.restrict.even(), second.restrict.odd()
    return (first_even * second_even - second_odd.reverse() * first_odd
            + second_odd * first_even + first_odd * second_even.reverse())

