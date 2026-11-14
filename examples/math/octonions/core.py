"""Octonions: the eight-component numbers whose product keeps lengths but not brackets, and the rotations
that preserve it.

The real numbers, the complex numbers, the quaternions and the octonions are the only number systems in which
the length of a product is the product of the lengths. The octonions pay for their eight components with
associativity: in a product of three, the brackets matter. The rotations of their seven units that square to
minus one, which keep the product intact, form the exceptional group G2.

An octonion is modelled as the scalar plus a vector of the seven-generator negative-definite Clifford algebra,
and the seven multiplication triples as one trivector, `orientation`. Their product is a single map from pairs
of octonions to octonions, `((Octonion * Octonion) * (1 - orientation)).cast(Octonion)`, the geometric product
times `1 - orientation` projected back onto scalars and vectors. The squared length is the product with the
conjugate, `value.symmetric_conjugate_product()`, and the associator is a map on triples of octonions. The
symmetries are the bivectors in the null space of `Bivector.commutator(orientation)`. Every multivector of
three-dimensional space also holds an octonion: the even parts multiply by the geometric product, while the odd
parts enter reversed and in swapped order, and with one basis vector squaring to minus one the same rule gives
the split octonions.
"""

from numga import Algebra, NumpyContext, stack
from numga.algebras import VGA3D as space

ga = Algebra("a-b-c-d-e-f-g-")
mv = NumpyContext(ga).multivector
Octonion = ga.gatype(ga.gatype.scalar() + ga.gatype.vector())               # Scalar+ + Vector+
Bivector = ga.gatype.bivector()                                             # Bivector+
Trivector = ga.gatype.trivector()                                           # Trivector+
# An octonion as a whole multivector of space, Cl(3, 0).
SpaceOctonion = space.gatype.full()                                         # Full+

# Each oriented triple closes under the product, `a` times `b` giving `c`; with the scalar, it multiplies as
# the even part of space does.
cycles = stack([mv.abc, mv.ade, mv.agf, mv.bdf, mv.beg, mv.cdg, mv.cfe])    # [7] Trivector+
orientation = cycles.sum(axis=0)                                            # [] Trivector+
# Projection is part of this product: discard the Clifford product's other grades.
product = ((Octonion * Octonion) * (1 - orientation)).cast(Octonion)        # [] Octonion+ <- (Octonion+, Octonion+)
# The difference between the two bracketings of three factors.
associator = (product(product(Octonion, Octonion), Octonion)
              - product(Octonion, product(Octonion, Octonion)))             # [] Octonion+ <- (Octonion+, Octonion+, Octonion+)


# --- math -----------------------------------------------------------------------------
def symmetry_generators(tolerance: float) -> Bivector:
    """The infinitesimal rotations preserving the octonion product form a batch of bivectors (`[...] Bivector+`)."""
    # Clifford multiplication and grade projection already commute with rotations.
    # Only the fixed trivector can change, so the bivectors that leave it unchanged are the symmetries.
    constraint = Bivector.commutator(orientation)                           # [] Trivector+ <- Bivector+
    _, singular_values, singular_bivectors = constraint.svd()
    return singular_bivectors[singular_values < tolerance]                  # [...] Bivector+


def space_product(first: SpaceOctonion, second: SpaceOctonion) -> SpaceOctonion:
    """Two multivectors of space multiply as octonions (`[...] SpaceOctonion+-`) by doubling the even subalgebra,
    multiplying even parts directly and bringing in odd parts reversed and in swapped order."""
    # The even part of each factor, a scalar and a bivector, scales and turns; its odd part, a vector and a
    # trivector, scales, turns and reflects, and is the second half of the pair.
    first_even, first_odd = first.restrict.even(), first.restrict.odd()
    second_even, second_odd = second.restrict.even(), second.restrict.odd()
    return (first_even * second_even - second_odd.reverse() * first_odd
            + second_odd * first_even + first_odd * second_even.reverse())
