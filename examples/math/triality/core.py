"""Spin(8) triality: three rotations of eight-dimensional space linked by the octonion product.

The vectors of Spin(8) and its two spinor spaces each have eight components, and the octonion product links
them: rotating a product agrees with multiplying its two factors, each turned by a companion rotation of its
own. One rotation acts on the product and two act on its factors. Requiring all three to coincide leaves the
fourteen generators of G2.

An octonion is modelled as the scalar plus a vector of Cl(0, 7), and the product as one map from pairs of
octonions to octonions, `multiply`. A generator of rotations of the eight components is a vector, turning the
scalar into the vectors, plus a bivector, turning the vectors among themselves. It splits into multiplication
by a vector on the left, multiplication by a vector on the right, and a derivation, which obeys the product
rule; minus the trace of a composition pairs skew maps, so one solve finds the split, and the two
multiplication parts, redistributed, give each generator its two companions.
"""

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a-b-c-d-e-f-g-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                 # Scalar+
Vector = ga.gatype.vector()                                                 # Vector+
Octonion = ga.gatype.from_grades([0, 1])                                    # Scalar+ + Vector+
Generator = ga.gatype.from_grades([1, 2])                                   # Vector+ + Bivector+
OctonionMap = ga.gatype((Octonion, Octonion))
OffsetMap = ga.gatype((Vector, Generator))

# Each oriented triple closes under the product, `a` times `b` giving `c`; summed, the seven triples are the
# oriented lines of the Fano plane, one trivector that carries the whole multiplication table.
cycles = stack([mv.abc, mv.ade, mv.agf, mv.bdf, mv.beg, mv.cdg, mv.cfe])   # [triples] Trivector+
fano = cycles.sum(axis=0)                                                   # [] Trivector+
multiply = ((Octonion * Octonion) * (1 - fano)).cast(Octonion)              # [] Octonion+ <- (Octonion+, Octonion+)

# Bivectors turn the seven vector components; vectors turn them into the scalar.
turn = (Generator.restrict.bivector().commutator(Octonion)
        + Generator.restrict.vector().anticommutator(Octonion))             # [] Octonion+ <- (Generator+, Octonion+)


# --- math -----------------------------------------------------------------------------
def multiplication_components() -> tuple[OffsetMap, OffsetMap]:
    """The vectors whose left and right multiplications make up a generator's skew action, apart from its
    derivation."""
    left_multiply = multiply(Vector, Octonion)                              # [] Octonion+ <- (Vector+, Octonion+)
    # The involution reverses the order of a product, so both maps take the multiplying vector first.
    right_multiply = -multiply(Vector, Octonion.involute()).involute()      # [] Octonion+ <- (Vector+, Octonion+)
    multiplications = stack([left_multiply, right_multiply])                # [sides] Octonion+ <- (Vector+, Octonion+)

    # Minus the trace of a composition is the positive pairing of two skew maps.
    projection_form = (
        -multiplications[:, None](Vector, multiplications[None, :]).trace(0, 3)
    ).field(1, 2)                                                           # [] (Scalar+ <- (Vector+, Vector+))[<- (sides, sides)]
    overlap = (-turn(Generator, multiplications).trace(0, 3)).field(2)      # [] (Scalar+ <- (Generator+, Vector+))[<- (, sides)]
    offsets = projection_form.solve(overlap)                                # [] (Vector+ <- Generator+)[sides]

    read_left = (mv.scalar([[1], [0]]) * Vector).field(1)                   # [] (Vector+ <- Vector+)[<- sides]
    read_right = (mv.scalar([[0], [1]]) * Vector).field(1)                  # [] (Vector+ <- Vector+)[<- sides]
    return read_left(offsets), read_right(offsets)


def linked_actions(generators: Generator, left_component: OffsetMap,
                   right_component: OffsetMap) -> tuple[OctonionMap, OctonionMap, OctonionMap]:
    """The product action and its two factor actions, obeying the product rule."""
    output = turn(generators, Octonion)                                     # [...] Octonion+ <- Octonion+
    left_offset = left_component(generators)                                # [...] Vector+
    right_offset = right_component(generators)                              # [...] Vector+
    derivation = (output - multiply(left_offset, Octonion)
                  - multiply(Octonion, right_offset))                       # [...] Octonion+ <- Octonion+
    left = (derivation + multiply(left_offset, Octonion)
            + multiply(Octonion, left_offset - right_offset))               # [...] Octonion+ <- Octonion+
    right = (derivation + multiply(right_offset - left_offset, Octonion)
             + multiply(Octonion, right_offset))                            # [...] Octonion+ <- Octonion+
    return output, left, right


def action_modes(left_component: OffsetMap,
                 right_component: OffsetMap) -> tuple[Scalar, Generator]:
    """A complete frame of rotation generators, ordered by their squared multiplication offsets."""
    constraint = (left_component.symmetric_conjugate_product()
                  + right_component.symmetric_conjugate_product())          # [] Scalar+ <- (Generator+, Generator+)
    # Clifford conjugation gives a positive pairing on both grades of a generator.
    metric = Generator.clifford_conjugate().scalar_product(Generator)       # [] Scalar+ <- (Generator+, Generator+)
    return constraint.eigh(metric)


def finite_actions(direction: Vector,
                   angles: np.ndarray) -> tuple[OctonionMap, OctonionMap, OctonionMap]:
    """A plane rotation and its two companions, for a unit vector direction."""
    # Powers of a single vector stay scalar plus vector; narrow the generic exponential.
    unit = (direction * (angles / 2)).exp().cast(Octonion)                  # [...] Octonion+
    left = multiply(unit, Octonion)                                         # [...] Octonion+ <- Octonion+
    right = multiply(Octonion, unit)                                        # [...] Octonion+ <- Octonion+
    output = multiply(unit, multiply(Octonion, unit))                       # [...] Octonion+ <- Octonion+
    return output, left, right
