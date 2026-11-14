"""Spin(8) triality: three rotations of eight-dimensional space linked by the octonion product.

The vectors of Spin(8) and its two spinor spaces each have eight components, and the octonion product links them so that rotating a product agrees with multiplying its two factors after turning each by a companion rotation of its own. Requiring all three rotations to coincide leaves the fourteen generators of G2.

An octonion is modelled as a scalar plus a vector in seven-dimensional negative-definite space, and the octonion product as one bilinear map `multiply`. A generator of rotations of the eight components is a vector, turning the scalar into the vectors, plus a bivector, turning the vectors among themselves. Each generator splits into multiplication by a vector on the left, multiplication by a vector on the right, and a derivation obeying the product rule; minus the trace of a composition pairs skew maps, so one solve finds the split, and redistributing the two multiplication parts gives each generator its two companions.
"""

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a-b-c-d-e-f-g-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                 # Scalar+
Vector = ga.gatype.vector()                                                 # Vector+
Octonion = ga.gatype.from_grades([0, 1])                                    # Scalar+ + Vector+
Generator = ga.gatype.from_grades([1, 2])                                   # Vector+ + Bivector+
OctonionMap = ga.gatype((Octonion, Octonion))                               # Octonion+ <- Octonion+
OffsetMap = ga.gatype((Vector, Generator))                                  # Vector+ <- Generator+

# Each oriented triple closes under the product, `a` times `b` giving `c`; summed, the seven triples are the
# oriented lines of the Fano plane, one trivector that carries the whole multiplication table.
cycles = stack([mv.abc, mv.ade, mv.agf, mv.bdf, mv.beg, mv.cdg, mv.cfe])   # [7] Trivector+
fano = cycles.sum(axis=0)                                                   # [] Trivector+
multiply = ((Octonion * Octonion) * (1 - fano)).cast(Octonion)              # [] Octonion+ <- (Octonion+, Octonion+)

# Bivectors turn the seven vector components; vectors turn them into the scalar.
turn = (Generator.restrict.bivector().commutator(Octonion)
        + Generator.restrict.vector().anticommutator(Octonion))             # [] Octonion+ <- (Generator+, Octonion+)


# --- math -----------------------------------------------------------------------------
def finite_actions(direction: Vector,
                   angles: np.ndarray) -> tuple[OctonionMap, OctonionMap, OctonionMap]:
    """A plane rotation and its two companions (`[...] Octonion+ <- Octonion+`) for a unit vector direction."""
    # Powers of a single vector stay scalar plus vector; narrow the generic exponential.
    unit = (direction * (angles / 2)).exp().cast(Octonion)                  # [...] Octonion+
    left = multiply(unit, Octonion)                                         # [...] Octonion+ <- Octonion+
    right = multiply(Octonion, unit)                                        # [...] Octonion+ <- Octonion+
    output = multiply(unit, multiply(Octonion, unit))                       # [...] Octonion+ <- Octonion+
    return output, left, right


def multiplication_components() -> tuple[OffsetMap, OffsetMap]:
    """The left and right multiplication offset maps (`Vector+ <- Generator+`) of an infinitesimal rotation."""
    left_multiply = multiply(Vector, Octonion)                              # [] Octonion+ <- (Vector+, Octonion+)
    # Grade involution reverses the order of an octonion product, so both maps take the multiplying vector first.
    right_multiply = -multiply(Vector, Octonion.involute()).involute()      # [] Octonion+ <- (Vector+, Octonion+)
    multiplications = stack([left_multiply, right_multiply])                # [2] Octonion+ <- (Vector+, Octonion+)

    # Minus the trace of a composition is the positive pairing of two skew maps.
    projection_form = (
        -multiplications[:, None](Vector, multiplications[None, :]).trace(0, 3)
    ).field(1, 2)                                                           # [] (Scalar+ <- (Vector+, Vector+))[2 <- 2]
    overlap = (-turn(Generator, multiplications).trace(0, 3)).field(2)      # [] (Scalar+ <- (Generator+, Vector+))[2]
    left_component, right_component = projection_form.solve(overlap).batch()  # [] Vector+ <- Generator+, Vector+ <- Generator+
    return left_component, right_component


def action_modes(left_component: OffsetMap,
                 right_component: OffsetMap) -> tuple[Scalar, Generator]:
    """A complete frame of 28 rotation generators (`[28] Generator+`), ordered by squared multiplication offsets (`[28] Scalar+`)."""
    constraint = (left_component.symmetric_conjugate_product()
                  + right_component.symmetric_conjugate_product())          # [] Scalar+ <- (Generator+, Generator+)
    # Clifford conjugation gives a positive pairing on both grades of a generator.
    metric = Generator.clifford_conjugate().scalar_product(Generator)       # [] Scalar+ <- (Generator+, Generator+)
    return constraint.eigh(metric)


def linked_actions(generators: Generator, left_component: OffsetMap,
                   right_component: OffsetMap) -> tuple[OctonionMap, OctonionMap, OctonionMap]:
    """The product action and its two factor actions (`[...] Octonion+ <- Octonion+`), obeying the product rule."""
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
