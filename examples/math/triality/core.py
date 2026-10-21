"""Spin(8) triality links three rotations through octonion multiplication.

One rotation acts on the product and two act on its factors. Splitting a skew map
into left multiplication, right multiplication and a derivation constructs its
companions. Requiring all three maps to coincide leaves the fourteen generators
of G2.
"""

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a-b-c-d-e-f-g-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Octonion = ga.gatype.from_grades([0, 1])
Rate = ga.gatype.from_grades([1, 2])
OctonionMap = ga.gatype((Octonion, Octonion))
OffsetMap = ga.gatype((Vector, Rate))

cycles = stack([mv.abc, mv.ade, mv.agf, mv.bdf, mv.beg, mv.cdg, mv.cfe])
# The seven oriented lines of the Fano plane, each a quaternion multiplication cycle.
fano = cycles.sum(axis=0)
multiply = ((Octonion * Octonion) * (1 - fano)).cast(Octonion)

# Bivectors turn the seven vector components; vectors turn them into the scalar.
turn = (Rate.restrict.bivector().commutator(Octonion)
        + Rate.restrict.vector().anticommutator(Octonion))                    # Octonion <- (Rate, Octonion)


# --- math -----------------------------------------------------------------------------
def multiplication_components() -> tuple[OffsetMap, OffsetMap]:
    """Project a skew action onto its left and right multiplication components."""
    left_multiply = multiply(Vector, Octonion)                               # Octonion <- (Vector, Octonion)
    # Conjugation reverses a product, so both maps take the multiplying vector first.
    right_multiply = -multiply(Vector, Octonion.involute()).involute()        # Octonion <- (Vector, Octonion)
    multiplications = stack([left_multiply, right_multiply])                 # [sides] Octonion <- (Vector, Octonion)

    # Minus the trace of a composition is the positive pairing of two skew maps.
    projection_form = (
        -multiplications[:, None](Vector, multiplications[None, :]).trace(0, 3)
    ).field(1, 2)                                                           # Scalar <- (Vector[2], Vector[2])
    overlap = (-turn(Rate, multiplications).trace(0, 3)).field(2)             # Scalar <- (Rate, Vector[2])
    offsets = projection_form.solve(overlap)                                # Vector[2] <- Rate

    read_left = (mv.scalar([[1], [0]]) * Vector).field(1)                     # Vector <- Vector[2]
    read_right = (mv.scalar([[0], [1]]) * Vector).field(1)                     # Vector <- Vector[2]
    return read_left(offsets), read_right(offsets)


def linked_actions(rates: Rate, left_component: OffsetMap,
                   right_component: OffsetMap) -> tuple[OctonionMap, OctonionMap, OctonionMap]:
    """The product action and its two factor actions, obeying the product rule."""
    output = turn(rates, Octonion)                                          # [...] Octonion <- Octonion
    left_offset = left_component(rates)                                    # [...] Vector
    right_offset = right_component(rates)                                  # [...] Vector
    derivation = (output - multiply(left_offset, Octonion)
                  - multiply(Octonion, right_offset))                      # [...] Octonion <- Octonion
    left = (derivation + multiply(left_offset, Octonion)
            + multiply(Octonion, left_offset - right_offset))               # [...] Octonion <- Octonion
    right = (derivation + multiply(right_offset - left_offset, Octonion)
             + multiply(Octonion, right_offset))                            # [...] Octonion <- Octonion
    return output, left, right


def action_modes(left_component: OffsetMap,
                 right_component: OffsetMap) -> tuple[Scalar, Rate]:
    """A complete rotation frame, ordered by the squared multiplication offsets."""
    constraint = (left_component.involute().scalar_product(left_component)
                  + right_component.involute().scalar_product(right_component))
    metric = Rate.clifford_conjugate().scalar_product(Rate)
    return constraint.eigh(metric)


def finite_actions(direction: Vector,
                   angles: np.ndarray) -> tuple[OctonionMap, OctonionMap, OctonionMap]:
    """A plane rotation and its two companions, for a unit vector direction."""
    # Powers of a single vector stay scalar plus vector; narrow the generic exponential.
    unit = (direction * (angles / 2)).exp().cast(Octonion)                   # [...] Octonion
    left = multiply(unit, Octonion)                                        # [...] Octonion <- Octonion
    right = multiply(Octonion, unit)                                       # [...] Octonion <- Octonion
    output = multiply(unit, multiply(Octonion, unit))                       # [...] Octonion <- Octonion
    return output, left, right
