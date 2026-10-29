"""The spin groups: the rotors of every signature up to six dimensions, inside one algebra.

The algebra has six directions squaring to plus one, x y z w v u, and three squaring to minus one,
t s r. Choosing p of the first and q of the second picks out the signature (p, q): the bivectors of
the chosen directions generate its rotors, the spin group, and their even products are its spinors.
Three maps, built from types alone, tell the groups apart.

The invariant form of the rotors' Lie algebra, the trace of the double commutator, a form
on pairs of bivectors, is the algebra's own inner product on those bivectors, times `2 * (n - 2)`:
the geometric product carries it already.

Sandwiching the bivectors with the product of the chosen positive directions is an involution: it
negates exactly the planes that mix a positive with a negative direction. The planes it fixes square
to minus one and generate rotations; the planes it negates square to plus one and generate boosts. There are `p * (p - 1) // 2 + q * (q - 1) // 2` of the first and `p * q` of the
second, and a group is compact when every plane is a rotation.

In an even number of dimensions the pseudoscalar of the chosen directions commutes with every
spinor. When it squares to plus one, `0.5 * (1 + pseudoscalar)` and `0.5 * (1 - pseudoscalar)` split
the spinors into two halves; when it squares to minus one, it acts on the spinors as a quarter turn of
every pair of their coefficients alike. In four dimensions it also maps bivectors to bivectors, so the
Lie algebra itself splits in two, or pairs each rotation with the boost it turns into. In four
Euclidean dimensions each generator is the sum of two commuting halves, and the rotor the product of
their exponentials; each factor turns every plane through one angle, and their orbits on the
three-sphere are the fibres of the Hopf fibration and of its mirror image.

In the notation of Lie groups the signatures read as Spin(3) = SU(2), Spin(2,1) = SL(2,R),
Spin(4) = SU(2) x SU(2), Spin(3,1) = SL(2,C), Spin(2,2) = SL(2,R) x SL(2,R), Spin(5) = Sp(2),
Spin(4,1) = Sp(1,1), Spin(3,2) = Sp(4,R), Spin(6) = SU(4), Spin(5,1) = SL(2,H), Spin(4,2) = SU(2,2)
and Spin(3,3) = SL(4,R); the invariant form as the Killing form, and the involution as the Cartan
involution.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType

ga = Algebra("x+y+z+w+v+u+t-s-r-")
context = NumpyContext(ga)
mv = context.multivector
positive_names, negative_names = "xyzwvu", "tsr"
# The points of the four Euclidean directions x y z w: the antivectors that contain all five others.
Euclidean = ga.gatype.antivector().derive.containing("v u t s r")   # Antivector-
Antitrivector = ga.gatype.antitrivector()                           # Antitrivector-

# The pseudoscalar of the four Euclidean directions commutes with their bivectors and squares to one:
# its two halves split every generator into two commuting generators, one turning each plane along
# with its dual plane, the other against it.
euclidean_pseudoscalar = mv.xyzw                                                     # [] Quadvector+
plus, minus = 0.5 * (1 + euclidean_pseudoscalar), 0.5 * (1 - euclidean_pseudoscalar)  # [] Even+, Even+


# --- math -----------------------------------------------------------------------------
def stereographic(points: Euclidean) -> Euclidean:
    """Points of the three-sphere projected from the pole at -w into the space of x y z."""
    w_coordinate = mv.w & points                                                # [...] Scalar+
    return (points - mv.w.dual() * w_coordinate) / (1 + w_coordinate)           # [...] Antivector-


def linking(first: Euclidean, second: Euclidean) -> Antitrivector:
    """Gauss's linking number of two closed polygons in the space of x y z, as a volume: the volume
    each pair of segments spans with the line between them, over the cube of its length, summed and
    divided by four pi. It is a whole number of unit volumes, oriented by the sense in which the
    polygons link; their mirror images link the other way."""
    step_first = first[1:] - first[:-1]                                          # [segments] Antivector-
    step_second = second[1:] - second[:-1]                                       # [segments] Antivector-
    middle_first = 0.5 * (first[1:] + first[:-1])                                # [segments] Antivector-
    middle_second = 0.5 * (second[1:] + second[:-1])                             # [segments] Antivector-
    separation = middle_first[:, None] - middle_second[None, :]                  # [segments, segments] Antivector-
    # The volume the separation spans with the two segments, their join.
    volume = separation & step_first[:, None] & step_second[None, :]             # [segments, segments] Antitrivector-
    # In this algebra an antivector squares to minus its length squared.
    length = separation.scalar_norm_squared().abs().square_root()                # [segments, segments] Scalar+
    return (volume / length ** 3).sum() / (4 * np.pi)                            # [] Antitrivector-


# --- plumbing -------------------------------------------------------------------------
def signature(p: int, q: int) -> tuple[GAType, GAType, Extensor, Extensor]:
    """The first p positive and q negative directions: their bivectors and even multivectors as
    types, the product of the positive ones, and their pseudoscalar."""
    names = positive_names[:p] + negative_names[:q]
    Bivector = ga.gatype.bivector().derive.within(" ".join(names))
    Even = ga.gatype.even().derive.within(" ".join(names))
    positives = getattr(mv, positive_names[:p])
    pseudoscalar = getattr(mv, names)
    return Bivector, Even, positives, pseudoscalar
