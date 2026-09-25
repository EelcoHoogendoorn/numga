"""The spin groups: the rotors of every signature up to six dimensions, inside one algebra.

The algebra has six directions squaring to plus one, x y z w v u, and three squaring to minus one,
t s r. Choosing p of the first and q of the second picks out the signature (p, q): the bivectors of
the chosen directions generate its rotors, the spin group, and their even products are its spinors.
Three maps with open slots tell the groups apart.

The invariant form of the rotors' Lie algebra, the trace of the double commutator with two
bivectors left open, is the algebra's own inner product on those bivectors, times `2 * (n - 2)`:
the geometric product carries it already.

Sandwiching the bivectors with the product of the chosen positive directions is an involution: it
negates exactly the planes that mix a positive with a negative direction. The planes it fixes square
to minus one and generate rotations; the planes it negates square to plus one and generate boosts. There are `p * (p - 1) // 2 + q * (q - 1) // 2` of the first and `p * q` of the
second, and a group is compact when every plane is a rotation.

In an even number of dimensions the pseudoscalar of the chosen directions commutes with every
spinor. When it squares to plus one, `0.5 * (1 + I)` and `0.5 * (1 - I)` split the spinors into two
halves; when it squares to minus one, it acts on the spinors as the complex unit. In four dimensions
it also maps bivectors to bivectors, so the Lie algebra itself splits in two, or becomes complex. In
four Euclidean dimensions each generator is the sum of two commuting halves, and the rotor the
product of their exponentials; each factor turns every plane through one angle, and their orbits on
the three-sphere are the fibres of the Hopf fibration and of its mirror image.

In the notation of Lie groups the signatures read as Spin(3) = SU(2), Spin(2,1) = SL(2,R),
Spin(4) = SU(2) x SU(2), Spin(3,1) = SL(2,C), Spin(2,2) = SL(2,R) x SL(2,R), Spin(5) = Sp(2),
Spin(4,1) = Sp(1,1), Spin(3,2) = Sp(4,R), Spin(6) = SU(4), Spin(5,1) = SL(2,H), Spin(4,2) = SU(2,2)
and Spin(3,3) = SL(4,R); the invariant form as the Killing form, and the involution as the Cartan
involution.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from numga import Algebra, NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType

ga = Algebra("x+y+z+w+v+u+t-s-r-")
context = NumpyContext(ga)
mv = context.multivector
POSITIVE, NEGATIVE = "xyzwvu", "tsr"
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# The four Euclidean directions of the isoclinic rotations.
Euclidean = ga.gatype(ga.subspace("x y z w"))


# --- math -----------------------------------------------------------------------------
def invariant_form(Bivector: GAType) -> Extensor:
    """The invariant form of the Lie algebra: the trace of the double commutator, with two bivectors
    open, exact in integers."""
    return Bivector.commutator(Bivector.commutator(Bivector)).trace(slot=2)   # [] Scalar <- (Bivector, Bivector)


def involution(Bivector: GAType, positives: Extensor) -> Extensor:
    """The bivectors sandwiched with the product of the positive directions: plus one on the planes of
    rotations, minus one on the planes of boosts."""
    return positives >> Bivector                                              # [] Bivector <- Bivector


def isoclinic(generator: Extensor, pseudoscalar: Extensor) -> tuple[Extensor, Extensor]:
    """A four-dimensional generator as the sum of two commuting generators, `0.5 * (1 + I)` and
    `0.5 * (1 - I)` times it: bivectors again, one turning each plane along with its dual plane,
    the other against it. Their exponentials multiply to the generator's rotor."""
    plus, minus = 0.5 * (1 + pseudoscalar), 0.5 * (1 - pseudoscalar)          # [] Even each
    return plus * generator, minus * generator                                # [...] Bivector each


def stereographic(points: Extensor) -> Extensor:
    """Unit vectors of the four Euclidean directions, projected from -w into the space of x y z."""
    return (points - mv.w * (points | mv.w)) / (1 + (points | mv.w))   # [...] Euclidean


def linking(first: Extensor, second: Extensor) -> Extensor:
    """Gauss's linking number of two closed polygons in the space of x y z: the volume each pair of
    segments spans with the line between them, over the cube of its length, summed and divided by
    four pi."""
    step_first = first[1:] - first[:-1]                                          # [n] Euclidean
    step_second = second[1:] - second[:-1]                                       # [m] Euclidean
    middle_first = 0.5 * (first[1:] + first[:-1])                                # [n] Euclidean
    middle_second = 0.5 * (second[1:] + second[:-1])                             # [m] Euclidean
    separation = middle_first[:, None] - middle_second[None, :]                  # [n, m] Euclidean
    # The trivector the separation spans with the two segments, measured against the unit volume xyz.
    volume = (separation ^ step_first[:, None] ^ step_second[None, :]) | mv.xyz.inverse()   # [n, m] Scalar
    return (volume / (separation | separation).square_root() ** 3).sum() / (4 * np.pi)   # [] Scalar


# --- plumbing -------------------------------------------------------------------------
def signature(p: int, q: int) -> tuple[GAType, GAType, Extensor, Extensor]:
    """The first p positive and q negative directions: their bivectors and even multivectors as
    types, the product of the positive ones, and their pseudoscalar."""
    names = POSITIVE[:p] + NEGATIVE[:q]
    closure = ga.gatype(ga.subspace(" ".join(names))).minimal_subalgebra.output_subspace
    Bivector = ga.gatype(closure.intersection(ga.subspace.bivector()))
    Even = ga.gatype(closure.intersection(ga.subspace.even()))
    one = mv.scalar([1.0])
    positives = reduce(lambda product, name: product * getattr(mv, name), POSITIVE[:p], one)
    pseudoscalar = reduce(lambda product, name: product * getattr(mv, name), names, one)
    return Bivector, Even, positives, pseudoscalar
