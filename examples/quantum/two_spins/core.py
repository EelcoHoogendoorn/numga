"""Two spins entangled by their exchange interaction, in the algebra of two copies of space.

One spin lives in the directions x y z, the other in X Y Z. The state of one spin is an even
multivector of its own space, as in the Pauli algebra; the state of the pair is a product of the
two, taken in the ideal of the correlator `CORRELATOR = 0.5 * (1 - xy * XY)`. In that ideal,
multiplying on the right by either spin's xy plane is the same: it is `IMAGINARY = CORRELATOR * xy`,
the pair's imaginary unit. The two spins' planes act on the pair from the left, and planes of
different spins commute. A state is normalized to `2 * (state >> ONE)` of one in its scalar part.

The exchange interaction couples each plane of one spin to the same plane of the other:
`COUPLING = yz * YZ + zx * ZX + xy * XY`. Since `COUPLING * COUPLING == 3 + 2 * COUPLING`, the
multivectors `(1 + COUPLING) / 4` and `(3 - COUPLING) / 4` are idempotent and sum to one: they split
every state into its singlet part, on which the coupling is 3, and its triplet part, on which it is
-1. Over time each part turns by its own rotor in the xy plane, multiplied on the right.

What one spin shows by itself is the part of its spin, `2 * (state >> IMAGINARY)`, in its own
planes, read as a vector of its space: its Bloch vector, of length one when the spin is in a state
of its own and shorter when the pair is entangled. What the two spins show together are their
correlations along a direction of each: a map from the second spin's directions to the first's,
read off the part of the density `2 * (state >> ONE)` that spans a plane of each spin.

The Bell combination of four correlations, along two directions of each spin, stays within 2 in any
account in which each spin carries its own answers. For a given state its largest value over all
directions is `2 * sqrt(s1**2 + s2**2)`, from the two largest singular values of the correlation map.
For given directions the combination is the expectation of a multivector acting on states from the
left, the Bell element, whose square is 4 minus four times the product of the plane spanned by the
first spin's directions and the plane spanned by the second's. Neither plane is larger than one, so
no state takes the combination past `2 * sqrt(2)`.

In bra-ket notation the state reads as a ket of two qubits, `-(pseudoscalar * direction) * state *
IMAGINARY` as the Pauli operator along a direction of one spin, `-COUPLING` as the Hamiltonian
sigma_1 . sigma_2, the Bloch vector of one spin as the trace of its reduced density matrix against
the Pauli matrices, the correlation map as the correlation tensor, and the Bell element as the CHSH
operator.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+z+X+Y+Z+")
context = NumpyContext(ga)
mv = context.multivector
ONE = mv.scalar([1.0])
Scalar = ga.gatype.scalar()
# The directions and the planes of each spin, and each spin's pseudoscalar.
First = ga.gatype(ga.subspace("x y z"))
Second = ga.gatype(ga.subspace("X Y Z"))
FirstPlanes = ga.gatype(ga.subspace("yz zx xy"))
SecondPlanes = ga.gatype(ga.subspace("YZ ZX XY"))
I_FIRST = mv.xyz                                                               # [] Trivector
I_SECOND = mv.XYZ                                                              # [] Trivector
# The state of one spin, and of the pair: a product of the two spins' states.
FirstSpinor = ga.gatype(ga.subspace("1 yz zx xy"))
SecondSpinor = ga.gatype(ga.subspace("1 YZ ZX XY"))
Spinor = ga.gatype((FirstSpinor * SecondSpinor).output_subspace)
Correlation = ga.gatype((First, Second))                                       # First <- Second
# The correlator, and the pair's imaginary unit.
CORRELATOR = 0.5 * (ONE - mv.xy * mv.XY)                                       # [] Spinor
IMAGINARY = CORRELATOR * mv.xy                                                 # [] Spinor
# The coupling of the exchange, and the singlet and triplet parts it splits the states into.
COUPLING = mv.yz * mv.YZ + mv.zx * mv.ZX + mv.xy * mv.XY                       # [] Spinor
SINGLET = 0.25 * (ONE + COUPLING)                                              # [] Spinor
TRIPLET = 0.25 * (3 - COUPLING)                                                # [] Spinor


# --- math -----------------------------------------------------------------------------
def exchange(state: Spinor, angle: np.ndarray) -> Spinor:
    """The state after the exchange interaction has acted for the given angle, the coupling strength
    times time: its singlet part turned by three times the angle, its triplet part back by the angle."""
    return SINGLET * state * (mv.xy * (3 * angle)).exp() + TRIPLET * state * (mv.xy * -angle).exp()   # [...] Spinor


def bloch(state: Spinor) -> tuple[First, Second]:
    """Each spin's Bloch vector: the part of the spin in its own planes, read as a vector of its space."""
    spin = 2 * (state >> IMAGINARY)                                            # [...] Bivector
    first = I_FIRST.inverse() * spin.cast(FirstPlanes)                         # [...] First
    second = I_SECOND.inverse() * spin.cast(SecondPlanes)                      # [...] Second
    return first, second


def correlation(state: Spinor) -> Correlation:
    """The correlations of the two spins, as a map from the second spin's directions to the first's:
    `a | correlation(b)` is the expected product of the two spins' values along a and b."""
    density = 2 * (state >> ONE)                                               # [...] Spinor
    return (I_FIRST.inverse() * (density * (I_SECOND * Second)).cast(FirstPlanes)).cast(Correlation)


def bell(correlations: Correlation) -> Scalar:
    """The largest Bell combination over all directions, from the two largest singular values of the
    correlation map."""
    values = correlations.svdvals()                                            # [..., 3] Scalar
    return 2 * (values[..., 0] ** 2 + values[..., 1] ** 2).square_root()      # [...] Scalar


def bell_element(first: First, first_other: First, second: Second, second_other: Second) -> Spinor:
    """The Bell combination along two directions of each spin, as a multivector acting on states from
    the left: the product of a plane of one spin with a plane of the other, `(I_FIRST * a) * (I_SECOND
    * b)`, is minus the product of the two spins' values along a and b."""
    return -((I_FIRST * first) * (I_SECOND * (second + second_other))
             + (I_FIRST * first_other) * (I_SECOND * (second - second_other)))   # [...] Spinor


def expectation(element: Spinor, state: Spinor) -> Scalar:
    """The expectation of a multivector acting on states from the left: the sandwich from the other
    side, `state.reverse() * element * state`."""
    return 2 * (state << element).select[0]                                    # [...] Scalar
