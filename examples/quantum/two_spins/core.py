"""Two spins entangled by their exchange interaction, in the algebra of two copies of space.

One spin lives in the directions x y z, the other in X Y Z. The state of one spin is an even
multivector of its own space, as in the Pauli algebra; the state of the pair is a product of the
two, taken in the ideal of the correlator `correlator = 0.5 * (1 - xy * XY)`. In that ideal,
multiplying on the right by either spin's xy plane is the same: it is the reference spin,
`reference_spin = correlator * xy`, which the state with both spins up carries to `xy + XY` and every
state to its own spin. The two spins' planes act on the pair from the left, and planes of
different spins commute. A state is normalized to one in the scalar part of its density,
`2 * state.symmetric_reverse_product()`.

The exchange interaction couples each plane of one spin to the same plane of the other:
`coupling = yz * YZ + zx * ZX + xy * XY`. Since `coupling * coupling == 3 + 2 * coupling`, the
multivectors `(1 + coupling) / 4` and `(3 - coupling) / 4` are idempotent and sum to one: they split
every state into its singlet part, on which the coupling is 3, and its triplet part, on which it is
-1. Over time each part turns by its own rotor in the xy plane, multiplied on the right.

What one spin shows by itself is the part of its spin, `2 * (state >> reference_spin)`, in its own
planes: the plane it turns in, of size one when the spin is in a state of its own and smaller when the
pair is entangled. A spin stays a plane here rather than the patch of its plane: the dual acts on the
whole algebra, so the patch of one spin's plane, `mv.xy.dual()`, spans the other spin's copy too. What
the two spins show together are their correlations in a plane of each: a map from the second spin's
planes to the first's, read off the part of the density that spans a plane of each spin, with
`a.dot(correlation(b))` the expected product of the two spins' values in the planes a and b.

In a field in the xy planes the singlet and the three triplets keep their shapes, and the field lowers
the triplet with both spins turning in its plane until it crosses the singlet. A field that differs
between the spins instead mixes the singlet with the triplet of zero spin in that plane: on those two
the generator of time applied twice is a number, so the pair turns between them by a cosine and a sine.

Measuring the first spin in a plane leaves the second in a new state: its plane after finding +1 in a
unit plane n is `(second + correlations.adjoint()(n)) / (1 + n.dot(first))`, so the correlation map,
read backwards, steers. Turning each spin by a rotor of its own space acts on the correlation map from
either side, which leaves its singular values alone: for every state of the pair they are one and
twice the concurrence C, and each spin's plane has size `sqrt(1 - C**2)`.

The Bell combination of four correlations, in two planes of each spin, stays within 2 in any account
in which each spin carries its own answers. For a given state its largest value over all planes is
`2 * sqrt(s1**2 + s2**2)`, from the two largest singular values of the correlation map. For given
planes the combination is the expectation of a multivector acting on states from the left, the Bell
element, whose square is 4 minus four times the product of the commutator of the first spin's two
planes and that of the second's. Neither commutator is larger than one, so no state takes the
combination past `2 * sqrt(2)`.

In bra-ket notation the state reads as a ket of two qubits, `-plane * state * reference_spin` as the
Pauli operator along the normal of a unit plane of one spin, right multiplication by `reference_spin`
as multiplication by i, `-coupling` as the Hamiltonian sigma_1 . sigma_2, the plane of one spin as the
Bloch vector, the trace of its reduced density matrix against the Pauli matrices, the correlation map
as the correlation tensor, and the Bell element as the CHSH operator.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+z+X+Y+Z+")
context = NumpyContext(ga)
mv = context.multivector
one = mv.scalar([1.0])
Scalar = ga.gatype.scalar()                                                    # Scalar+
# The planes of each spin.
FirstPlanes = ga.gatype.bivector().derive.within("x y z")                      # Bivector+
SecondPlanes = ga.gatype.bivector().derive.within("X Y Z")                     # Bivector+
# The state of one spin, and of the pair: a product of the two spins' states.
FirstSpinor = ga.gatype.even().derive.within("x y z")                          # Even+
SecondSpinor = ga.gatype.even().derive.within("X Y Z")                         # Even+
Spinor = (FirstSpinor * SecondSpinor).output_gatype                            # Even+
Correlation = ga.gatype((FirstPlanes, SecondPlanes))                           # FirstPlanes+ <- SecondPlanes+
# The correlator, and the reference spin, which every state carries to its own spin.
correlator = 0.5 * (one - mv.xy * mv.XY)                                       # [] Spinor+
reference_spin = correlator * mv.xy                                            # [] Spinor+
# The coupling of the exchange, and the singlet and triplet parts it splits the states into.
coupling = mv.yz * mv.YZ + mv.zx * mv.ZX + mv.xy * mv.XY                       # [] Spinor+
singlet = 0.25 * (one + coupling)                                              # [] Spinor+
triplet = 0.25 * (3 - coupling)                                                # [] Spinor+
# The plane of a field that turns both spins in their xy planes.
field_plane = mv.xy + mv.XY                                                    # [] Bivector+


# --- math -----------------------------------------------------------------------------
def exchange(state: Spinor, angle: np.ndarray) -> Spinor:
    """The state after the exchange interaction has acted for the given angle, the coupling strength
    times time: its singlet part turned by three times the angle, its triplet part back by the angle."""
    singlet_turn = (mv.xy * (3 * angle)).exp()                                 # [...] Rotor+
    triplet_turn = (mv.xy * -angle).exp()                                      # [...] Rotor+
    return singlet * state * singlet_turn + triplet * state * triplet_turn     # [...] Spinor+


def energy(state: Spinor, exchange_rate: np.ndarray, field: np.ndarray) -> Scalar:
    """The expected energy of a state under the exchange, shifted to minus the exchange rate on the
    singlet and plus it on the triplet, and a field in the xy planes that lowers each spin turning in
    its plane."""
    first, second = spin_planes(state)                                         # [...] FirstPlanes+, SecondPlanes+
    return exchange_rate * expectation(0.5 * (one - coupling), state) - 0.5 * field * field_plane.dot(first + second)   # [...] Scalar+


def qubit_turn(state: Spinor, exchange_rate: np.ndarray, difference: np.ndarray, times: np.ndarray) -> Spinor:
    """A state of zero spin in the xy planes after the given times under the exchange and a field in
    those planes that differs between the spins. Such a state is a combination of the singlet and the
    triplet of zero spin in the xy planes, on which the generator applied twice is minus the rate
    squared: its exponential is a cosine and a sine, as for a rotor."""
    # The exchange, minus one on the singlet and one on the triplet, turned about the reference spin; and
    # the field difference, turning the first spin's xy plane against the second's.
    generator = -exchange_rate * (0.5 * (one - coupling) * state) * reference_spin - difference * 0.5 * (mv.xy - mv.XY) * state   # [...] Spinor+
    rate = np.sqrt(exchange_rate**2 + difference**2)
    return np.cos(rate * times) * state + np.sin(rate * times) / rate * generator   # [...] Spinor+


def qubit(state: Spinor) -> tuple[Scalar, Scalar, Scalar]:
    """The pair's place on the sphere of the qubit made of the singlet and the triplet of zero spin in
    the xy planes: how far it leans to up-down against down-up, half the difference of the two spins'
    parts in their xy planes; the turn between them, the expectation of a product of the shifted exchange
    and the difference of the two xy planes; and the balance of singlet against triplet."""
    first, second = spin_planes(state)                                         # [...] FirstPlanes+, SecondPlanes+
    across = 0.5 * (mv.xy.dot(first) - mv.XY.dot(second))                      # [...] Scalar+
    turned = expectation(-0.25 * (one - coupling) * (mv.xy - mv.XY), state)    # [...] Scalar+
    balance = expectation(0.5 * (coupling - one), state)                       # [...] Scalar+
    return across, turned, balance


def spin_planes(state: Spinor) -> tuple[FirstPlanes, SecondPlanes]:
    """The plane each spin turns in: the part of the pair's spin in that spin's own planes."""
    spin = 2 * (state >> reference_spin)                                       # [...] Bivector+
    return spin.cast(FirstPlanes), spin.cast(SecondPlanes)


def correlation(state: Spinor) -> Correlation:
    """The correlations of the two spins, as a map from the second spin's planes to the first's:
    `a.dot(correlation(b))` is the expected product of the two spins' values in the planes a and b."""
    density = 2 * state.symmetric_reverse_product()                            # [...] Spinor+
    return (density * SecondPlanes).cast(FirstPlanes)                          # [...] FirstPlanes+ <- SecondPlanes+


def steer(state: Spinor, plane: FirstPlanes) -> tuple[Scalar, SecondPlanes]:
    """Measuring the first spin in a unit plane and finding +1: how likely that is, and the second
    spin's plane after. The first spin's value in the plane acts on the pair from the left as
    `-plane * state * reference_spin`, and the measurement keeps the part of the state on which it
    is +1."""
    measured = 0.5 * (state - plane * state * reference_spin)                  # [...] Spinor+
    probability = expectation(one, measured)                                   # [...] Scalar+
    _, steered = spin_planes(measured / probability.square_root())             # [...] SecondPlanes+
    return probability, steered


def bell(correlations: Correlation) -> Scalar:
    """The largest Bell combination over all planes, from the two largest singular values of the
    correlation map."""
    values = correlations.svdvals()                                            # [..., 3] Scalar+
    return 2 * (values[..., 0] ** 2 + values[..., 1] ** 2).square_root()       # [...] Scalar+


def bell_element(first: FirstPlanes, first_other: FirstPlanes, second: SecondPlanes, second_other: SecondPlanes) -> Spinor:
    """The Bell combination in two planes of each spin, as a multivector acting on states from the
    left: the product of a plane of one spin with a plane of the other, `a * b`, is minus the product
    of the two spins' values in a and b."""
    return -(first * (second + second_other) + first_other * (second - second_other))   # [...] Spinor+


def expectation(element: Spinor, state: Spinor) -> Scalar:
    """The expectation of a multivector acting on states from the left: the sandwich from the other
    side, `state.reverse() * element * state`."""
    return 2 * (state << element).select[0]                                    # [...] Scalar+
