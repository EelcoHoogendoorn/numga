"""Rolling a ball over a sphere three times its radius, planned by the product of split octonions, in
the geometric algebra of space.

A configuration of the ball, its contact direction and its attitude, is modelled as a multivector of space
without scalar part: its even part is the half turn in the plane through the centre square to the contact
direction, and its odd part the attitude followed by the reflection in that plane. The split-octonion product
`multiply` multiplies even parts by the geometric product and lets odd parts enter in swapped order, with
reversal for conjugation. Two configurations are one roll apart when their product vanishes: the kernel of multiplication by a
configuration holds every configuration one roll away, and maps preserving the product, split G2, carry routes
to routes. The contact direction is intrinsic and the half turn extrinsic, so identifying the two takes a sense
of turning, `handedness`; the other sense negates every odd part, which preserves the product.
"""

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                          # Scalar+
# The sense of turning that identifies a direction with the plane square to it, as a plane to turn in.
Handedness = Scalar.derive.flipped_orientation                       # Scalar-
Plane = ga.gatype.vector()                                           # Vector+
# Contact points are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()                                       # Antivector-
Rotor = ga.gatype.rotor()                                            # Rotor+
Octonion = ga.gatype.full()                                          # Full+
State = ga.gatype.nonscalar()                                        # Nonscalar+
StateMap = ga.gatype((State, State))
handedness = mv(Handedness, [1])                                     # [] Scalar-
# The split norm's bilinear form: contact part minus attitude part.
pairing = Octonion.clifford_conjugate().scalar_product(Octonion)     # [] Scalar+ <- (Octonion+, Octonion+)


# --- math -----------------------------------------------------------------------------
def multiply(first: Octonion, second: Octonion) -> Octonion:
    """The split-octonion product of two multivectors of space, doubling the even part: the even parts multiply
    by the geometric product, and the odd parts enter in swapped order, with reversal for conjugation."""
    first_even, first_odd = first.restrict.even(), first.restrict.odd()   # Even+, Odd+
    second_even, second_odd = second.restrict.even(), second.restrict.odd()   # Even+, Odd+
    # The odd parts enter in the opposite order; reversal supplies their conjugation.
    return (first_even * second_even + second_odd * first_odd.reverse()
            + first_even.reverse() * second_odd + second_even * first_odd)


def encode(contact: Point, attitude: Rotor) -> State:
    """A unit contact point and ball attitude as one null configuration."""
    # The half turn in the plane square to the contact, a bivector of the attitude's own, extrinsic kind, so that
    # the two multiply as quaternions; and the reflection in that plane, through the centre.
    half_turn = handedness * contact                                 # Bivector+
    mirror = contact.dual()                                          # Plane+
    return half_turn - mirror * attitude


def normalize(state: State) -> State:
    """The representative of a configuration whose contact direction has unit length."""
    return state / state.restrict[2].norm()


def lift(state: State, hemisphere: Plane) -> State:
    """The representative whose contact lies on the positive side of the plane `hemisphere`, through the
    sphere's centre."""
    contact = state.restrict[2] / handedness                         # Point-
    return normalize(state * (hemisphere & contact))


def decode(state: State) -> tuple[Point, Rotor]:
    """The contact point and ball attitude of a configuration."""
    state = normalize(state)                                         # State+
    contact = state.restrict[2] / handedness                         # Point-
    mirror = contact.dual()                                          # Plane+
    return contact, -mirror.inverse() * state.restrict.odd()


def annihilator(state: State) -> StateMap:
    """The projection onto the configurations one roll away from this one."""
    action = multiply(state, State)                                  # Octonion+ <- State+
    return State - action.lstsq(action)


def three_rolls(start: State, finish: State, preferred_contact: Point) -> tuple[State, State]:
    """Two intermediate configurations between endpoints that require three rolls, the first as near as the rolls
    allow to a contact at `preferred_contact`."""
    reachable = annihilator(start)                                   # State+ <- State+
    # Orthogonality to the finish means it is at most two rolls away.
    remaining = pairing(finish, reachable)                           # Scalar+ <- State+
    preference = handedness * preferred_contact                      # Bivector+
    first = reachable(preference - remaining.lstsq(remaining(preference)))   # State+
    # Their product is the unique configuration joining this two-roll pair.
    second = multiply(first, finish).cast(State)                     # State+
    return normalize(first), normalize(second)


def derivation(first: Octonion, second: Octonion) -> StateMap:
    """An infinitesimal product-preserving map, assembled from multiplication actions."""
    left_first = multiply(first, Octonion)                           # Octonion+ <- Octonion+
    left_second = multiply(second, Octonion)                         # Octonion+ <- Octonion+
    right_first = multiply(Octonion, first)                          # Octonion+ <- Octonion+
    right_second = multiply(Octonion, second)                        # Octonion+ <- Octonion+
    action = (left_first(left_second) - left_second(left_first)
              + left_first(right_second) - right_second(left_first)
              + right_first(right_second) - right_second(right_first))
    return action(State).cast(State)


def roll(start: State, finish: State, fractions: np.ndarray) -> State:
    """The configurations along the projective line between two configurations whose product vanishes: one
    roll."""
    return normalize(start * (1 - fractions) + finish * fractions)
