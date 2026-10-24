"""Rolling a ball over a sphere three times its radius, planned by the product of split octonions, in
the geometric algebra of space.

A split octonion is a multivector of space whose even part, and odd part divided by the volume, are two
even multivectors, coupled by `multiply`. A configuration of the ball, its contact direction and its
attitude, is a null split octonion, up to scale, and two configurations are one roll apart when their
product vanishes: the kernel of multiplication by a configuration holds every configuration one roll
away, and maps preserving the product, split G2, carry routes to routes.
"""

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# Contact points are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
Even = ga.gatype.even()
Octonion = ga.gatype.full()
State = ga.gatype.nonscalar()
StateMap = ga.gatype((State, State))
volume = mv.xyz


# --- math -----------------------------------------------------------------------------
def multiply(first: Octonion, second: Octonion) -> Octonion:
    """Split-octonion multiplication, packing two even multivectors into even and odd grades."""
    first_even = first.restrict.even()                         # Even
    second_even = second.restrict.even()                       # Even
    first_partner = first.restrict.odd() * volume.inverse()     # Even
    second_partner = second.restrict.odd() * volume.inverse()   # Even
    # The partners enter in the opposite order; reversal supplies their conjugation.
    even = first_even * second_even + second_partner * first_partner.reverse()  # Even
    partner = first_even.reverse() * second_partner + second_even * first_partner  # Even
    return even + partner * volume


def pairing(first: Octonion, second: Octonion) -> Scalar:
    """The split norm's bilinear form: contact part minus attitude part."""
    return first.clifford_conjugate().scalar_product(second)


def encode(contact: Point, attitude: Rotor) -> State:
    """A unit contact point and ball attitude as one null split octonion."""
    # The contact enters by the plane through the centre square to it, `contact.dual_inverse()`, times the volume: a
    # bivector of the attitude's own, extrinsic kind, so that the two multiply as quaternions.
    contact_plane = contact.dual_inverse() * volume            # Bivector
    return contact_plane + contact_plane * attitude * volume


def normalize(state: State) -> State:
    """Choose the null representative with unit contact direction."""
    return state / state.restrict[2].norm()


def lift(state: State, hemisphere: Vector) -> State:
    """Choose the representative whose contact lies on the positive side of the plane `hemisphere`, through the
    sphere's centre."""
    contact = (state.restrict[2] / volume).dual()              # Point
    return normalize(state * (hemisphere & contact))


def decode(state: State) -> tuple[Point, Rotor]:
    """Recover the contact point and ball attitude from a null representative."""
    state = normalize(state)                                   # State
    contact_plane = state.restrict[2]                          # Bivector
    partner = state.restrict.odd() / volume                    # Even
    return (contact_plane / volume).dual(), contact_plane.inverse() * partner


def annihilator(state: State) -> StateMap:
    """Project onto configurations reachable from this one in a single roll."""
    action = multiply(state, State)                            # Octonion <- State
    return State - action.lstsq(action)


def three_rolls(start: State, finish: State, contact_preference: Bivector) -> tuple[State, State]:
    """Two intermediate configurations between endpoints that require three rolls."""
    reachable = annihilator(start)                            # State <- State
    # Orthogonality to the finish means it is at most two rolls away.
    remaining = pairing(finish, reachable)                    # Scalar <- State
    first = reachable(contact_preference - remaining.lstsq(remaining(contact_preference)))  # State
    # Their product is the unique configuration joining this two-roll pair.
    second = multiply(first, finish).cast(State)               # State
    return normalize(first), normalize(second)


def derivation(first: Octonion, second: Octonion) -> StateMap:
    """An infinitesimal product-preserving map, assembled from multiplication actions."""
    left_first = multiply(first, Octonion)                     # Octonion <- Octonion
    left_second = multiply(second, Octonion)                   # Octonion <- Octonion
    right_first = multiply(Octonion, first)                    # Octonion <- Octonion
    right_second = multiply(Octonion, second)                  # Octonion <- Octonion
    action = (left_first(left_second) - left_second(left_first)
              + left_first(right_second) - right_second(left_first)
              + right_first(right_second) - right_second(right_first))
    return action(State).cast(State)


def roll(start: State, finish: State, fractions: np.ndarray) -> State:
    """Sample the projective line between two configurations whose product vanishes."""
    return normalize(start * (1 - fractions) + finish * fractions)
