"""Split-octonion multiplication in the geometric algebra of space plans the rolling of a ball over a sphere three times its radius.

Each ball configuration combines its contact direction and attitude into one scalar-free multivector whose even part is the half turn in the plane perpendicular to the contact direction and whose odd part is the attitude followed by reflection in that plane. The split-octonion product multiplies even parts by the geometric product while reversing and swapping odd parts. Two configurations lie one great-circle roll apart whenever their split-octonion product vanishes, so the multiplication kernel of a starting configuration gives every configuration reachable in one roll, and product-preserving maps in split G2 carry whole routes to new routes. Identifying an intrinsic contact direction with an extrinsic turning plane uses a unit chiral scalar `handedness`; negating that choice negates every odd part and preserves the product.
"""

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
# Unit chiral scalar identifying an intrinsic contact direction with the extrinsic plane perpendicular to it.
Handedness = ga.gatype.antipseudoscalar()                            # Antipseudoscalar-
Plane = ga.gatype.vector()                                           # Vector+
Point = ga.gatype.antivector()                                       # Antivector-
Rotor = ga.gatype.rotor()                                            # Rotor+
Octonion = ga.gatype.full()                                          # Full+
State = ga.gatype.nonscalar()                                        # Nonscalar+
StateMap = ga.gatype((State, State))                                 # State+ <- State+
handedness = mv(Handedness, [1])                                     # [] Scalar-
# Bilinear form of the split norm: even contact norm squared minus odd attitude norm squared.
pairing = Octonion.clifford_conjugate().scalar_product(Octonion)     # [] Scalar+ <- (Octonion+, Octonion+)


# --- math -----------------------------------------------------------------------------
def split_product(first: Octonion, second: Octonion) -> Octonion:
    """Cayley-Dickson doubling across the even and odd grades multiplies two space multivectors (`[...] Octonion+`) as split octonions."""
    first_even, first_odd = first.restrict.even(), first.restrict.odd()        # [...] Even+, Odd+
    second_even, second_odd = second.restrict.even(), second.restrict.odd()    # [...] Even+, Odd+
    # Odd parts multiply in swapped order, with reversal acting as quaternion conjugation.
    return (first_even * second_even + second_odd * first_odd.reverse()
            + first_even.reverse() * second_odd + second_even * first_odd)


def configuration(contact: Point, attitude: Rotor) -> State:
    """A unit contact point (`[...] Point-`) and ball attitude (`[...] Rotor+`) combine into one null configuration (`[...] State+`)."""
    # Convert the intrinsic contact point into an extrinsic half-turn bivector in its tangent plane,
    # and form the central reflection plane dual to the contact point.
    half_turn = handedness * contact                                 # [...] Bivector+
    mirror = contact.dual()                                          # [...] Plane+
    return half_turn - mirror * attitude


def contact_and_attitude(state: State) -> tuple[Point, Rotor]:
    """Unit-contact normalization recovers the contact point (`[...] Point-`) and ball attitude (`[...] Rotor+`) from a configuration (`[...] State+`)."""
    unit_state = state / state.restrict[2].norm()                    # [...] State+
    contact = unit_state.restrict[2] / handedness                    # [...] Point-
    mirror = contact.dual()                                          # [...] Plane+
    return contact, -mirror.inverse() * unit_state.restrict.odd()


def hemisphere_lift(state: State, hemisphere: Plane) -> State:
    """Projective sign selection chooses the unit-contact representative (`[...] State+`) whose contact lies on the positive side of `hemisphere`."""
    contact = state.restrict[2] / handedness                         # [...] Point-
    signed_state = state * (hemisphere & contact)                    # [...] State+
    return signed_state / signed_state.restrict[2].norm()


def contact_arcs(stops: State, n_samples: int) -> tuple[Point, ...]:
    """Linear interpolation between consecutive `stops` samples `n_samples` contact points (`[n_samples] Point-`) along each great-circle roll."""
    fractions = np.linspace(0, 1, n_samples)
    states = stops[:-1, None] * (1 - fractions) + stops[1:, None] * fractions   # [..., n_samples] State+
    contacts, _ = contact_and_attitude(states)                                  # [..., n_samples] Point-, Rotor+
    return tuple(contacts)


def route_motion(stops: State, n_steps: int, n_hold_frames: int) -> Iterator[tuple[Point, Rotor]]:
    """Yields `(Point-, Rotor+)` ball placements across `n_steps` interpolation steps per roll of `stops`, holding each endpoint for `n_hold_frames` frames."""
    fractions = np.linspace(0, 1, n_steps)
    for _ in range(n_hold_frames):
        yield contact_and_attitude(stops[0])
    for first, second in zip(stops[:-1], stops[1:]):
        for fraction in fractions:
            yield contact_and_attitude(first * (1 - fraction) + second * fraction)
    for _ in range(n_hold_frames):
        yield contact_and_attitude(stops[-1])


def annihilator(state: State) -> StateMap:
    """Orthogonal projection onto the left-multiplication kernel isolates the configurations (`State+ <- State+`) one roll away from `state`."""
    left_action = split_product(state, State)                        # [] Octonion+ <- State+
    return State - left_action.lstsq(left_action)


def three_roll_waypoints(start: State, finish: State, preferred_contact: Point) -> tuple[State, State]:
    """Kernel projection and one split-octonion product determine two intermediate waypoints (`State+`) between endpoints three rolls apart."""
    first_roll_space = annihilator(start)                            # [] State+ <- State+
    # Vanishing split pairing with the destination puts a first waypoint within two rolls of `finish`.
    finish_pairing = pairing(finish, first_roll_space)               # [] Scalar+ <- State+
    preference = handedness * preferred_contact                      # [] Bivector+
    first = first_roll_space(preference - finish_pairing.lstsq(finish_pairing(preference)))   # [] State+
    # The split-octonion product supplies the unique waypoint joining a two-roll pair.
    second = split_product(first, finish).cast(State)                # [] State+
    return first / first.restrict[2].norm(), second / second.restrict[2].norm()


def derivation(first: Octonion, second: Octonion) -> StateMap:
    """Commutators of left and right split-octonion multiplications form an infinitesimal automorphism (`State+ <- State+`) of the product."""
    left_first = split_product(first, Octonion)                      # [] Octonion+ <- Octonion+
    left_second = split_product(second, Octonion)                    # [] Octonion+ <- Octonion+
    right_first = split_product(Octonion, first)                     # [] Octonion+ <- Octonion+
    right_second = split_product(Octonion, second)                   # [] Octonion+ <- Octonion+
    # Alternating left and right multiplication commutators cancel the non-associator on every product.
    octonion_action = (left_first(left_second) - left_second(left_first)
                       + left_first(right_second) - right_second(left_first)
                       + right_first(right_second) - right_second(right_first))   # [] Octonion+ <- Octonion+
    return octonion_action(State).cast(State)
