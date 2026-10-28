"""A mass on a spring with a damper that warms a thermal store: reversible motion and irreversible friction in one
state, as a GENERIC system.

A state holds the mass's position and momentum and the store's entropy, an antivector of the space spanned by
`mv.q.dual()`, `mv.p.dual()` and `mv.s.dual()`. The energy's gradient, the spring's force, the mass's velocity and the
store's temperature, pairs with a change of state to the change of energy. Two maps from gradients to changes of state
drive the motion. The Poisson map, skew, sends the energy's gradient to the reversible motion, the swing of the
oscillator, and the entropy's gradient, `mv.s`, to nothing. The friction mobility, a positive dyad along the direction
trading momentum for the store's heat, sends the entropy's gradient to the irreversible motion and the energy's
gradient to nothing. So the energy holds and the entropy never falls, and the oscillation settles with all its energy
in the store.
"""
from __future__ import annotations

import jax
import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext

ga = Algebra("q0p0s0")
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()                               # Scalar+
State = ga.gatype.antivector()                            # Antivector-
Gradient = ga.gatype.vector()                             # Vector+
Map = ga.gatype((State, Gradient))                        # State- <- Gradient+

# The canonical Poisson map: the change of position follows the momentum's slope, the change of momentum minus the
# position's.
poisson_map = (mv.q.dual() * (Gradient & mv.p.dual())
               - mv.p.dual() * (Gradient & mv.q.dual()))  # [] State- <- Gradient+
# The entropy's gradient: the store's entropy, the one reading of a state it is.
entropy_gradient = mv.s                                   # [] Gradient+


# --- math -----------------------------------------------------------------------------
def temperature(state: State, thermal_capacity: float, reference_temperature: float) -> Scalar:
    return reference_temperature * ((mv.s & state) / thermal_capacity).exp()


def kinetic_energy(state: State, mass: float) -> Scalar:
    return (mv.p & state).squared() / (2 * mass)


def potential_energy(state: State, stiffness: float) -> Scalar:
    return stiffness * (mv.q & state).squared() / 2


def internal_energy(state: State, thermal_capacity: float, reference_temperature: float) -> Scalar:
    return thermal_capacity * temperature(state, thermal_capacity, reference_temperature)


def total_energy(state: State, mass: float, stiffness: float, thermal_capacity: float,
                 reference_temperature: float) -> Scalar:
    return (kinetic_energy(state, mass) + potential_energy(state, stiffness)
            + internal_energy(state, thermal_capacity, reference_temperature))


def energy_gradient(state: State, mass: float, stiffness: float, thermal_capacity: float,
                    reference_temperature: float) -> Gradient:
    """The energy's slopes along position, momentum and entropy: the spring's force, the velocity, the temperature."""
    return (mv.q * (stiffness * (mv.q & state)) + mv.p * ((mv.p & state) / mass)
            + mv.s * temperature(state, thermal_capacity, reference_temperature))


def friction_mobility(state: State, mass: float, friction: float, thermal_capacity: float,
                      reference_temperature: float) -> Map:
    """A positive dyad along the direction in which the momentum lost is the store's heat gained."""
    store_temperature = temperature(state, thermal_capacity, reference_temperature)
    direction = -mv.p.dual() + mv.s.dual() * ((mv.p & state) / mass / store_temperature)
    return direction * (Gradient & direction) * (friction * store_temperature)


def rates(state: State, mass: float, stiffness: float, friction: float, thermal_capacity: float,
          reference_temperature: float) -> tuple[State, State]:
    """The reversible motion, the Poisson map of the energy's gradient, and the irreversible, the friction mobility of
    the entropy's."""
    reversible = poisson_map(energy_gradient(state, mass, stiffness, thermal_capacity, reference_temperature))
    irreversible = friction_mobility(state, mass, friction, thermal_capacity, reference_temperature)(entropy_gradient)
    return reversible, irreversible


@jax.jit
def advance(state: State, mass: float, stiffness: float, friction: float, thermal_capacity: float,
            reference_temperature: float, dt: float) -> State:
    """A step of the combined motion by the classical fourth-order Runge–Kutta method."""
    def rate(current: State) -> State:
        reversible, irreversible = rates(current, mass, stiffness, friction, thermal_capacity, reference_temperature)
        return reversible + irreversible
    k1 = rate(state)
    k2 = rate(state + k1 * (dt / 2))
    k3 = rate(state + k2 * (dt / 2))
    k4 = rate(state + k3 * dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
