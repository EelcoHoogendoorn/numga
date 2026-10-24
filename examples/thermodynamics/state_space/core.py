"""Heating a gas under a constraint: equal heat warms a gas held at fixed volume more than one held at fixed pressure.

A state of the gas is an antivector of the plane spanned by entropy, `mv.s.dual()`, and volume, `mv.v.dual()`, and
quantities are read from it by pairing, `mv.s & state`. The gas is modelled by its energy as a function of the state;
the energy's derivative is the reading of a change of state that gives temperature and minus pressure, and its second
derivative how those two change. A constraint is another reading of the change of state, one that must stay zero.
Stacked with the temperature's change, the two readings are one map from a change of state to the changes of both,
and solving it for one degree warmer and no change of the constrained reading gives the heating direction; the heat
a change takes in is the temperature times its change of entropy.

Entropy is in nR, volume in its initial value, temperature in its initial value and energy in nRT₀.
"""
from __future__ import annotations

from functools import partial

import jax
import numpy as np

from numga import Algebra, stack
from numga.backend.jax import JaxContext, derivative

jax.config.update("jax_enable_x64", True)
ga = Algebra("s0v0")
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()
State = ga.gatype.antivector()
Reading = ga.gatype((Scalar, State))
Hessian = ga.gatype((Scalar, State, State))

entropy: Reading = mv.s & State                            # Scalar <- State
volume: Reading = mv.v & State                             # Scalar <- State


# --- math -----------------------------------------------------------------------------
def energy(state: State, thermal_capacity: float) -> Scalar:
    """The energy of an ideal gas with a constant heat capacity at fixed volume."""
    return thermal_capacity * ((entropy(state) - volume(state).log()) / thermal_capacity).exp()


def response(state: State, thermal_capacity: float) -> tuple[Scalar, Scalar, Reading, Reading]:
    """Temperature and pressure, and the readings of how they change, all differentiated from the energy."""
    potential = partial(energy, thermal_capacity=thermal_capacity)
    differential: Reading = derivative(potential)(state)              # [...] Scalar <- State
    hessian: Hessian = derivative(derivative(potential))(state)       # [...] Scalar <- (State, State)
    return differential(mv.s.dual()), -differential(mv.v.dual()), hessian(mv.s.dual()), -hessian(mv.v.dual())


def heating_direction(state: State, constraint: Reading, thermal_capacity: float) -> tuple[State, Scalar]:
    """The change of state per degree warmer with the constraint's reading held, and the heat it takes in."""
    temperature, _, temperature_change, _ = response(state, thermal_capacity)
    # The two readings of one unknown change: the temperature's rises by one, the constraint's stays put.
    changes = stack((temperature_change.broadcast_to(constraint.shape), constraint), axis=-1).field(0)
    requested = mv.scalar([[1], [0]]).field(0)                         # Scalar[readings]
    direction: State = changes.solve(requested)                        # [...] State
    return direction, temperature * entropy(direction)


def held(states: State, thermal_capacity: float) -> Reading:
    """What each of the two cylinders holds `[2]`: the latched one its volume, the loaded one its pressure."""
    _, _, _, pressure_change = response(states, thermal_capacity)
    return stack((volume, pressure_change[1]))


def heated(states: State, heat: float, thermal_capacity: float) -> State:
    """The cylinders' states after taking in a little heat each, along their heating directions, by the midpoint rule."""
    direction, capacity = heating_direction(states, held(states, thermal_capacity), thermal_capacity)
    halfway = states + direction * (heat / capacity / 2)
    direction, capacity = heating_direction(halfway, held(halfway, thermal_capacity), thermal_capacity)
    return states + direction * (heat / capacity)
