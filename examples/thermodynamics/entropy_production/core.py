"""Entropy production as heat passes between two finite bodies.

A state holds the two bodies' energies, an antivector of the plane spanned by `mv.a.dual()` and `mv.b.dual()`. The
entropy's gradient, each body's received heat over its temperature, `mv.a / temperature_a + mv.b / temperature_b`,
pairs with a change to the entropy it produces. The mobility is a map from gradients to changes of state: a dyad that
sends every gradient into the direction trading one body's energy for the other's, with a positive strength. It sends
the entropy's gradient to the heat flow, and the total energy's gradient, `mv.a + mv.b`, to nothing, so the flow
conserves energy and never lowers the entropy.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext

ga = Algebra("a0b0")
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()                               # Scalar+
State = ga.gatype.antivector()                            # Antivector-
Gradient = ga.gatype.vector()                             # Vector+
Mobility = ga.gatype((State, Gradient))                   # State- <- Gradient+

energy_a = mv.a & State                                   # [] Scalar+ <- State-
energy_b = mv.b & State                                   # [] Scalar+ <- State-
# The total energy's gradient, and the direction trading body a's energy for body b's, which it reads as nothing.
energy_gradient = mv.a + mv.b                             # [] Gradient+
total_energy = energy_gradient & State                    # [] Scalar+ <- State-
transfer = mv.b.dual() - mv.a.dual()                      # [] State-
# The mobility of unit strength: it sends every gradient along the transfer, by the gradient's reading of it.
unit_mobility = transfer * (Gradient & transfer)          # [] State- <- Gradient+


# --- math -----------------------------------------------------------------------------
def temperatures(state: State, capacity_a: float, capacity_b: float) -> tuple[Scalar, Scalar]:
    return energy_a(state) / capacity_a, energy_b(state) / capacity_b


def entropy(state: State, capacity_a: float, capacity_b: float) -> Scalar:
    """Total entropy with temperatures measured relative to one reference temperature."""
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)
    return capacity_a * temperature_a.log() + capacity_b * temperature_b.log()


def entropy_gradient(state: State, capacity_a: float, capacity_b: float) -> Gradient:
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)
    # Each body's entropy increases by its received heat divided by its temperature.
    return mv.a / temperature_a + mv.b / temperature_b


def mobility(state: State, capacity_a: float, capacity_b: float, conductance: float) -> Mobility:
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)
    # Every response follows the energy-conserving direction, with positive strength.
    return unit_mobility * (conductance * temperature_a * temperature_b)


def heat_flow(state: State, capacity_a: float, capacity_b: float, conductance: float) -> State:
    return mobility(state, capacity_a, capacity_b, conductance)(entropy_gradient(state, capacity_a, capacity_b))


def advance(state: State, capacity_a: float, capacity_b: float, conductance: float, dt: float) -> State:
    """The state after a nonnegative time interval, the heat flow integrated exactly over it."""
    inverse_capacity = 1 / capacity_a + 1 / capacity_b
    # The temperature difference decays exponentially, giving the integrated transfer strength.
    integrated_conductance = -np.expm1(-conductance * inverse_capacity * dt) / inverse_capacity
    integrated_mobility = mobility(state, capacity_a, capacity_b, integrated_conductance)
    return state + integrated_mobility(entropy_gradient(state, capacity_a, capacity_b))
