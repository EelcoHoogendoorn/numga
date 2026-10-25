"""Entropy production as heat passes between two finite bodies.

A state holds the two bodies' energies, an antivector of the plane spanned by `mv.a.dual()` and `mv.b.dual()`. The
entropy's covector, each body's received heat over its temperature, `mv.a / temperature_a + mv.b / temperature_b`,
pairs with a change to the entropy it produces. The mobility is a map from covectors to changes of state: a dyad that
sends every covector into the direction trading one body's energy for the other's, with a positive strength. It sends
the entropy's covector to the heat flow, and the total energy's covector, `mv.a + mv.b`, to nothing, so the flow
conserves energy and never lowers the entropy.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext

ga = Algebra("a0b0")
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()
State = ga.gatype.antivector()
Covector = ga.gatype.vector()
Mobility = ga.gatype((State, Covector))

energy_a = mv.a & State                                   # Scalar <- State
energy_b = mv.b & State                                   # Scalar <- State
# The total energy's covector, and the direction trading body a's energy for body b's, which it reads as nothing.
energy_covector = mv.a + mv.b                             # [] Covector
total_energy = energy_covector & State                    # Scalar <- State
transfer = mv.b.dual() - mv.a.dual()                      # [] State


# --- math -----------------------------------------------------------------------------
def temperatures(state: State, capacity_a: float, capacity_b: float) -> tuple[Scalar, Scalar]:
    return energy_a(state) / capacity_a, energy_b(state) / capacity_b


def entropy(state: State, capacity_a: float, capacity_b: float) -> Scalar:
    """Total entropy with temperatures measured relative to one reference temperature."""
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)
    return capacity_a * temperature_a.log() + capacity_b * temperature_b.log()


def entropy_covector(state: State, capacity_a: float, capacity_b: float) -> Covector:
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)
    # Each body's entropy increases by its received heat divided by its temperature.
    return mv.a / temperature_a + mv.b / temperature_b


def mobility(state: State, capacity_a: float, capacity_b: float, conductance: float) -> Mobility:
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)
    # Every response follows the energy-conserving direction, with positive strength.
    return transfer * (Covector & transfer) * conductance * temperature_a * temperature_b


def heat_flow(state: State, capacity_a: float, capacity_b: float, conductance: float) -> State:
    return mobility(state, capacity_a, capacity_b, conductance)(entropy_covector(state, capacity_a, capacity_b))


def advance(state: State, capacity_a: float, capacity_b: float, conductance: float, dt: float) -> State:
    """Integrate the heat flow exactly over a nonnegative time interval."""
    inverse_capacity = 1 / capacity_a + 1 / capacity_b
    # The temperature difference decays exponentially, giving the integrated transfer strength.
    integrated_conductance = -np.expm1(-conductance * inverse_capacity * dt) / inverse_capacity
    integrated_mobility = mobility(state, capacity_a, capacity_b, integrated_conductance)
    return state + integrated_mobility(entropy_covector(state, capacity_a, capacity_b))
