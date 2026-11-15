"""Two insulated bodies at different temperatures exchange heat until their temperatures equalize.

The algebra `a0 b0` has one degenerate quantity direction for each body's internal energy. A state is modelled as an
antivector `energy_a * a.dual() + energy_b * b.dual()`, and a thermodynamic potential is a vector that pairs with a
change of state. The entropic potential `a / temperature_a + b / temperature_b` pairs with a change of state to give
the entropy produced, while `a + b` reads the change in total energy. The mobility maps potentials to rates of state
change through a positive dyad along `transfer = b.dual() - a.dual()`, the step that moves a unit of energy from body
`a` to body `b`. Because `a + b` pairs to zero with `transfer`, the heat flow driven by the entropic potential
conserves total energy and produces nonnegative entropy.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("a0b0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                               # Scalar+
State = ga.gatype.antivector()                            # Antivector-
Potential = ga.gatype.vector()                            # Vector+
Mobility = ga.gatype((State, Potential))                  # State- <- Potential+

energy_a = mv.a & State                                   # [] Scalar+ <- State-
energy_b = mv.b & State                                   # [] Scalar+ <- State-
# Total-energy potential and the energy-conserving unit transfer from body a to body b, which pair to zero.
energy_potential = mv.a + mv.b                            # [] Potential+
transfer = mv.b.dual() - mv.a.dual()                      # [] State-
# Unit mobility dyad sending any potential along transfer in proportion to how that potential reads transfer.
unit_mobility = transfer * (Potential & transfer)         # [] State- <- Potential+


# --- math -----------------------------------------------------------------------------
def temperatures(state: State, capacity_a: float, capacity_b: float) -> tuple[Scalar, Scalar]:
    """Absolute temperatures of the two bodies from their internal energies and heat capacities."""
    return energy_a(state) / capacity_a, energy_b(state) / capacity_b


def entropy(state: State, capacity_a: float, capacity_b: float) -> Scalar:
    """Total entropy of the two bodies relative to a unit reference temperature."""
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)    # [...] Scalar+, Scalar+
    return capacity_a * temperature_a.log() + capacity_b * temperature_b.log()


def entropy_potential(state: State, capacity_a: float, capacity_b: float) -> Potential:
    """Thermodynamic potential that pairs with a change of state to give the entropy produced."""
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)    # [...] Scalar+, Scalar+
    # Each body's entropy rises by the heat it receives divided by its temperature.
    return mv.a / temperature_a + mv.b / temperature_b


def mobility(state: State, capacity_a: float, capacity_b: float, conductance: float) -> Mobility:
    """Positive dyad mapping potentials to energy-conserving rates of state change."""
    temperature_a, temperature_b = temperatures(state, capacity_a, capacity_b)    # [...] Scalar+, Scalar+
    # Scaling by both temperatures makes the heat flow proportional to the temperature difference.
    return unit_mobility * (conductance * temperature_a * temperature_b)


def advance(state: State, capacity_a: float, capacity_b: float, conductance: float, dt: float) -> State:
    """State after integrating the heat exchange analytically over time step `dt`."""
    inverse_capacity = 1 / capacity_a + 1 / capacity_b
    # Exponential decay of the temperature gap integrates to one mobility step with an effective conductance.
    integrated_conductance = -np.expm1(-conductance * inverse_capacity * dt) / inverse_capacity
    integrated_mobility = mobility(state, capacity_a, capacity_b, integrated_conductance)   # [...] State- <- Potential+
    return state + integrated_mobility(entropy_potential(state, capacity_a, capacity_b))
