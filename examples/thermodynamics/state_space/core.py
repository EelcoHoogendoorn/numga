"""Equal heat warms a gas held at fixed volume more than one held at fixed pressure.

A state of the gas is modelled as an antivector in the exterior plane `s0 v0` of entropy `s.dual()` and volume
`v.dual()`, with each extensive variable read by pairing (`s & state` and `v & state`). An ideal gas's internal energy
sets its conjugate potentials, temperature along `s` and minus pressure along `v`, together with the covector slopes
that pair with a step in state through `&` to read how temperature and pressure change. A constraint is another
covector whose pairing with an allowed change of state vanishes. Dividing `constraint` by `(temperature_slope ^
constraint).dual()` gives the heating direction per unit rise in temperature, and each step takes in heat equal to the
temperature times its entropy change.

Entropy is measured in units of nR, volume and temperature in units of their initial values, and energy in units of
nRT₀.
"""
from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("s0v0")
mv = NumpyContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
State = ga.gatype.antivector()                                                # Antivector-
Potential = ga.gatype.vector()                                                # Vector+


# --- math -----------------------------------------------------------------------------
def energy(state: State, thermal_capacity: float) -> Scalar:
    """Internal energy of an ideal gas with a constant heat capacity at fixed volume."""
    entropies = mv.s & state                                                  # [...] Scalar+
    volumes = mv.v & state                                                    # [...] Scalar+
    return thermal_capacity * ((entropies - volumes.log()) / thermal_capacity).exp()


def response(state: State, thermal_capacity: float) -> tuple[Scalar, Scalar, Potential, Potential]:
    """Temperature, pressure, and the potential slopes pairing with a step in state to give their changes."""
    volumes = mv.v & state                                                    # [...] Scalar+
    temperatures = energy(state, thermal_capacity) / thermal_capacity         # [...] Scalar+
    pressures = temperatures / volumes                                        # [...] Scalar+
    temperature_slope = (temperatures / thermal_capacity) * (mv.s - mv.v / volumes)                # [...] Potential+
    pressure_slope = ((pressures / thermal_capacity) * mv.s
                      - (pressures * (1 + 1 / thermal_capacity) / volumes) * mv.v)                 # [...] Potential+
    return temperatures, pressures, temperature_slope, pressure_slope


def heating_direction(state: State, constraint: Potential, thermal_capacity: float) -> tuple[State, Scalar]:
    """Change of state per unit rise in temperature along `constraint & step == 0`, and the heat capacity along it."""
    temperatures, _, temperature_slope, _ = response(state, thermal_capacity)
    # Dividing the constraint covector by the intrinsic scalar `(temperature_slope ^ constraint).dual()` gives the
    # intrinsic state step that pairs to zero with the constraint and to one with the temperature slope.
    direction = constraint / (temperature_slope ^ constraint).dual()          # [...] State-
    return direction, temperatures * (mv.s & direction)


def constraints(states: State, thermal_capacity: float) -> Potential:
    """Covectors paired to zero along each cylinder's path, fixing volume on the latched cylinder and pressure on the
    loaded one."""
    _, _, _, pressure_slope = response(states, thermal_capacity)
    return stack((mv.v, pressure_slope[1]))                                   # [2] Potential+


def heating_step(states: State, heat: float, thermal_capacity: float) -> State:
    """Cylinder states after adding a slice of heat along their constrained heating directions by the midpoint rule."""
    direction, capacity = heating_direction(states, constraints(states, thermal_capacity), thermal_capacity)
    halfway = states + direction * (heat / capacity / 2)                      # [2] State-
    direction, capacity = heating_direction(halfway, constraints(halfway, thermal_capacity), thermal_capacity)
    return states + direction * (heat / capacity)                             # [2] State-
