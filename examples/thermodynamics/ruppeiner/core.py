"""Near the critical point of a van der Waals fluid, volume fluctuations grow cheap and the scalar curvature of the
Ruppeiner fluctuation metric diverges.

Two degenerate directions `t0 v0` form an exterior state space in which a fluid state is modelled as an antivector
combining temperature `t.dual()` and volume `v.dual()`. The fluctuation metric maps each state variation to a conjugate
potential `Potential <- State`, whose regressive pairing with the variation gives the fluctuation cost. Maxwell's
relation cancels the cross term between temperature and volume variations, leaving a thermal entry from the heat
capacity and a mechanical entry from the isothermal volume slope of pressure. Approaching the critical point flattens
the isotherms, drives the mechanical entry toward zero, and sends the scalar curvature of the metric to infinity.

For a fixed quantity of fluid, temperature and volume are measured in units of their critical values, energy in units
of the gas constant times the critical temperature, and pressure in that energy per critical volume.
"""
from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("t0v0")
mv = NumpyContext(ga, np.float64).multivector

Scalar = ga.gatype.scalar()                                   # Scalar+
State = ga.gatype.antivector()                                # Antivector-
Potential = ga.gatype.vector()                                # Vector+
Metric = ga.gatype((Potential, State))                        # Potential+ <- State-

# Regressive pairings read coordinates without assuming a metric.
temperature = mv.t & State                                    # [] Scalar+ <- State-
volume = mv.v & State                                         # [] Scalar+ <- State-


# --- math -----------------------------------------------------------------------------
def pressure(state: State, attraction: float, exclusion: float) -> Scalar:
    """The van der Waals equation of state balances thermal pressure in the free volume against molecular attraction."""
    return temperature(state) / (volume(state) - exclusion) - attraction / volume(state).squared()


def metric(state: State, attraction: float, exclusion: float, capacity: float) -> Metric:
    """The Ruppeiner metric maps each state variation to the potential that reads its fluctuation cost."""
    t, v = temperature(state), volume(state)                  # [...] Scalar+, Scalar+
    # Maxwell's relation cancels the cross term between temperature and volume fluctuations.
    thermal = capacity / t.squared()                          # [...] Scalar+
    mechanical = 1 / (v - exclusion).squared() - 2 * attraction / (t * v ** 3)   # [...] Scalar+
    return mv.t * thermal * temperature + mv.v * mechanical * volume             # [...] Potential+ <- State-


def thermodynamic_curvature(state: State, attraction: float, exclusion: float, capacity: float) -> Scalar:
    """Scalar curvature of the fluctuation metric equals twice its Gaussian curvature, which is positive on a sphere."""
    t, v = temperature(state), volume(state)                  # [...] Scalar+, Scalar+
    thermal = capacity / t.squared()                          # [...] Scalar+
    mechanical = 1 / (v - exclusion).squared() - 2 * attraction / (t * v ** 3)   # [...] Scalar+
    area_squared = thermal * mechanical                       # [...] Scalar+
    # Because the metric is diagonal and its thermal entry depends on temperature alone, scalar curvature
    # follows from temperature derivatives of the mechanical entry and of the squared area element.
    slope = 2 * attraction / (t.squared() * v ** 3)                              # [...] Scalar+
    slope_change = -4 * attraction / (t ** 3 * v ** 3)                           # [...] Scalar+
    area_change = -2 * thermal / t * mechanical + thermal * slope                # [...] Scalar+
    return -slope_change / area_squared + slope * area_change / (2 * area_squared.squared())
