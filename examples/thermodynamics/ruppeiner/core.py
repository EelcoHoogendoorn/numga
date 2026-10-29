"""The fluctuation metric of a van der Waals fluid and its curvature near the critical point.

A state of the fluid is modelled as an antivector of the plane spanned by temperature, `mv.t.dual()`, and volume,
`mv.v.dual()`. The metric is a map from a variation of the state to the gradient of half its cost, which pairs with the
variation to the variation's cost: how improbable a fluctuation that large is. It is built from the Helmholtz energy's
second derivative in its own variables, temperature and volume, the thermal part with its sign turned, both over the
temperature; near the critical point the pressure hardly changes with the volume, the mechanical part of the cost
vanishes, and fluctuations of the volume become cheap. The scalar curvature of the metric grows without bound as the
critical point is approached.

Temperature and volume are in their critical values, pressure in R T_c / V_c, energy in R T_c.
"""

from __future__ import annotations

import jax
import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext, derivative

ga = Algebra("t0v0")
mv = JaxContext(ga, np.float64).multivector

Scalar = ga.gatype.scalar()                                   # Scalar+
State = ga.gatype.antivector()                                # Antivector-
Gradient = ga.gatype.vector()                                 # Vector+
Metric = ga.gatype((Gradient, State))                         # Gradient+ <- State-

# Regressive pairings read coordinates without assuming a metric.
temperature = mv.t & State                                    # [] Scalar+ <- State-
volume = mv.v & State                                         # [] Scalar+ <- State-


# --- math -----------------------------------------------------------------------------
@jax.jit
def pressure(state: State, attraction: float, volume_exclusion: float) -> Scalar:
    """The van der Waals equation of state."""
    return temperature(state) / (volume(state) - volume_exclusion) - attraction / volume(state).squared()


def helmholtz_energy(state: State, attraction: float,
                     volume_exclusion: float, heat_capacity: float) -> Scalar:
    t, v = temperature(state), volume(state)                  # [...] Scalar+, Scalar+
    return -heat_capacity * t * t.log() - t * (v - volume_exclusion).log() - attraction / v


@jax.jit
def metric(state: State, attraction: float,
           volume_exclusion: float, heat_capacity: float) -> Metric:
    """The cost of variations about a state, as a map from a variation to a gradient."""
    potential = lambda point: helmholtz_energy(point, attraction, volume_exclusion, heat_capacity)
    hessian = derivative(derivative(potential))(state)        # [...] Scalar+ <- (State-, State-)
    # The Helmholtz energy's second derivative in temperature and volume, its thermal part with the sign turned: in
    # these variables the cost has no cross term between the two.
    thermal = -hessian(mv.t.dual(), mv.t.dual()) / temperature(state)        # [...] Scalar+
    mechanical = hessian(mv.v.dual(), mv.v.dual()) / temperature(state)      # [...] Scalar+
    return mv.t * thermal * temperature + mv.v * mechanical * volume


@jax.jit
def thermodynamic_curvature(state: State, attraction: float,
                            volume_exclusion: float, heat_capacity: float) -> Scalar:
    """The metric's scalar curvature, twice its Gaussian curvature, positive on a sphere."""
    form = lambda point: metric(point, attraction, volume_exclusion, heat_capacity)
    thermal = lambda point: form(point)(mv.t.dual()) & mv.t.dual()
    mechanical = lambda point: form(point)(mv.v.dual()) & mv.v.dual()
    area_squared = lambda point: thermal(point) * mechanical(point)
    # The metric has no cross term and its thermal entry does not change with the volume: the curvature needs only the
    # mechanical entry's changes along the temperature and the change of the squared area the two entries span.
    slope = derivative(mechanical)(state)(mv.t.dual())                                    # [...] Scalar+
    slope_change = derivative(derivative(mechanical))(state)(mv.t.dual(), mv.t.dual())   # [...] Scalar+
    area_change = derivative(area_squared)(state)(mv.t.dual())                           # [...] Scalar+
    return -slope_change / area_squared(state) + slope * area_change / (2 * area_squared(state).squared())
