"""The fluctuation metric of a van der Waals fluid and its curvature near the critical point.

A state of the fluid is an antivector of the plane spanned by temperature, `mv.t.dual()`, and volume, `mv.v.dual()`.
The metric is a map from a variation of the state to the gradient of half its cost, which pairs with the variation to the variation's
cost: how improbable a fluctuation that large is. It is built from the Helmholtz energy's second derivative in its own
variables, temperature and volume, the thermal part with its sign turned, both over the temperature; near the critical
point the pressure hardly changes with the volume, the mechanical part of the cost vanishes, and fluctuations of the
volume become cheap. The scalar curvature of the metric grows without bound as the critical point is approached.

Temperature and volume are in their critical values, pressure in R T_c / V_c, energy in R T_c.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, stack
from numga.backend.jax import JaxContext, derivative

ga = Algebra("t0v0")
mv = JaxContext(ga, np.float64).multivector

Scalar = ga.gatype.scalar()
State = ga.gatype.antivector()
Gradient = ga.gatype.vector()
Metric = ga.gatype((Gradient, State))

# Regressive pairings read coordinates without assuming a metric.
temperature = mv.t & State                                    # Scalar <- State
volume = mv.v & State                                         # Scalar <- State


# --- math -----------------------------------------------------------------------------
def pressure(state: State, attraction: float, volume_exclusion: float) -> Scalar:
    return temperature(state) / (volume(state) - volume_exclusion) - attraction / volume(state).squared()


def helmholtz_energy(state: State, attraction: float,
                     volume_exclusion: float, heat_capacity: float) -> Scalar:
    warmth, room = temperature(state), volume(state)
    return -heat_capacity * warmth * warmth.log() - warmth * (room - volume_exclusion).log() - attraction / room


def metric(state: State, attraction: float,
           volume_exclusion: float, heat_capacity: float) -> Metric:
    potential = lambda point: helmholtz_energy(point, attraction, volume_exclusion, heat_capacity)
    hessian = derivative(derivative(potential))(state)
    # The Helmholtz energy's second derivative in temperature and volume, its thermal part with the sign turned: in
    # these variables the cost has no cross term between the two.
    warmth = temperature(state)
    thermal = -hessian(mv.t.dual(), mv.t.dual()) / warmth
    mechanical = hessian(mv.v.dual(), mv.v.dual()) / warmth
    return mv.t * thermal * temperature + mv.v * mechanical * volume


def thermodynamic_curvature(state: State, attraction: float,
                            volume_exclusion: float, heat_capacity: float) -> Scalar:
    """Scalar curvature R = 2K, with positive curvature for a sphere."""
    form = lambda point: metric(point, attraction, volume_exclusion, heat_capacity)
    thermal = lambda point: form(point)(mv.t.dual()) & mv.t.dual()
    mechanical = lambda point: form(point)(mv.v.dual()) & mv.v.dual()
    determinant = lambda point: thermal(point) * mechanical(point)
    # For this diagonal metric the thermal coefficient has no volume dependence.
    # Differentiating its volume coefficient suffices for the curvature.
    first = derivative(mechanical)(state)(mv.t.dual())
    second = derivative(derivative(mechanical))(state)(mv.t.dual(), mv.t.dual())
    area_squared = determinant(state)
    area_change = derivative(determinant)(state)(mv.t.dual())
    return -second / area_squared + first * area_change / (2 * area_squared.squared())
