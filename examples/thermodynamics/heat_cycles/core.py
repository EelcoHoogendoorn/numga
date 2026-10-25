"""A Carnot engine: a gas taken around a cycle between a hot and a cold reservoir turns part of the heat it takes in into
work.

A state of the gas is an antivector of the plane spanned by entropy and volume, and the energy's derivative is the
reading of changes of state that gives temperature and minus pressure. Temperature times the entropy's reading is the
heat a change takes in, pressure times the volume's reading the work it does; their difference is the energy's
derivative, while neither alone is the derivative of anything, so heat and work depend on the path and their
difference does not. The antisymmetrised derivatives of the two readings are forms on pairs of changes, reading
oriented areas: the heat circulated around an area is the temperature's change wedged with the entropy's, the work the
pressure's change wedged with the volume's, and the two are one form. Around a closed cycle the energy returns, and
the enclosed areas on the temperature–entropy and pressure–volume charts, the net heat and the net work, are equal.

Entropy is in nR, volume and temperature in reference values, energy in nRT₀; work is positive when the gas does it.
"""
from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext, derivative

ga = Algebra("s0v0")
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()
State = ga.gatype.antivector()
Reading = ga.gatype((Scalar, State))
AreaReading = ga.gatype((Scalar, State, State))

entropy: Reading = mv.s & State                            # Scalar <- State
volume: Reading = mv.v & State                             # Scalar <- State


# --- math -----------------------------------------------------------------------------
def energy(state: State, capacity: float) -> Scalar:
    """The energy of an ideal gas with a constant heat capacity at fixed volume."""
    return capacity * ((entropy(state) - volume(state).log()) / capacity).exp()


def lifted(entropies: Scalar, temperatures: Scalar, capacity: float) -> State:
    """The gas's states at entropies and temperatures: the volume where the energy's slope along the entropy is the
    temperature."""
    return mv.s.dual() * entropies + mv.v.dual() * (entropies - capacity * temperatures.log()).exp()


def strokes(fraction: Scalar, corners_entropy: Scalar, corners_temperature: Scalar, capacity: float) -> State:
    """The states a fraction of the way along each stroke between successive corners, straight in entropy and
    temperature: at the reservoirs' temperatures the gas takes in or gives up heat, between them it is insulated."""
    entropies = corners_entropy[:-1] + (corners_entropy[1:] - corners_entropy[:-1]) * fraction
    temperatures = corners_temperature[:-1] + (corners_temperature[1:] - corners_temperature[:-1]) * fraction
    return lifted(entropies, temperatures, capacity)


def exchanges(states: State, tangents: State, capacity: float) -> tuple[Scalar, Scalar]:
    """The heat a change of state along each tangent takes in and the work it does."""
    differential = derivative(lambda state: energy(state, capacity))(states)        # [...] Scalar <- State
    heat = differential(mv.s.dual()) * entropy                                     # [...] Scalar <- State
    work = -differential(mv.v.dual()) * volume                                     # [...] Scalar <- State
    return heat(tangents), work(tangents)


def circulations(state: State, capacity: float) -> tuple[AreaReading, AreaReading]:
    """The heat and the work circulated around small oriented areas of states, as forms on the pairs of changes that
    span them."""
    differential = derivative(lambda point: energy(point, capacity))
    temperature_change = derivative(lambda value: differential(value)(mv.s.dual()))(state)   # [...] Scalar <- State
    pressure_change = derivative(lambda value: -differential(value)(mv.v.dual()))(state)     # [...] Scalar <- State
    heat_area = temperature_change * entropy - entropy * temperature_change        # [...] Scalar <- (State, State)
    work_area = pressure_change * volume - volume * pressure_change                # [...] Scalar <- (State, State)
    return heat_area, work_area
