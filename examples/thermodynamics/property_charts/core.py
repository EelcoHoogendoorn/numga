"""Cooling, drying and reheating moist air at fixed pressure, on its psychrometric chart.

A state of the air is an antivector of the plane spanned by its temperature, `mv.t.dual()`, and its water content per
kilogram of dry air, `mv.w.dual()`, each read back by pairing, `mv.w & state`. Relative humidity and enthalpy are two
more readings of the same state, functions of it. A cooling coil first holds the water content; once the air is
saturated, it holds the relative humidity at one, the water beyond that condensing out; the reheater holds the water
content again. A held reading is a constraint on the change of state, and stacked with the temperature's change it is
a map that, solved for one degree warmer and the held reading unchanged, gives the direction the air moves in.

Temperature is in degrees Celsius, pressure in kPa, water content in kg water per kg dry air, and enthalpy in kJ per kg
dry air. Saturation uses the Magnus approximation over liquid water.
"""
from __future__ import annotations

import jax
import numpy as np

from numga import Algebra, stack
from numga.backend.jax import JaxContext, derivative

jax.config.update('jax_enable_x64', True)
ga = Algebra('t0w0')
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()
State = ga.gatype.antivector()
Reading = ga.gatype((Scalar, State))
temperature: Reading = mv.t & State                       # Scalar <- State
humidity: Reading = mv.w & State                          # Scalar <- State

# The Magnus approximation of the saturation pressure, in kPa, over a temperature in degrees Celsius; the ratio of the
# molar masses of water and dry air; the specific heats of dry air, of vapour and of liquid water, and the latent heat
# of vaporisation at zero degrees, in kJ per kg.
MAGNUS_PRESSURE, MAGNUS_SLOPE, MAGNUS_OFFSET = 0.61078, 17.27, 237.3
MOLAR_RATIO = 0.62198
DRY_AIR_HEAT, VAPOUR_HEAT, LIQUID_HEAT, LATENT_HEAT = 1.006, 1.86, 4.186, 2501.0


# --- math -----------------------------------------------------------------------------
def saturation_pressure(state: State) -> Scalar:
    warmth = temperature(state)
    return MAGNUS_PRESSURE * (MAGNUS_SLOPE * warmth / (warmth + MAGNUS_OFFSET)).exp()


def saturation_humidity(state: State, pressure: float) -> Scalar:
    """The most water the air holds as vapour at its temperature."""
    vapour = saturation_pressure(state)
    return MOLAR_RATIO * vapour / (pressure - vapour)


def relative_humidity(state: State, pressure: float) -> Scalar:
    """The vapour's partial pressure over the saturation pressure."""
    water = humidity(state)
    return pressure * water / (MOLAR_RATIO + water) / saturation_pressure(state)


def air_state(warmth: np.ndarray, relative: np.ndarray, pressure: float) -> State:
    """Air at temperatures and relative humidities."""
    dry = mv.t.dual() * warmth + mv.w.dual() * 0
    vapour = relative * saturation_pressure(dry)
    return dry + mv.w.dual() * (MOLAR_RATIO * vapour / (pressure - vapour))


def enthalpy(state: State) -> Scalar:
    """The dry air's heat and the vapour's, latent and sensible, per kg dry air."""
    warmth, water = temperature(state), humidity(state)
    return DRY_AIR_HEAT * warmth + water * (LATENT_HEAT + VAPOUR_HEAT * warmth)


def dew_point(state: State, pressure: float) -> Scalar:
    """The temperature at which the air's water content saturates it."""
    water = humidity(state)
    logarithm = (pressure * water / ((MOLAR_RATIO + water) * MAGNUS_PRESSURE)).log()
    return MAGNUS_OFFSET * logarithm / (MAGNUS_SLOPE - logarithm)


def cooling_path(initial: State, outlet_temperature: float, pressure: float, fraction: np.ndarray) -> State:
    """The air cooled a fraction of the way to the coil's outlet: its water content held until it saturates, then
    the water beyond saturation condensed out."""
    warmth = temperature(initial) * (1 - fraction) + outlet_temperature * fraction
    dry = mv.t.dual() * warmth + mv.w.dual() * humidity(initial)
    condensate = (humidity(initial) - saturation_humidity(dry, pressure)).clip(0, np.inf)
    return dry - mv.w.dual() * condensate


def reheating_path(initial: State, outlet_temperature: float, fraction: np.ndarray) -> State:
    """The air reheated a fraction of the way to the supply temperature, its water content held."""
    warmth = temperature(initial) * (1 - fraction) + outlet_temperature * fraction
    return mv.t.dual() * warmth + mv.w.dual() * humidity(initial)


def coil_balance(initial: State, outlet: State) -> tuple[Scalar, Scalar]:
    """The heat the coil removes and the water it collects per kg dry air, the water leaving at the outlet's
    temperature."""
    condensate = humidity(initial) - humidity(outlet)
    cooling = enthalpy(initial) - enthalpy(outlet) - condensate * LIQUID_HEAT * temperature(outlet)
    return cooling, condensate


def constrained_direction(state: State, constraint: Reading) -> State:
    """The change of state per degree warmer with the constraint's reading held."""
    readings = stack((temperature.broadcast_to(state.shape), constraint), axis=-1).field(0)
    requested = mv.scalar([[1], [0]]).field(0)
    return readings.solve(requested)
