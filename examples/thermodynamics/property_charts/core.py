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

import numpy as np

from numga import Algebra, stack
from numga.backend.jax import JaxContext, derivative

ga = Algebra('t0w0')
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()                               # Scalar+
State = ga.gatype.antivector()                            # Antivector-
Reading = ga.gatype((Scalar, State))                      # Scalar+ <- State-

temperature = mv.t & State                                # [] Scalar+ <- State-
humidity = mv.w & State                                   # [] Scalar+ <- State-

# The Magnus approximation of the saturation pressure, in kPa, over a temperature in degrees Celsius; the ratio of the
# molar masses of water and dry air; the specific heats of dry air, of vapour and of liquid water, and the latent heat
# of vaporisation at zero degrees, in kJ per kg.
magnus_pressure, magnus_slope, magnus_offset = 0.61078, 17.27, 237.3
molar_ratio = 0.62198
dry_air_heat, vapour_heat, liquid_heat, latent_heat = 1.006, 1.86, 4.186, 2501.0


# --- math -----------------------------------------------------------------------------
def saturation_pressure(state: State) -> Scalar:
    warmth = temperature(state)
    return magnus_pressure * (magnus_slope * warmth / (warmth + magnus_offset)).exp()


def saturation_humidity(state: State, pressure: float) -> Scalar:
    """The most water the air holds as vapour at its temperature."""
    vapour = saturation_pressure(state)
    return molar_ratio * vapour / (pressure - vapour)


def relative_humidity(state: State, pressure: float) -> Scalar:
    """The vapour's partial pressure over the saturation pressure."""
    water = humidity(state)
    return pressure * water / (molar_ratio + water) / saturation_pressure(state)


def air_state(warmth: np.ndarray, relative: np.ndarray, pressure: float) -> State:
    """Air at temperatures and relative humidities."""
    dry = mv.t.dual() * warmth + mv.w.dual() * 0
    vapour = relative * saturation_pressure(dry)
    return dry + mv.w.dual() * (molar_ratio * vapour / (pressure - vapour))


def enthalpy(state: State) -> Scalar:
    """The dry air's heat and the vapour's, latent and sensible, per kg dry air."""
    warmth, water = temperature(state), humidity(state)
    return dry_air_heat * warmth + water * (latent_heat + vapour_heat * warmth)


def dew_point(state: State, pressure: float) -> Scalar:
    """The temperature at which the air's water content saturates it."""
    water = humidity(state)
    logarithm = (pressure * water / ((molar_ratio + water) * magnus_pressure)).log()
    return magnus_offset * logarithm / (magnus_slope - logarithm)


def cooling_path(initial: State, outlet_temperature: float, pressure: float, fractions: np.ndarray) -> State:
    """The air cooled a fraction of the way to the coil's outlet: its water content held until it saturates, then
    the water beyond saturation condensed out."""
    warmth = temperature(initial) * (1 - fractions) + outlet_temperature * fractions
    uncondensed = mv.t.dual() * warmth + mv.w.dual() * humidity(initial)                # [...] State-
    # The water beyond what saturated air holds at that temperature, none until the air reaches its dew point.
    condensate = (humidity(initial) - saturation_humidity(uncondensed, pressure)).clip(0, np.inf)  # [...] Scalar+
    return uncondensed - mv.w.dual() * condensate


def reheating_path(initial: State, outlet_temperature: float, fractions: np.ndarray) -> State:
    """The air reheated a fraction of the way to the supply temperature, its water content held."""
    warmth = temperature(initial) * (1 - fractions) + outlet_temperature * fractions
    return mv.t.dual() * warmth + mv.w.dual() * humidity(initial)


def coil_balance(initial: State, outlet: State) -> tuple[Scalar, Scalar]:
    """The heat the coil removes and the water it collects per kg dry air, the water leaving at the outlet's
    temperature."""
    condensate = humidity(initial) - humidity(outlet)
    cooling = enthalpy(initial) - enthalpy(outlet) - condensate * liquid_heat * temperature(outlet)
    return cooling, condensate


def constrained_direction(state: State, constraint: Reading) -> State:
    """The change of state per degree warmer with the constraint's reading held."""
    # The two readings of one unknown change: the temperature's rises by one, the constraint's stays put.
    readings = stack((temperature.broadcast_to(state.shape), constraint),
                     axis=-1).field(0)                                     # [...] (Scalar+ <- State-)[readings]
    requested = mv.scalar([[1], [0]]).field(0)                             # Scalar+[readings]
    return readings.solve(requested)
