"""Cooling, drying, and reheating moist air at fixed pressure on its psychrometric chart.

A state of moist air is modelled as an antivector in the exterior plane `t0 w0` of temperature `t.dual()` and water
content `w.dual()` per kilogram of dry air, with each read back by pairing (`t & state` and `w & state`). Relative
humidity and enthalpy are scalar functions of the state. A cooling coil holds the water content fixed until the air
saturates, condenses out the water beyond saturation while keeping the relative humidity at one, and hands the cold
saturated air to a reheater that warms it at fixed water content. Each held quantity gives a covector constraint whose
pairing `constraint & step` vanishes along an allowed step, and dividing `constraint` by `(t ^ constraint).dual()`
gives the change of state per degree warmer.

Temperature is measured in degrees Celsius, pressure in kPa, water content in kg of water per kg of dry air, and
enthalpy in kJ per kg of dry air. Saturation vapour pressure uses the Magnus approximation over liquid water.
"""
from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("t0w0")
mv = NumpyContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
State = ga.gatype.antivector()                                                # Antivector-
Potential = ga.gatype.vector()                                                # Vector+

# Magnus saturation constants (kPa and degrees Celsius), ratio of molar masses of water and dry air, specific heats of
# dry air, vapour, and liquid water (kJ / (kg K)), and latent heat of vaporisation at zero degrees Celsius (kJ / kg).
magnus_pressure, magnus_slope, magnus_offset = 0.61078, 17.27, 237.3
molar_ratio = 0.62198
dry_air_heat, vapour_heat, liquid_heat, latent_heat = 1.006, 1.86, 4.186, 2501.0


# --- math -----------------------------------------------------------------------------
def saturation_pressure(state: State) -> Scalar:
    """Saturation vapour pressure over liquid water at the state's temperature."""
    warmth = mv.t & state                                                     # [...] Scalar+
    return magnus_pressure * (magnus_slope * warmth / (warmth + magnus_offset)).exp()


def saturation_humidity(state: State, pressure: float) -> Scalar:
    """Maximum water vapour per kilogram of dry air at the state's temperature and total pressure."""
    vapour = saturation_pressure(state)                                       # [...] Scalar+
    return molar_ratio * vapour / (pressure - vapour)


def relative_humidity(state: State, pressure: float) -> Scalar:
    """Ratio of the water vapour's partial pressure to the saturation pressure."""
    water = mv.w & state                                                      # [...] Scalar+
    return pressure * water / ((molar_ratio + water) * saturation_pressure(state))


def relative_humidity_slope(state: State, pressure: float) -> Potential:
    """Covector slope pairing with a step in state to give the change in relative humidity."""
    warmth, water = mv.t & state, mv.w & state                                # [...] Scalar+, Scalar+
    saturation = saturation_pressure(state)                                   # [...] Scalar+
    relative = pressure * water / ((molar_ratio + water) * saturation)        # [...] Scalar+
    d_warmth = -relative * magnus_slope * magnus_offset / (warmth + magnus_offset).squared()
    d_water = pressure * molar_ratio / ((molar_ratio + water).squared() * saturation)
    return d_warmth * mv.t + d_water * mv.w                                   # [...] Potential+


def air_state(warmth: float | np.ndarray, relative: float | np.ndarray, pressure: float) -> State:
    """Moist air states at given temperatures, relative humidities, and total pressure."""
    dry = mv.t.dual() * warmth + mv.w.dual() * 0                              # [...] State-
    vapour = relative * saturation_pressure(dry)                              # [...] Scalar+
    return dry + mv.w.dual() * (molar_ratio * vapour / (pressure - vapour))   # [...] State-


def enthalpy(state: State) -> Scalar:
    """Sensible and latent heat of dry air and water vapour per kilogram of dry air."""
    warmth, water = mv.t & state, mv.w & state                                # [...] Scalar+, Scalar+
    return dry_air_heat * warmth + water * (latent_heat + vapour_heat * warmth)


def dew_point(state: State, pressure: float) -> Scalar:
    """Temperature at which the state's water content reaches saturation."""
    water = mv.w & state                                                      # [...] Scalar+
    logarithm = (pressure * water / ((molar_ratio + water) * magnus_pressure)).log()
    return magnus_offset * logarithm / (magnus_slope - logarithm)


def constrained_direction(constraint: Potential) -> State:
    """Change of state per degree warmer along `constraint & step == 0`."""
    # Dividing `constraint` by the intrinsic scalar `(mv.t ^ constraint).dual()` gives the state step that pairs to
    # zero with `constraint` and to one with `mv.t`.
    return (constraint / (mv.t ^ constraint).dual()).retyped(State)           # [...] State-


def cooling_path(initial: State, outlet_temperature: float, pressure: float, fractions: np.ndarray) -> State:
    """Air states along the cooling coil, keeping water content fixed until saturation and condensing excess vapour
    beyond the dew point."""
    warmth = (mv.t & initial) * (1 - fractions) + outlet_temperature * fractions
    uncondensed = mv.t.dual() * warmth + mv.w.dual() * (mv.w & initial)       # [...] State-
    # Water vapour above saturation condenses out once the air cools below its dew point.
    condensate = ((mv.w & initial) - saturation_humidity(uncondensed, pressure)).clip(0, np.inf)  # [...] Scalar+
    return uncondensed - mv.w.dual() * condensate                             # [...] State-


def reheating_path(initial: State, outlet_temperature: float, fractions: np.ndarray) -> State:
    """Air states along the reheater, warming toward the supply temperature at fixed water content."""
    warmth = (mv.t & initial) * (1 - fractions) + outlet_temperature * fractions
    return mv.t.dual() * warmth + mv.w.dual() * (mv.w & initial)              # [...] State-


def coil_balance(initial: State, outlet: State) -> tuple[Scalar, Scalar]:
    """Heat removed and liquid water condensed per kilogram of dry air when condensate drains at the outlet
    temperature."""
    condensate = (mv.w & initial) - (mv.w & outlet)                           # [...] Scalar+
    cooling = enthalpy(initial) - enthalpy(outlet) - condensate * liquid_heat * (mv.t & outlet)
    return cooling, condensate
