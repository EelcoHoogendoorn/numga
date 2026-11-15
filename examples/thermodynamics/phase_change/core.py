"""Below the critical temperature, a van der Waals fluid separates into coexisting liquid and vapour phases and boils
at fixed temperature and pressure.

Two degenerate quantity directions `l0 g0` generate an exterior state space in which a coexistence state is modelled as
an antivector combining the liquid and vapour volumes. The two phases coexist where their pressures and chemical
potentials match, so the tangent lines to the Helmholtz free energy at the two volumes form one common tangent.
Evaluating the pressure and chemical-potential differences maps each trial state to a residual in the same state space.
Because a phase's chemical potential changes with volume at fixed temperature as volume times its pressure slope, the
derivative of that residual is a two-dyad linear map on state variations that Newton's method inverts across a batch of
temperatures. Along the common tangent, a two-phase mixture divides its volume and enthalpy linearly by the vapour
fraction.

For a fixed quantity of fluid, temperature and volume are measured in units of their critical values, energy in units
of the gas constant times the critical temperature, and pressure in that energy per critical volume.
"""
from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("l0g0")
mv = NumpyContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
State = ga.gatype.antivector()                             # Antivector-
Jacobian = ga.gatype((State, State))                       # State- <- State-

liquid_volume = mv.l & State                               # [] Scalar+ <- State-
vapour_volume = mv.g & State                               # [] Scalar+ <- State-


# --- math -----------------------------------------------------------------------------
def pressure(volume: Scalar, temperature: np.ndarray | float, attraction: float, exclusion: float) -> Scalar:
    """The van der Waals equation of state balances thermal pressure in the free volume against molecular attraction."""
    return temperature / (volume - exclusion) - attraction / volume.squared()


def pressure_slope(volume: Scalar, temperature: np.ndarray | float, attraction: float, exclusion: float) -> Scalar:
    """At fixed temperature, pressure changes with volume from free-volume compression and molecular attraction."""
    return -temperature / (volume - exclusion).squared() + 2 * attraction / volume ** 3


def helmholtz_energy(volume: Scalar, temperature: np.ndarray | float, capacity: float, attraction: float,
                     exclusion: float) -> Scalar:
    """The Helmholtz free energy has minus the pressure as its slope with respect to volume at fixed temperature."""
    return (-capacity * temperature * np.log(temperature) - temperature * (volume - exclusion).log()
            - attraction / volume)


def chemical_potential(volume: Scalar, temperature: np.ndarray | float, capacity: float, attraction: float,
                       exclusion: float) -> Scalar:
    """The chemical potential adds pressure-volume work to the Helmholtz free energy, giving the free-energy tangent's
    intercept at zero volume."""
    return (helmholtz_energy(volume, temperature, capacity, attraction, exclusion)
            + pressure(volume, temperature, attraction, exclusion) * volume)


def enthalpy(volume: Scalar, temperature: np.ndarray | float, capacity: float, attraction: float,
             exclusion: float) -> Scalar:
    """The enthalpy adds pressure-volume work to the internal energy, so its change at fixed pressure equals the heat
    absorbed."""
    return (capacity * temperature - attraction / volume
            + pressure(volume, temperature, attraction, exclusion) * volume)


def isobar_temperature(volume: Scalar, isobar_pressure: Scalar, attraction: float, exclusion: float) -> Scalar:
    """Along an isobar of fixed pressure, the equation of state gives the temperature at each volume."""
    return (isobar_pressure + attraction / volume.squared()) * (volume - exclusion)


def volume_at(temperature: np.ndarray | float, isobar_pressure: Scalar, guess: Scalar, attraction: float,
              exclusion: float, n_steps: int) -> Scalar:
    """At fixed pressure, each temperature corresponds to a volume on the liquid or vapour branch of the isobar."""
    volume = guess
    for _ in range(n_steps):
        gap = isobar_temperature(volume, isobar_pressure, attraction, exclusion) - temperature
        slope = isobar_pressure - attraction * (volume - 2 * exclusion) / volume ** 3
        volume = volume - gap / slope                                                    # [...] Scalar+
    return volume


def mismatch(state: State, temperature: np.ndarray | float, capacity: float, attraction: float,
             exclusion: float) -> State:
    """Two trial volumes fail to coexist by their pressure difference along the liquid direction and their
    chemical-potential difference along the vapour direction."""
    liquid, vapour = liquid_volume(state), vapour_volume(state)
    pressure_gap = (pressure(liquid, temperature, attraction, exclusion)
                    - pressure(vapour, temperature, attraction, exclusion))
    chemical_gap = (chemical_potential(liquid, temperature, capacity, attraction, exclusion)
                    - chemical_potential(vapour, temperature, capacity, attraction, exclusion))
    return mv.l.dual() * pressure_gap + mv.g.dual() * chemical_gap


def mismatch_jacobian(state: State, temperature: np.ndarray | float, attraction: float,
                      exclusion: float) -> Jacobian:
    """By the Gibbs-Duhem relation a phase's isothermal chemical-potential slope is volume times its pressure slope,
    forming a two-dyad map from volume variations to pressure and chemical-potential changes."""
    liquid, vapour = liquid_volume(state), vapour_volume(state)
    liquid_step = (mv.l.dual() + liquid * mv.g.dual()) * pressure_slope(liquid, temperature, attraction, exclusion)
    vapour_step = (mv.l.dual() + vapour * mv.g.dual()) * pressure_slope(vapour, temperature, attraction, exclusion)
    return liquid_step * liquid_volume - vapour_step * vapour_volume                     # [...] State- <- State-


def coexistence(temperature: np.ndarray | float, capacity: float, attraction: float, exclusion: float,
                n_steps: int) -> State:
    """Liquid and vapour coexist at each temperature at the two volumes where both pressure and chemical potential
    match."""
    # Below the critical point the coexisting volumes part with the square root of the temperature drop,
    # starting the liquid branch below the critical volume and the vapour branch above it.
    departure = np.sqrt(1 - temperature)
    state = mv.l.dual() / (1 + 2 * departure) + mv.g.dual() / (1 - departure) ** 3       # [...] State-
    for _ in range(n_steps):
        jacobian = mismatch_jacobian(state, temperature, attraction, exclusion)          # [...] State- <- State-
        state = state - jacobian.solve(mismatch(state, temperature, capacity, attraction, exclusion))
    return state


def mixture(state: State, vapour_fraction: float, temperature: float, capacity: float, attraction: float,
            exclusion: float) -> tuple[Scalar, Scalar, Scalar]:
    """A two-phase mixture shares volume and enthalpy linearly between liquid and vapour by the vapour fraction."""
    liquid, vapour = liquid_volume(state), vapour_volume(state)
    volume = (1 - vapour_fraction) * liquid + vapour_fraction * vapour
    mixture_enthalpy = ((1 - vapour_fraction) * enthalpy(liquid, temperature, capacity, attraction, exclusion)
                        + vapour_fraction * enthalpy(vapour, temperature, capacity, attraction, exclusion))
    return volume, mixture_enthalpy, (1 - vapour_fraction) * liquid
