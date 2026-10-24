"""Coexisting liquid and vapour of a van der Waals fluid, and boiling between them at fixed temperature and pressure.

A state of coexistence holds the two phases' volumes, an antivector of the plane spanned by the liquid's and the
vapour's. Coexistence is two conditions on the two volumes, equal pressure and equal chemical potential, which say
together that the Helmholtz energy's tangent lines at the two volumes are one line, the common tangent. Their
mismatch is a function from states to states; Newton's method solves it with the map from a correction to the change
of the mismatch that its derivative gives at each guess, batched over as many temperatures as asked for. A mixture of
the two phases lies on the common tangent, its volume and its Helmholtz energy shared out by the vapour's fraction.

Temperature and volume are in their critical values, energy in R T_c and pressure in R T_c / V_c, for a fixed amount
of the fluid.
"""
from __future__ import annotations

import jax
import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext, derivative

jax.config.update("jax_enable_x64", True)
ga = Algebra("l0g0")
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()
State = ga.gatype.antivector()


# --- math -----------------------------------------------------------------------------
def pressure(volume: Scalar, temperature: np.ndarray, attraction: float, exclusion: float) -> Scalar:
    """The van der Waals equation of state."""
    return temperature / (volume - exclusion) - attraction / volume.squared()


def helmholtz_energy(volume: Scalar, temperature: np.ndarray, capacity: float, attraction: float,
                     exclusion: float) -> Scalar:
    """The Helmholtz energy whose slope along the volume is minus the pressure."""
    return -capacity * temperature * np.log(temperature) - temperature * (volume - exclusion).log() - attraction / volume


def chemical_potential(volume: Scalar, temperature: np.ndarray, capacity: float, attraction: float,
                       exclusion: float) -> Scalar:
    """The Helmholtz energy plus the work of making room at the pressure: where the tangent at a volume crosses zero
    volume."""
    return (helmholtz_energy(volume, temperature, capacity, attraction, exclusion)
            + pressure(volume, temperature, attraction, exclusion) * volume)


def enthalpy(volume: Scalar, temperature: np.ndarray, capacity: float, attraction: float, exclusion: float) -> Scalar:
    """The energy, the heat capacity's share less the attraction's, plus the work of making room at the pressure: the
    heat taken in at fixed pressure is its change."""
    return (capacity * temperature - attraction / volume
            + pressure(volume, temperature, attraction, exclusion) * volume)


def isobar_temperature(volume: Scalar, level: float, attraction: float, exclusion: float) -> Scalar:
    """The equation of state solved for the temperature: where the isobar at a pressure level passes each volume."""
    return (level + attraction / volume.squared()) * (volume - exclusion)


def volume_at(temperature: np.ndarray, level: float, guess: Scalar, attraction: float, exclusion: float,
              n_steps: int) -> Scalar:
    """The volume where the isobar at a pressure level reaches each temperature, by Newton's method from a guess on
    the liquid's or the vapour's side."""
    gap = lambda volume: isobar_temperature(volume, level, attraction, exclusion) - temperature
    volume = guess
    for _ in range(n_steps):
        volume = volume - derivative(gap)(volume).solve(gap(volume))
    return volume


def mismatch(state: State, temperature: np.ndarray, capacity: float, attraction: float, exclusion: float) -> State:
    """How far the phases at the state's two volumes are from coexisting: their difference in pressure along the
    liquid's volume, in chemical potential along the vapour's."""
    liquid, vapour = mv.l & state, mv.g & state
    pressure_gap = pressure(liquid, temperature, attraction, exclusion) - pressure(vapour, temperature, attraction, exclusion)
    chemical_gap = (chemical_potential(liquid, temperature, capacity, attraction, exclusion)
                    - chemical_potential(vapour, temperature, capacity, attraction, exclusion))
    return mv.l.dual() * pressure_gap + mv.g.dual() * chemical_gap


def coexistence(temperature: np.ndarray, capacity: float, attraction: float, exclusion: float,
                n_steps: int) -> State:
    """The coexisting volumes at each temperature `[...]`, by Newton's method: each step solves the mismatch's
    derivative, a map from a correction of the state to the change of the mismatch, for the correction that cancels
    the mismatch."""
    # Below the critical point the two volumes part as the square root of the distance from it, the liquid's
    # slowly, the vapour's fast; this starts each on its own side.
    below = np.sqrt(1 - temperature)
    state = mv.l.dual() / (1 + 2 * below) + mv.g.dual() / (1 - below) ** 3                 # [...] State
    gap = lambda state: mismatch(state, temperature, capacity, attraction, exclusion)
    for _ in range(n_steps):
        state = state - derivative(gap)(state).solve(gap(state))                          # [...] State
    return state


def mixture(coexisting: State, vapour_fraction: float, temperature: float, capacity: float, attraction: float,
            exclusion: float) -> tuple[Scalar, Scalar, Scalar]:
    """A mixture with a fraction of its amount as vapour: its volume and its enthalpy, each shared out between the
    phases by that fraction, and the volume its liquid takes up."""
    liquid, vapour = mv.l & coexisting, mv.g & coexisting
    heat_content = lambda volume: enthalpy(volume, temperature, capacity, attraction, exclusion)
    volume = (1 - vapour_fraction) * liquid + vapour_fraction * vapour
    shared_enthalpy = (1 - vapour_fraction) * heat_content(liquid) + vapour_fraction * heat_content(vapour)
    return volume, shared_enthalpy, (1 - vapour_fraction) * liquid
