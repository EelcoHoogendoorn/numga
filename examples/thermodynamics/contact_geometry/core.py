"""A Legendre transformation maps a gas's internal energy as a function of entropy to its Helmholtz free energy as a
function of temperature while preserving the contact form.

The five thermodynamic variables of the gas (internal energy, entropy, volume, temperature, and pressure) are modelled
as components of an antivector in the exterior space `u0 s0 v0 t0 p0`, read by pairing (`t & point`). Equilibrium embeds
each state of entropy and volume as the point with the internal energy, temperature, and pressure determined by the
fundamental relation, forming a two-dimensional surface in that five-dimensional space. Along the surface, the contact
covector `u - temperature * s + pressure * v` pairs to zero with every tangent step because a change of internal energy
equals heat added minus mechanical work done. The Legendre transformation to the Helmholtz free energy maps the ambient
space to itself, exchanging entropy and temperature up to sign and replacing internal energy with `energy - temperature
* entropy`. Pulling the contact covector back through the adjugate of the transformation's pushforward recovers the
original contact covector, so equilibrium surfaces map to equilibrium surfaces. Wedging the contact derivative `pv - ts`
twice with the contact covector produces a nonvanishing five-blade everywhere, showing that the contact hyperplanes are
maximally non-integrable and admit only two-dimensional equilibrium surfaces.

Entropy is measured in units of the gas constant times the number of moles, volume and temperature in units of their
reference values, and energy in units of the reference thermal energy.
"""
from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("u0s0v0t0p0")
mv = NumpyContext(ga, dtype=np.float64).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Ambient = ga.gatype.antivector()                                              # Antivector-
# States in the two-dimensional subspace of entropy and volume.
State = ga.gatype(mv.s.dual() + mv.v.dual())                                  # Antivector-
Potential = ga.gatype.vector()                                                # Vector+
Pushforward = ga.gatype((Ambient, Ambient))

# Exterior derivative of the contact covector, pairing temperature with entropy against pressure with volume.
contact_derivative = mv.pv - mv.ts                                            # [] Bivector+

# Identity component along volume and pressure, fixed under the entropy-temperature Legendre map.
unchanged_mechanical = mv.v.dual() * (mv.v & Ambient) + mv.p.dual() * (mv.p & Ambient)   # [] Ambient- <- Ambient-
entropy_temperature_swap = mv.s.dual() * (mv.t & Ambient) - mv.t.dual() * (mv.s & Ambient)   # [] Ambient- <- Ambient-


# --- math -----------------------------------------------------------------------------
def internal_energy(state: State, heat_capacity: float) -> Scalar:
    """Internal energy of an ideal gas with constant heat capacity at fixed volume."""
    return heat_capacity * (((mv.s & state) - (mv.v & state).log()) / heat_capacity).exp()


def equilibrium(state: State, heat_capacity: float) -> Ambient:
    """Equilibrium point in the five-dimensional space for each state of entropy and volume."""
    energies = internal_energy(state, heat_capacity)                          # [...] Scalar+
    temperatures = energies / heat_capacity                                   # [...] Scalar+
    pressures = temperatures / (mv.v & state)                                 # [...] Scalar+
    return state + mv.u.dual() * energies + mv.t.dual() * temperatures + mv.p.dual() * pressures


def entropy_tangent(state: State, heat_capacity: float) -> Ambient:
    """Tangent vector to the equilibrium surface along a unit step of entropy at fixed volume."""
    energies = internal_energy(state, heat_capacity)                          # [...] Scalar+
    temperatures = energies / heat_capacity                                   # [...] Scalar+
    pressures = temperatures / (mv.v & state)                                 # [...] Scalar+
    return (mv.s.dual() + mv.u.dual() * temperatures
            + (mv.t.dual() * temperatures + mv.p.dual() * pressures) / heat_capacity)


def contact_element(point: Ambient) -> Potential:
    """Contact covector whose pairing with a step vanishes when energy change equals heat minus work."""
    return mv.u - (mv.t & point) * mv.s + (mv.p & point) * mv.v               # [...] Potential+


def helmholtz(point: Ambient) -> Ambient:
    """Legendre image replacing internal energy with Helmholtz free energy and swapping entropy with temperature."""
    entropy, temperature = mv.s & point, mv.t & point                         # [...] Scalar+, Scalar+
    return (mv.u.dual() * ((mv.u & point) - temperature * entropy) + mv.s.dual() * temperature - mv.t.dual() * entropy
            + mv.v.dual() * (mv.v & point) + mv.p.dual() * (mv.p & point))


def helmholtz_pushforward(point: Ambient) -> Pushforward:
    """Linear pushforward of ambient tangent vectors under the Helmholtz Legendre transformation."""
    entropy, temperature = mv.s & point, mv.t & point                         # [...] Scalar+, Scalar+
    energy_step = (mv.u - entropy * mv.t - temperature * mv.s) & Ambient      # [...] Scalar+ <- Ambient-
    return mv.u.dual() * energy_step + entropy_temperature_swap + unchanged_mechanical   # [...] Ambient- <- Ambient-


def enthalpy(point: Ambient) -> Ambient:
    """Legendre image replacing internal energy with enthalpy and swapping volume with pressure."""
    volume, pressure = mv.v & point, mv.p & point                             # [...] Scalar+, Scalar+
    return (mv.u.dual() * ((mv.u & point) + pressure * volume) + mv.s.dual() * (mv.s & point)
            + mv.t.dual() * (mv.t & point) + mv.v.dual() * pressure - mv.p.dual() * volume)
