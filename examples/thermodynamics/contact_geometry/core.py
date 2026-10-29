"""A Legendre change of the controlled variable: from a gas's energy as a function of its entropy to its Helmholtz
energy as a function of its temperature.

The five quantities of the gas, energy, entropy, volume, temperature and pressure, are the readings of an antivector
of a five-dimensional space, `mv.t & point`. Equilibrium lifts each state of entropy and volume to the point whose
energy, temperature and pressure the energy function gives there, a surface in that space. Along the surface the
contact reading, `mv.u - temperature * mv.s + pressure * mv.v` paired with a change, vanishes: a change of energy is
the heat less the work. The Legendre change to the Helmholtz energy is a map of the whole space, exchanging entropy and
temperature and the energy for the energy less temperature times entropy; its derivative carries changes, and the
contact reading pulled back through it is the contact reading again, so the transformed surface is an equilibrium
surface too. The contact reading's own derivative, `mv.pv - mv.ts`, wedged twice with it, fills the
space: the contact planes twist everywhere, and no surface but such a lifted one lies in them.

Entropy is in nR, volume and temperature in reference values, energy in nRT₀.
"""
from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext, derivative

ga = Algebra("u0s0v0t0p0")
mv = JaxContext(ga, dtype=np.float64).multivector
Scalar = ga.gatype.scalar()                                         # Scalar+
Ambient = ga.gatype.antivector()                                    # Antivector-
# The states of entropy and volume, a subspace of the five quantities.
State = (mv.s.dual() + mv.v.dual()).gatype.derive.output            # Antivector-
ContactElement = ga.gatype.vector()                                 # Vector+

# The contact reading's derivative: the temperature's change wedged with the entropy's, against the pressure's with the
# volume's.
contact_derivative = mv.pv - mv.ts                                    # [] Bivector+


# --- math -----------------------------------------------------------------------------
def internal_energy(state: State, heat_capacity: float) -> Scalar:
    """The energy of an ideal gas with a constant heat capacity at fixed volume."""
    return heat_capacity * (((mv.s & state) - (mv.v & state).log()) / heat_capacity).exp()


def equilibrium(state: State, heat_capacity: float) -> Ambient:
    """Each state of entropy and volume lifted to its five quantities: the energy, and its slopes along the entropy and
    the volume, the temperature and minus the pressure."""
    potential = lambda state: internal_energy(state, heat_capacity)
    differential = derivative(potential)(state)                                       # [...] Scalar+ <- State-
    return (state + mv.u.dual() * potential(state) + mv.t.dual() * differential(mv.s.dual())
            - mv.p.dual() * differential(mv.v.dual()))


def contact_element(point: Ambient) -> ContactElement:
    """The vector whose pairing with a change vanishes when the change of energy is the heat less the work."""
    return mv.u - (mv.t & point) * mv.s + (mv.p & point) * mv.v


def helmholtz(point: Ambient) -> Ambient:
    """The Legendre change exchanging entropy and temperature: the Helmholtz energy, the energy less temperature times
    entropy, in place of the energy, the temperature in place of the entropy and minus the entropy in place of the
    temperature."""
    entropy, temperature = mv.s & point, mv.t & point
    return (mv.u.dual() * ((mv.u & point) - temperature * entropy) + mv.s.dual() * temperature - mv.t.dual() * entropy
            + mv.v.dual() * (mv.v & point) + mv.p.dual() * (mv.p & point))


def enthalpy(point: Ambient) -> Ambient:
    """The Legendre change exchanging volume and pressure, with the enthalpy as the potential."""
    volume, pressure = mv.v & point, mv.p & point
    return (mv.u.dual() * ((mv.u & point) + pressure * volume) + mv.s.dual() * (mv.s & point)
            + mv.t.dual() * (mv.t & point) + mv.v.dual() * pressure - mv.p.dual() * volume)
