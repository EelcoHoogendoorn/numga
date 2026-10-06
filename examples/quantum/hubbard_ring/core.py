"""Four electrons hopping around a ring of four sites, with an energy cost for sharing a site.

The eight vector blades are the orbitals, spin up and spin down on each site. A state of four electrons
is a 4-vector, the wedge of the four occupied orbitals: it vanishes when two share an orbital and flips
sign when two trade places, also when an electron hops across the ring's seam past the other three. A
one-electron map acts on each occupied orbital in turn, lifted by an incidence trace, the same one line
for any number of electrons. Hopping lifts that way, and so does each orbital's projector, the number of
electrons in it; composing the up and down numbers of a site counts whether the site holds both. All
energies use the hopping strength as their unit, and evolution uses hbar = 1.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+e+f+g+h+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Orbital = ga.gatype.vector()
State = ga.gatype(ga.subspace.k_vector(4))
Complement = ga.gatype.antivector()
OneElectron = ga.gatype((Orbital, Orbital))
Hamiltonian = ga.gatype((State, State))

# The orbitals by site and spin, up then down, and each site's neighbour along the ring.
ORBITALS = stack([stack([mv.a, mv.b]), stack([mv.c, mv.d]), stack([mv.e, mv.f]), stack([mv.g, mv.h])])   # [sites, spins] Orbital
FOLLOWING = stack([ORBITALS[1], ORBITALS[2], ORBITALS[3], ORBITALS[0]])          # [sites, spins] Orbital


# --- math -----------------------------------------------------------------------------
def lift(one_electron: OneElectron) -> Hamiltonian:
    """A one-electron map acting on each of the four electrons in turn: the join with an open
    complement removes one occupied orbital, wedging its image back in replaces it, with its exchange
    sign, and the trace pairs the map's input with the complementary slot."""
    return (one_electron ^ (State & Complement)).trace(1, 3)   # [...] State <- State


def hopping(strength: float) -> Hamiltonian:
    """Move an electron to a neighbouring site without changing its spin, lifted to four electrons."""
    single = -strength * (FOLLOWING * (ORBITALS | Orbital) + ORBITALS * (FOLLOWING | Orbital)).sum(axis=0).sum(axis=0)
    return lift(single)                                         # [] State <- State


def numbers() -> Hamiltonian:
    """How many electrons each orbital holds: its projector, lifted."""
    return lift(ORBITALS * (ORBITALS | Orbital))               # [sites, spins] State <- State


def double_occupancy() -> Hamiltonian:
    """How many sites hold both spins: on each site the up number composed with the down number."""
    counts = numbers()                                          # [sites, spins] State <- State
    return counts[:, 0](counts[:, 1]).sum(axis=0)               # [] State <- State


def hamiltonian(strength: float, repulsion: np.ndarray) -> Hamiltonian:
    """Electrons hop around the ring, and sharing a site costs the repulsion."""
    return hopping(strength) + repulsion * double_occupancy()   # [...] State <- State


def turn(energies: Scalar, modes: State, start: State, times: np.ndarray) -> tuple[State, State]:
    """The cosine and the sine of the Hamiltonian times time applied to a start, from the Hamiltonian's
    states: each keeps its share of the start and turns by its own energy."""
    shares = modes * modes.reverse().scalar_product(start)     # [..., modes] State
    angles = energies[..., None, :] * times[..., :, None]      # [..., times, modes] Scalar
    return (shares[..., None, :] * angles.cos()).sum(axis=-1), (shares[..., None, :] * angles.sin()).sum(axis=-1)


def occupation(parts: tuple[State, State]) -> Scalar:
    """How many electrons each orbital holds, summed over the cosine and sine parts."""
    counts = numbers()                                          # [sites, spins] State <- State
    cosine, sine = (part[..., None, None] for part in parts)    # [..., 1, 1] State each
    return cosine.reverse().scalar_product(counts(cosine)) + sine.reverse().scalar_product(counts(sine))   # [..., sites, spins] Scalar
