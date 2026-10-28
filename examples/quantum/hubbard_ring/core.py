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
Scalar = ga.gatype.scalar()                                     # Scalar+
Orbital = ga.gatype.vector()                                    # Vector+
State = ga.gatype.k_vector(4)                                   # 4-vector+
Complement = ga.gatype.antivector()                             # Antivector-
OneElectron = ga.gatype((Orbital, Orbital))
Hamiltonian = ga.gatype((State, State))

# The orbitals by site and spin, up then down, and each site's neighbour along the ring.
orbitals = stack([stack([mv.a, mv.b]), stack([mv.c, mv.d]), stack([mv.e, mv.f]), stack([mv.g, mv.h])])   # [sites, spins] Orbital+
following = stack([orbitals[1], orbitals[2], orbitals[3], orbitals[0]])   # [sites, spins] Orbital+


# --- math -----------------------------------------------------------------------------
def lift(one_electron: OneElectron) -> Hamiltonian:
    """A one-electron map acting on each of the four electrons in turn: the join with `Complement`, the
    space of all complements, removes one occupied orbital, wedging its image back in replaces it, with
    its exchange sign, and the trace pairs the map's input with the orbital removed."""
    return (one_electron ^ (State & Complement)).trace(1, 3)    # [...] State+ <- State+


def hopping(strength: float) -> Hamiltonian:
    """An electron moving to a neighbouring site without changing its spin, lifted to four electrons."""
    single = -strength * (following * (orbitals | Orbital) + orbitals * (following | Orbital)).sum(axis=0).sum(axis=0)   # [] Orbital+ <- Orbital+
    return lift(single)                                         # [] State+ <- State+


# How many electrons each orbital holds: its projector, lifted.
numbers = lift(orbitals * (orbitals | Orbital))                 # [sites, spins] State+ <- State+
# How many sites hold both spins: on each site the up number composed with the down number.
double_occupancy = numbers[:, 0](numbers[:, 1]).sum(axis=0)     # [] State+ <- State+


def turn(energies: Scalar, modes: State, start: State, times: np.ndarray) -> tuple[State, State]:
    """The cosine and the sine of the Hamiltonian times time applied to a start, from the Hamiltonian's
    states: each keeps its share of the start and turns by its own energy."""
    shares = modes * modes.reverse().scalar_product(start)      # [..., modes] State+
    angles = energies[..., None, :] * times[..., :, None]       # [..., times, modes] Scalar+
    cosine = (shares[..., None, :] * angles.cos()).sum(axis=-1)   # [..., times] State+
    sine = (shares[..., None, :] * angles.sin()).sum(axis=-1)   # [..., times] State+
    return cosine, sine


def occupation(parts: tuple[State, State]) -> Scalar:
    """How many electrons each orbital holds, summed over the cosine and sine parts."""
    cosine, sine = (part[..., None, None] for part in parts)    # [..., 1, 1] State+, State+
    return cosine.reverse().scalar_product(numbers(cosine)) + sine.reverse().scalar_product(numbers(sine))   # [..., sites, spins] Scalar+
