"""Exterior powers of single-electron orbitals represent many-fermion states on a four-site Hubbard ring.

Eight orthogonal vector blades represent the spin-up and spin-down orbitals on four ring sites, and a
four-electron state is a 4-vector formed by wedging four occupied orbitals. Repeated orbitals vanish by
exclusion, and exchanging two orbitals flips the sign, including hops across the periodic seam past the
remaining three electrons. A single-orbital linear map lifts to the four-electron space through one
incidence trace that removes each occupied orbital in turn, inserts its image with the fermion sign, and
contracts the map's input against the removed orbital. Nearest-neighbour hopping and orbital projectors
both lift through that trace, and composing the spin-up and spin-down number maps on each site counts
doubly occupied sites.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+e+f+g+h+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                         # Scalar+
Orbital = ga.gatype.vector()                                        # Vector+
State = ga.gatype.k_vector(4)                                       # 4-vector+
Complement = ga.gatype.antivector()                                 # Antivector-
OneElectron = ga.gatype((Orbital, Orbital))                         # Orbital+ <- Orbital+
Hamiltonian = ga.gatype((State, State))                             # State+ <- State+

# Spin-up and spin-down orbitals on the four ring sites, and on each site's next neighbour around the ring.
orbitals = stack([stack([mv.a, mv.b]), stack([mv.c, mv.d]), stack([mv.e, mv.f]), stack([mv.g, mv.h])])   # [4, 2] Orbital+
next_orbitals = stack([orbitals[1], orbitals[2], orbitals[3], orbitals[0]])                              # [4, 2] Orbital+


def lifted_map(one_electron: OneElectron) -> Hamiltonian:
    """The derivation trace lifts `one_electron` (`Orbital+ <- Orbital+`) to act on each of the four
    occupied orbitals in turn (`[...] State+ <- State+`)."""
    # Joining with every complement removes one occupied orbital, wedging inserts its image with the
    # fermion exchange sign, and the trace pairs the map's input with the orbital removed.
    return (one_electron ^ (State & Complement)).trace(1, 3)        # [...] State+ <- State+


# Nearest-neighbour orbital transfer around the ring, lifted to four electrons, alongside the lifted
# orbital projectors and their per-site spin-up/spin-down product counting doubly occupied sites.
orbital_transfer = (next_orbitals * (orbitals | Orbital) + orbitals * (next_orbitals | Orbital)).sum(axis=0).sum(axis=0)   # [] Orbital+ <- Orbital+
ring_hopping = lifted_map(orbital_transfer)                         # [] State+ <- State+
orbital_numbers = lifted_map(orbitals * (orbitals | Orbital))       # [4, 2] State+ <- State+
double_occupancy = orbital_numbers[:, 0](orbital_numbers[:, 1]).sum(axis=0)   # [] State+ <- State+


def exchange_rate(hopping_strength: float, repulsions: np.ndarray) -> np.ndarray:
    """Virtual neighbour hops at strong on-site `repulsions` (`[...]`) couple adjacent spins at rate
    `4 * hopping_strength**2 / repulsions` (`[...]`)."""
    return 4 * hopping_strength**2 / repulsions


def spin_excitations(hopping_strength: float, repulsions: np.ndarray, n_spin_levels: int) -> Scalar:
    """Diagonalizing the Hubbard ring across `repulsions` (`[...]`) gives the lowest `n_spin_levels`
    excitation energies in units of the spin exchange (`[..., n_spin_levels] Scalar+`)."""
    hamiltonians = -hopping_strength * ring_hopping + repulsions * double_occupancy   # [...] State+ <- State+
    energies = hamiltonians.eigvalsh()                                                # [..., 70] Scalar+
    exchange = exchange_rate(hopping_strength, repulsions)                            # [...]
    return (energies[..., :n_spin_levels] - energies[..., :1]) / exchange[..., None]  # [..., n_spin_levels] Scalar+


def spin_walk(hopping_strength: float, repulsions: np.ndarray, initial_state: State,
              exchange_phases: np.ndarray) -> Iterator[Scalar]:
    """Yields the electron occupation of every site and spin orbital (`[..., 4, 2] Scalar+`) across
    `exchange_phases` (`[n_frames]`) as a flipped spin walks around the ring."""
    hamiltonians = -hopping_strength * ring_hopping + repulsions * double_occupancy   # [...] State+ <- State+
    energies, modes = hamiltonians.eigh()                                             # [..., 70] Scalar+, State+
    shares = modes * modes.reverse().scalar_product(initial_state)                    # [..., 70] State+
    exchange = exchange_rate(hopping_strength, repulsions)                            # [...]
    for phase in exchange_phases:
        # Each eigenstate rotates at its own energy, splitting the evolved state into real cosine and sine parts.
        angles = energies * (phase / exchange)[..., None]                             # [..., 70] Scalar+
        cosine_part = (shares * angles.cos()).sum(axis=-1)[..., None, None]           # [..., 1, 1] State+
        sine_part = (shares * angles.sin()).sum(axis=-1)[..., None, None]             # [..., 1, 1] State+
        yield (cosine_part.reverse().scalar_product(orbital_numbers(cosine_part))
               + sine_part.reverse().scalar_product(orbital_numbers(sine_part)))      # [..., 4, 2] Scalar+
