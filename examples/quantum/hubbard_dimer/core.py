"""The Hubbard dimer models two electrons hopping between two sites with an energy cost for sharing a site.

Four vector directions represent the left up, left down, right up, and right down orbitals. A
two-electron state is a bivector, whose wedge enforces Pauli exclusion and fermionic exchange signs.
One-electron maps act on either occupied orbital in turn via an incidence trace, while repulsion acts
on the two shared-site configurations. Energies share one unit, and time evolution sets reduced Planck's
constant to 1.

Hopping couples the singlet only to the symmetric double occupancy, leaving the three triplets at zero
energy. On that two-state block, applying the Hamiltonian minus half the repulsion twice multiplies by a
scalar, the square of the block's rate. The block's two energies are half the repulsion plus or minus
that rate, its ground state projects the singlet by the rate minus the shifted Hamiltonian, and the
singlet-triplet gap equals the rate minus half the repulsion. Time evolution turns each state through
the cosine and sine of the Hamiltonian times time, pairing real cosine and sine state components whose
configuration amplitudes trace planar dials.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                     # Scalar+
Orbital = ga.gatype.vector()                                    # Vector+
Pair = ga.gatype.bivector()                                     # Bivector+
Complement = ga.gatype.antivector()                             # Antivector-
Hopping = ga.gatype((Orbital, Orbital))
Hamiltonian = ga.gatype((Pair, Pair))

# Orbitals within each site are ordered spin up then spin down.
left = stack([mv.a, mv.b])                                      # [2] Vector+
right = stack([mv.c, mv.d])                                     # [2] Vector+
orbitals = stack([mv.a, mv.b, mv.c, mv.d])                      # [4] Vector+
# A magnetic field opposite on the two sites raises spin up and lowers spin down on the left, and reverses on the right.
field_signs = np.array([1, -1, -1, 1])
# A site energy offset lowers both spins on the left site and raises both on the right site.
site_signs = np.array([-1, -1, 1, 1])
configurations = stack([mv.ab, mv.ac, mv.ad, mv.bc, mv.bd, mv.cd])   # [6] Bivector+
doubles = stack([mv.ab, mv.cd])                                 # [2] Bivector+
singlet = (mv.ad - mv.bc) / np.sqrt(2)                          # [] Bivector+
zero_triplet = (mv.ad + mv.bc) / np.sqrt(2)                     # [] Bivector+
triplets = stack([mv.ac, zero_triplet, mv.bd])                  # [3] Bivector+
symmetric_double = (mv.ab + mv.cd) / np.sqrt(2)                 # [] Bivector+

# Each dyad projects onto one shared-site configuration; their sum counts double occupancy.
double_occupancy = (doubles * doubles.reverse().scalar_product(Pair)).sum(axis=0)   # [] Bivector+ <- Bivector+
singlet_weight = singlet * singlet.reverse().scalar_product(Pair)   # [] Bivector+ <- Bivector+
# Doubly occupied states carry zero local spin, while singlets have spin alignment -3/4 and triplets +1/4.
spin_alignment = (Pair - double_occupancy - 4 * singlet_weight) / 4   # [] Bivector+ <- Bivector+


# --- math -----------------------------------------------------------------------------
def pair_extension(one_electron: Hopping) -> Hamiltonian:
    """An orbital map extends to two-electron pairs as a derivation acting on each wedge factor in turn.
    Joining `Pair & Complement` removes one occupied orbital, wedging `one_electron` inserts its image
    with the fermionic exchange sign, and the trace contracts the removed orbital with `Complement`."""
    return (one_electron ^ (Pair & Complement)).trace(1, 3)     # [] Bivector+ <- Bivector+


# Unit-strength hopping moves an electron to the opposite site with its spin preserved, then extends to pairs.
hop = (left * (right | Orbital) + right * (left | Orbital)).sum(axis=0)   # [] Vector+ <- Vector+
pair_hop = pair_extension(hop)                                  # [] Bivector+ <- Bivector+
# Opposite magnetic fields of +-1/2 on the two sites extend to a two-electron map.
field_difference = pair_extension((field_signs * (orbitals * (orbitals | Orbital))).sum(axis=0) / 2)   # [] Bivector+ <- Bivector+
# Site offsets of -1/2 on the left and +1/2 on the right measure half the right occupation minus the left.
site_difference = pair_extension((site_signs * (orbitals * (orbitals | Orbital))).sum(axis=0) / 2)   # [] Bivector+ <- Bivector+


def hamiltonian(strength: float, repulsion: np.ndarray) -> Hamiltonian:
    """Two-electron hopping at `strength` competes with on-site `repulsion` (`[...] Bivector+ <- Bivector+`)."""
    return -strength * pair_hop + repulsion * double_occupancy  # [...] Bivector+ <- Bivector+


def block_rate(strength: float, repulsion: np.ndarray) -> np.ndarray:
    """Applying the Hamiltonian minus half `repulsion` twice on the singlet block multiplies by the
    square of this rate (`[...]`)."""
    return np.sqrt((repulsion / 2) ** 2 + 4 * strength**2)


def ground_state(strength: float, repulsion: np.ndarray) -> Pair:
    """Subtracting the offset Hamiltonian from `block_rate` projects the singlet onto the normalized
    ground state (`[...] Bivector+`)."""
    offset_singlet = hamiltonian(strength, repulsion)(singlet) - repulsion / 2 * singlet   # [...] Bivector+
    lower = block_rate(strength, repulsion) * singlet - offset_singlet   # [...] Bivector+
    return lower / lower.scalar_norm_squared().square_root()    # [...] Bivector+


def exchange_parts(strength: float, repulsion: np.ndarray, times: np.ndarray) -> tuple[Pair, Pair]:
    """Starting from spin up on the left and spin down on the right, time evolution over `times` yields
    the cosine and sine state parts (`[...] Bivector+`, `[...] Bivector+`). The zero-energy triplet stays
    in the cosine part while the singlet rotates at `block_rate` and mixes under half `repulsion`."""
    rate = block_rate(strength, repulsion)
    offset_singlet = hamiltonian(strength, repulsion)(singlet) - repulsion / 2 * singlet   # [...] Bivector+
    singlet_cosine = np.cos(rate * times) * singlet             # [...] Bivector+
    singlet_sine = np.sin(rate * times) / rate * offset_singlet   # [...] Bivector+
    half_phase = repulsion * times / 2
    cosine_part = (zero_triplet + np.cos(half_phase) * singlet_cosine - np.sin(half_phase) * singlet_sine) / np.sqrt(2)   # [...] Bivector+
    sine_part = (np.sin(half_phase) * singlet_cosine + np.cos(half_phase) * singlet_sine) / np.sqrt(2)   # [...] Bivector+
    return cosine_part, sine_part


def exchange_states(strength: float, repulsion: np.ndarray, exchange_phase: np.ndarray) -> Iterator[tuple[Pair, Pair]]:
    """Yields `(cosine_part, sine_part)` (`[...] Bivector+`, `[...] Bivector+`) step by step along
    `exchange_phase`, with each regime scaled to its own singlet-triplet gap."""
    gap = block_rate(strength, repulsion) - repulsion / 2       # [...]
    for step_phase in exchange_phase:
        yield exchange_parts(strength, repulsion, step_phase / gap)


def mode_evolution(energies: Scalar, modes: Pair, start: Pair, times: np.ndarray) -> Iterator[tuple[Pair, Pair]]:
    """Yields `(cosine_part, sine_part)` (`[...] Bivector+`, `[...] Bivector+`) along `times` as each
    eigenmode in `modes` (`[..., 6] Bivector+`) turns its share of `start` at its energy in `energies`."""
    shares = modes * modes.reverse().scalar_product(start)      # [..., 6] Bivector+
    for step_time in times:
        angles = energies * step_time                           # [..., 6] Scalar+
        cosine_part = (shares * angles.cos()).sum(axis=-1)      # [...] Bivector+
        sine_part = (shares * angles.sin()).sum(axis=-1)        # [...] Bivector+
        yield cosine_part, sine_part


def probabilities(state: Pair) -> Scalar:
    """Squared overlaps with the six basis configurations give the occupation probabilities
    (`[..., 6] Scalar+`) of a real pair `state`."""
    return configurations.reverse().scalar_product(state[..., None]).squared()   # [..., 6] Scalar+


def expectation(state: Pair, observable: Hamiltonian) -> Scalar:
    """Pairing `state` with `observable(state)` gives the quadratic expectation value (`[...] Scalar+`)."""
    return state.reverse().scalar_product(observable(state))    # [...] Scalar+
