"""The Hubbard dimer: two electrons hopping between two sites, with an energy cost for sharing a site.

The four vector blades label left up, left down, right up and right down orbitals. A
two-electron state is a bivector: its wedge supplies exclusion and exchange signs. The
one-electron hopping map acts on each occupied orbital in turn, lifted by an incidence
trace. Repulsion acts on the two same-site configurations. All energies use the same
unit, and evolution uses hbar = 1.

Hopping takes the singlet only to the symmetric double occupancy, and back. On those two states
the Hamiltonian less half the repulsion, applied twice, is a number, the square of the block's
rate: so their energies are half the repulsion plus or minus the rate, the lower one is the singlet
projected by the rate less the shifted Hamiltonian, and time turns the singlet by a cosine and a
sine. The three triplets stay at zero energy, and the singlet-triplet gap is the rate less half the
repulsion. Time acts on a state through the cosine and the sine of the Hamiltonian times time, so a
state in time is a pair of real states, its cosine part and its sine part: each configuration's
amplitude is a point in a plane with those two coordinates, and its probability the squared distance.
"""

from __future__ import annotations

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

# Orbitals within each site are ordered by spin, up then down.
left = stack([mv.a, mv.b])                                      # [spins] Orbital+
right = stack([mv.c, mv.d])                                     # [spins] Orbital+
orbitals = stack([mv.a, mv.b, mv.c, mv.d])                      # [orbitals] Orbital+
# A field along z that differs between the sites raises up and lowers down on the left, the other way
# round on the right.
FIELD_SIGNS = np.array([1, -1, -1, 1])
# An asymmetry between the sites lowers the left one and raises the right, for both spins alike.
SITE_SIGNS = np.array([-1, -1, 1, 1])
configurations = stack([mv.ab, mv.ac, mv.ad, mv.bc, mv.bd, mv.cd])   # [configurations] Pair+
doubles = stack([mv.ab, mv.cd])                                 # [sites] Pair+
singlet = (mv.ad - mv.bc) / np.sqrt(2)                          # [] Pair+
zero_triplet = (mv.ad + mv.bc) / np.sqrt(2)                     # [] Pair+
triplets = stack([mv.ac, zero_triplet, mv.bd])                  # [spin_projections] Pair+
symmetric_double = (mv.ab + mv.cd) / np.sqrt(2)                 # [] Pair+

# Each dyad measures a configuration and returns that same configuration.
double_occupancy = (doubles * doubles.reverse().scalar_product(Pair)).sum(axis=0)   # [] Pair+ <- Pair+
singlet_weight = singlet * singlet.reverse().scalar_product(Pair)   # [] Pair+ <- Pair+
# Empty and doubly occupied sites carry no spin. Singlets have correlation -3/4,
# triplets +1/4, measured in hbar squared.
spin_correlation = (Pair - double_occupancy - 4 * singlet_weight) / 4   # [] Pair+ <- Pair+


# --- math -----------------------------------------------------------------------------
def lift(one_electron: Hopping) -> Hamiltonian:
    """A one-electron map acting on either electron of a pair in turn. The trace pairs the orbital
    removed from the state with its replacement:
    lift(single)(first ^ second) == single(first) ^ second + first ^ single(second)."""
    return (one_electron ^ (Pair & Complement)).trace(1, 3)     # [] Pair+ <- Pair+


# An electron moved to the other site with its spin kept, at unit hopping strength, and lifted to pairs.
hop = (left * (right | Orbital) + right * (left | Orbital)).sum(axis=0)   # [] Orbital+ <- Orbital+
pair_hop = lift(hop)                                            # [] Pair+ <- Pair+
# A unit field along z differing between the sites, by half on each, lifted to pairs.
field_difference = lift(0.5 * (FIELD_SIGNS * (orbitals * (orbitals | Orbital))).sum(axis=0))   # [] Pair+ <- Pair+
# A unit asymmetry between the sites, by half on each, lifted to pairs: half the right site's occupation
# less the left's.
site_difference = lift(0.5 * (SITE_SIGNS * (orbitals * (orbitals | Orbital))).sum(axis=0))   # [] Pair+ <- Pair+


def hamiltonian(strength: float, repulsion: np.ndarray) -> Hamiltonian:
    """Either electron hops, and sharing a site costs the repulsion."""
    return -strength * pair_hop + repulsion * double_occupancy  # [...] Pair+ <- Pair+


def block_rate(strength: float, repulsion: np.ndarray) -> np.ndarray:
    """The singlet block has a rate whose square is the shifted Hamiltonian applied twice on the singlet
    and the symmetric double occupancy."""
    return np.sqrt((repulsion / 2) ** 2 + 4 * strength**2)


def shifted_singlet(strength: float, repulsion: np.ndarray) -> Pair:
    """The Hamiltonian less half the repulsion takes the singlet to this state."""
    return hamiltonian(strength, repulsion)(singlet) - repulsion / 2 * singlet   # [...] Pair+


def ground(strength: float, repulsion: np.ndarray) -> Pair:
    """The lowest state is the singlet projected onto the lower state of its block, by the rate less
    the shifted Hamiltonian, and normalized."""
    lower = block_rate(strength, repulsion) * singlet - shifted_singlet(strength, repulsion)   # [...] Pair+
    return lower / lower.scalar_norm_squared().square_root()    # [...] Pair+


def exchange(strength: float, repulsion: np.ndarray, times: np.ndarray) -> tuple[Pair, Pair]:
    """After the given times, spin up on the left and spin down on the right, half singlet and half
    triplet of zero spin, has become the cosine and the sine of the Hamiltonian times time applied to it. The
    triplet, at zero energy, sits in the cosine part. On the singlet's block the Hamiltonian is half
    the repulsion plus a shifted part whose cosine and sine act on the singlet as the cosine and sine
    of the rate; the half repulsion mixes the two by its own cosine and sine."""
    rate = block_rate(strength, repulsion)
    singlet_cosine = np.cos(rate * times) * singlet             # [...] Pair+
    singlet_sine = np.sin(rate * times) / rate * shifted_singlet(strength, repulsion)   # [...] Pair+
    half_phase = 0.5 * repulsion * times
    cosine_part = (zero_triplet + np.cos(half_phase) * singlet_cosine - np.sin(half_phase) * singlet_sine) / np.sqrt(2)   # [...] Pair+
    sine_part = (np.sin(half_phase) * singlet_cosine + np.cos(half_phase) * singlet_sine) / np.sqrt(2)   # [...] Pair+
    return cosine_part, sine_part


def turn(energies: Scalar, modes: Pair, start: Pair, times: np.ndarray) -> tuple[Pair, Pair]:
    """The cosine and the sine of a Hamiltonian times time applied to a start follow from the
    Hamiltonian's states: each keeps its share of the start and turns by its own energy."""
    shares = modes * modes.reverse().scalar_product(start)      # [..., modes] Pair+
    angles = energies[..., None, :] * times[..., :, None]       # [..., times, modes] Scalar+
    cosine_part = (shares[..., None, :] * angles.cos()).sum(axis=-1)   # [..., times] Pair+
    sine_part = (shares[..., None, :] * angles.sin()).sum(axis=-1)   # [..., times] Pair+
    return cosine_part, sine_part


def probabilities(state: Pair) -> Scalar:
    """A real state occupies the six two-electron configurations with these probabilities; for a state
    in time, they are the sum over its cosine and sine parts."""
    return configurations.reverse().scalar_product(state[..., None]).squared()


def expectation(state: Pair, observable: Hamiltonian) -> Scalar:
    """A real state contributes this quadratic expectation to a real observable."""
    return state.reverse().scalar_product(observable(state))
