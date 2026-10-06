"""Two electrons hopping between two sites, with an energy cost for sharing a site.

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
Scalar = ga.gatype.scalar()
Orbital = ga.gatype.vector()
Pair = ga.gatype.bivector()
Complement = ga.gatype.antivector()
Hopping = ga.gatype((Orbital, Orbital))
Hamiltonian = ga.gatype((Pair, Pair))

# Orbitals within each site are ordered by spin, up then down.
LEFT = stack([mv.a, mv.b])                                      # [spins] Orbital
ORBITALS = stack([mv.a, mv.b, mv.c, mv.d])                      # [orbitals] Orbital
# A field along z that differs between the sites raises up and lowers down on the left, the other way
# round on the right.
FIELD_SIGNS = np.array([1.0, -1.0, -1.0, 1.0])
# An asymmetry between the sites lowers the left one and raises the right, for both spins alike.
SITE_SIGNS = np.array([-1.0, -1.0, 1.0, 1.0])
RIGHT = stack([mv.c, mv.d])                                     # [spins] Orbital
CONFIGURATIONS = stack([mv.ab, mv.ac, mv.ad, mv.bc, mv.bd, mv.cd]) # [configurations] Pair
DOUBLES = stack([mv.ab, mv.cd])                                 # [sites] Pair
SINGLET = (mv.ad - mv.bc) / np.sqrt(2)
TRIPLETS = stack([mv.ac, (mv.ad + mv.bc) / np.sqrt(2), mv.bd])    # [spin_projections] Pair
SYMMETRIC_DOUBLE = (mv.ab + mv.cd) / np.sqrt(2)

# Each dyad measures a configuration and returns that same configuration.
DOUBLE_OCCUPANCY = (DOUBLES * DOUBLES.reverse().scalar_product(Pair)).sum(axis=0)
SINGLET_WEIGHT = SINGLET * SINGLET.reverse().scalar_product(Pair)
# Empty and doubly occupied sites carry no spin. Singlets have correlation -3/4,
# triplets +1/4, measured in hbar squared.
SPIN_CORRELATION = (Pair - DOUBLE_OCCUPANCY - 4 * SINGLET_WEIGHT) / 4


# --- math -----------------------------------------------------------------------------
def hopping(strength: float) -> Hopping:
    """Move an electron to the other site without changing its spin."""
    return -strength * (LEFT * (RIGHT | Orbital) + RIGHT * (LEFT | Orbital)).sum(axis=0)


def lift(one_electron: Hopping) -> Hamiltonian:
    """A one-electron map acting on either electron of a pair in turn. The trace pairs the orbital
    removed from the state with its replacement:
    lift(single)(first ^ second) == single(first) ^ second + first ^ single(second)."""
    return (one_electron ^ (Pair & Complement)).trace(1, 3)   # [] Pair <- Pair


def hamiltonian(strength: float, repulsion: np.ndarray) -> Hamiltonian:
    """Let either electron hop, and charge repulsion when both occupy the same site."""
    return lift(hopping(strength)) + repulsion * DOUBLE_OCCUPANCY   # [...] Pair <- Pair


def field_difference() -> Hamiltonian:
    """A unit field along z differing between the sites, by half on each, lifted to pairs."""
    return lift(0.5 * (FIELD_SIGNS * (ORBITALS * (ORBITALS | Orbital))).sum(axis=0))   # [] Pair <- Pair


def block_rate(strength: float, repulsion: np.ndarray) -> np.ndarray:
    """The rate of the singlet block: the shifted Hamiltonian applied twice on the singlet and the
    symmetric double occupancy is its square."""
    return np.sqrt((repulsion / 2) ** 2 + 4 * strength**2)


def shifted_singlet(strength: float, repulsion: np.ndarray) -> Pair:
    """The Hamiltonian less half the repulsion, on the singlet."""
    return hamiltonian(strength, repulsion)(SINGLET) - repulsion / 2 * SINGLET   # [...] Pair


def ground(strength: float, repulsion: np.ndarray) -> Pair:
    """The lowest state: the singlet projected onto the lower state of its block, by the rate less
    the shifted Hamiltonian, and normalized."""
    lower = block_rate(strength, repulsion) * SINGLET - shifted_singlet(strength, repulsion)   # [...] Pair
    return lower / lower.scalar_norm_squared().square_root()   # [...] Pair


def exchange(strength: float, repulsion: np.ndarray, times: np.ndarray) -> tuple[Pair, Pair]:
    """Spin up on the left and spin down on the right, half singlet and half triplet of zero spin,
    after the given times: the cosine and the sine of the Hamiltonian times time, applied to it. The
    triplet, at zero energy, sits in the cosine part. On the singlet's block the Hamiltonian is half
    the repulsion plus a shifted part whose cosine and sine act on the singlet as the cosine and sine
    of the rate; the half repulsion mixes the two by its own cosine and sine."""
    rate = block_rate(strength, repulsion)
    along = np.cos(rate * times) * SINGLET                      # [...] Pair
    across = np.sin(rate * times) / rate * shifted_singlet(strength, repulsion)   # [...] Pair
    half = 0.5 * repulsion * times
    cosine_part = (TRIPLETS[1] + np.cos(half) * along - np.sin(half) * across) / np.sqrt(2)   # [...] Pair
    sine_part = (np.sin(half) * along + np.cos(half) * across) / np.sqrt(2)                  # [...] Pair
    return cosine_part, sine_part


def site_difference() -> Hamiltonian:
    """A unit asymmetry between the sites, by half on each, lifted to pairs: half the right site's
    occupation less the left's."""
    return lift(0.5 * (SITE_SIGNS * (ORBITALS * (ORBITALS | Orbital))).sum(axis=0))   # [] Pair <- Pair


def turn(energies: Scalar, modes: Pair, start: Pair, times: np.ndarray) -> tuple[Pair, Pair]:
    """The cosine and the sine of a Hamiltonian times time applied to a start, from the Hamiltonian's
    states: each keeps its share of the start and turns by its own energy."""
    shares = modes * modes.reverse().scalar_product(start)     # [..., modes] Pair
    angles = energies[..., None, :] * times[..., :, None]      # [..., times, modes] Scalar
    return (shares[..., None, :] * angles.cos()).sum(axis=-1), (shares[..., None, :] * angles.sin()).sum(axis=-1)


def probabilities(state: Pair) -> Scalar:
    """Occupation probabilities of a real state in the six two-electron configurations; for a state
    in time, the sum over its cosine and sine parts."""
    return CONFIGURATIONS.reverse().scalar_product(state[..., None]).squared()


def expectation(state: Pair, observable: Hamiltonian) -> Scalar:
    """Quadratic expectation contribution of a real state to a real observable."""
    return state.reverse().scalar_product(observable(state))
