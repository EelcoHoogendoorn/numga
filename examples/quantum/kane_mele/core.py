"""Edge states of a graphene flake with spin-orbit coupling, in the geometric algebra of
three-dimensional space.

An electron on a graphene flake hops between neighbouring carbon atoms. Its state on each atom is a
spinor, an even multivector, and the Hamiltonian is a sparse extensor coupling the atoms through
multivectors: minus one for each hop to a neighbour; the spin-orbit strength times `mv.xy`, signed by
the sense the path turns through the atom between, for each hop to a second neighbour; and the mass,
signed by sublattice, on each atom. Every coupling multiplies from the left by a scalar or by `mv.xy`,
so the spin along z is kept: the spin-up spinors, `Up`, the even multivectors of the xy plane, and the
spin-down ones, `Down`, the planes through z, each make up a sector of their own.

The mass alone opens a plain gap. Where the spin-orbit coupling outweighs the mass, `mass < 3 * 3**0.5
* spin_orbit`, the gap holds states on the flake's edge, and the two spins run around it in opposite
senses. Time turns a state of energy E by `(mv.xy * (-E * time)).exp()` on its right.

In the notation of Kane and Mele, with the imaginary unit and the Pauli matrices acting on a column of
two complex amplitudes per atom, the Hamiltonian reads as
$H = -\\sum_{\\langle ij\\rangle} c_i^\\dagger c_j + i\\lambda\\sum_{\\langle\\langle ij\\rangle\\rangle} \\nu_{ij}\\,
c_i^\\dagger s_z c_j + m\\sum_i \\xi_i\\, c_i^\\dagger c_i$.

Energies are in units of the hop between neighbours, lengths in lattice constants, and times in the
inverse of the hop's energy.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga
from numga.gatype import GAType
from numga.sparse import SparseExtensor

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Even = ga.gatype.even()
# The spinors with spin up along z, and those with spin down.
Up = ga.gatype(ga.subspace("1 xy"))
Down = ga.gatype(ga.subspace("yz zx"))


# --- math -----------------------------------------------------------------------------
def hamiltonian(flake: Flake, spin_orbit: float, mass: float) -> SparseExtensor:
    """The flake's Hamiltonian, coupling its atoms through multivectors."""
    atoms = flake.sublattice.shape[0]
    # Minus one for each hop between neighbours.
    hop = SparseExtensor(mv.scalar(-np.ones((len(flake.first), 1))), *flake.first.T, (atoms, atoms))   # [atoms, atoms] Scalar
    # For each hop to a second neighbour, the xy plane, signed by the sense the path turns.
    turn = SparseExtensor(mv.xy * flake.senses, *flake.second.T, (atoms, atoms))   # [atoms, atoms] Bivector
    # Plus one on the atoms of one sublattice, minus one on the other.
    stagger = SparseExtensor.from_diagonal(mv.scalar(flake.sublattice[:, None]).field())   # [atoms, atoms] Scalar
    return hop + turn * spin_orbit + stagger * mass                            # [atoms, atoms] Up


def levels(energy: SparseExtensor, sector: GAType, count: int) -> tuple[Scalar, Even]:
    """The count energies nearest zero within a sector of spinors, and their states, orthonormal."""
    atoms = energy.shape[0]
    unit = SparseExtensor.from_diagonal(mv.scalar(np.ones((atoms, 1))).field())   # [atoms, atoms] Scalar
    return (energy * sector).eigh(unit * sector, count)                       # [count] Scalar, [count] sector[atoms]


def rim_share(flake: Flake, states: Even) -> Scalar:
    """How much of each state lies on the flake's rim."""
    density = states.scalar_norm_squared()                                     # [...] Scalar[atoms]
    return (density * flake.rim).batch().sum(axis=-1) / density.batch().sum(axis=-1)   # [...] Scalar


def spread(states: Even, energies: Scalar, start: int, spinor: Even, times: np.ndarray) -> Iterator[Vector]:
    """An electron started on one atom with the given spinor, keeping only its part in the given
    states, at each time: its spin density `psi >> mv.z`, as long as the electron's density and
    pointing along its spin."""
    weights = states.batch()[:, start].reverse().scalar_product(spinor)        # [states] Scalar
    for time in times:
        # Each state turns on its right at the rate of its energy.
        turned = weights * (mv.xy * (-energies * time)).exp()                  # [states] Even
        yield (states * turned).sum(axis=0) >> mv.z                            # Vector[atoms]


# --- plumbing -------------------------------------------------------------------------
class Flake:
    """A hexagonal graphene flake with armchair edges: the atoms' positions `Vector[atoms]` and
    sublattices, plus and minus one; the pairs of neighbours `[pairs, 2]` and of second neighbours,
    each with the sense its path turns, both ways round; the atoms on the rim; and the atom in the
    middle of the edge facing x."""

    def __init__(self, positions: Vector, sublattice: np.ndarray, first: np.ndarray, second: np.ndarray,
                 senses: np.ndarray, rim: Scalar, start: int) -> None:
        self.positions = positions
        self.sublattice = sublattice
        self.first = first
        self.second = second
        self.senses = senses
        self.rim = rim
        self.start = start


def flake(size: int, rim: float) -> Flake:
    """The atoms of the honeycomb whose projections on the three lattice directions, in units of half
    a lattice constant, are at most size; the rim is their outer share `rim` by that measure."""
    # Cell (i, j) of the lattice spanned by two directions 60 degrees apart, and sublattice s, a
    # third of the way along the cell's long diagonal for the second.
    span = np.arange(-size, size + 1)
    i, j, s = (index.ravel() for index in np.meshgrid(span, span, [0, 1], indexing="ij"))
    u, v = i + s / 3, j + s / 3
    reach = np.maximum.reduce([np.abs(2 * u + v), np.abs(u + 2 * v), np.abs(v - u)])
    inside = reach <= size + 1e-9
    # Keep only atoms with two neighbours or more.
    inside[inside] = _degrees(i[inside], j[inside], s[inside], size) >= 2
    i, j, s, u, v, reach = i[inside], j[inside], s[inside], u[inside], v[inside], reach[inside]

    lookup = _lookup(i, j, s, size)
    # An atom of the first sublattice neighbours the second in its own cell and in the cells before it
    # along each direction; one of the second, the first in its own cell and the cells after it.
    sign = 1 - 2 * s
    offsets = np.array([[0, 0], [-1, 0], [0, -1]])
    neighbours = lookup(i[:, None] + sign[:, None] * offsets[:, 0], j[:, None] + sign[:, None] * offsets[:, 1], 1 - s[:, None])
    first = _pairs(neighbours)
    # A second neighbour is one cell away on the same sublattice; the path through the atom between
    # turns one way for three of the six and the other way for the rest, opposite on the two
    # sublattices.
    steps = np.array([[1, 0], [-1, 1], [0, -1], [-1, 0], [1, -1], [0, 1]])
    turns = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    seconds = lookup(i[:, None] + steps[:, 0], j[:, None] + steps[:, 1], s[:, None])
    second = _pairs(seconds)
    senses = (sign[:, None] * turns)[seconds >= 0]

    a = mv.x
    b = ((mv.x ^ mv.y) * (-np.pi / 6)).exp() >> a                             # [] Vector, 60 degrees from a
    positions = (a * u + b * v).field()                                        # Vector[atoms]
    facing = np.flatnonzero(2 * u + v == (2 * u + v).max())
    start = facing[np.argmin(np.abs(v[facing]))]
    # One on the atoms of the rim, zero elsewhere.
    edge = mv.scalar((reach > (1 - rim) * size)[:, None]).field()               # Scalar[atoms]
    return Flake(positions, sign.astype(float), first, second, senses, edge, start)


def _lookup(i: np.ndarray, j: np.ndarray, s: np.ndarray, size: int):
    """A function from cells and sublattices to the index of the atom there, or -1 where there is none."""
    table = -np.ones((2 * size + 3, 2 * size + 3, 2), dtype=int)
    table[i + size + 1, j + size + 1, s] = np.arange(len(i))
    return lambda ii, jj, ss: table[np.clip(ii + size + 1, 0, 2 * size + 2), np.clip(jj + size + 1, 0, 2 * size + 2), ss]


def _degrees(i: np.ndarray, j: np.ndarray, s: np.ndarray, size: int) -> np.ndarray:
    """How many neighbours each atom has."""
    sign = 1 - 2 * s
    offsets = np.array([[0, 0], [-1, 0], [0, -1]])
    lookup = _lookup(i, j, s, size)
    return (lookup(i[:, None] + sign[:, None] * offsets[:, 0], j[:, None] + sign[:, None] * offsets[:, 1], 1 - s[:, None]) >= 0).sum(axis=1)


def _pairs(partners: np.ndarray) -> np.ndarray:
    """The pairs (atom, partner) `[pairs, 2]` where a partner exists, from partners `[atoms, k]`."""
    atoms, slots = np.nonzero(partners >= 0)
    return np.stack([atoms, partners[atoms, slots]], axis=-1)
