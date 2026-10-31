"""Edge states of a graphene flake with spin-orbit coupling, in the geometric algebra of
three-dimensional space.

An electron on a graphene flake hops between neighbouring carbon atoms. Its state on each atom is a
spinor, an even multivector, and the Hamiltonian is an extensor coupling the atoms through
multivectors: minus one for each hop to a neighbour; the spin-orbit strength times the plane the hop's
path turns in, the unit bivector its two bonds span through the site between, for each hop to a second
neighbour; and the mass, signed by sublattice, on each atom. The sense of a hop's turn is the sense of
its plane, so the coupling's handedness is the geometry's, with no sign assigned to a hop. Every
coupling multiplies from the left by a scalar or by a bivector of the xy plane, so the spin along z is
kept: the spin-up spinors, `Up`, the even multivectors of the xy plane, and the spin-down ones, the
planes through z, each make up a sector of their own.

The mass alone opens a plain gap. Where the spin-orbit coupling outweighs the mass, `mass < 3 * 3**0.5
* spin_orbit`, the gap holds states on the flake's edge, and the two spins run around it in opposite
senses. Time turns a state of energy E by `(mv.xy * (-E * time)).exp()` on its right.

In the second-quantized notation of Kane and Mele, with the imaginary unit and the Pauli matrices
acting on a column of two complex amplitudes per atom, the Hamiltonian reads as
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
Scalar = ga.gatype.scalar()                                                    # Scalar+
# A spin, read out in its own space: intrinsic, like every angular momentum.
Spin = ga.gatype.antibivector()                                                # Antibivector-
# The atoms' positions, and displacements between them, are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()                                                 # Antivector-
Even = ga.gatype.even()                                                        # Even+
# The spinors with spin up along z.
Up = Even.derive.within("x y")                                                 # Even+
# The spin of the spinor one, spin up: the patch of the plane xy.
up_spin = mv.xy.dual()                                                         # [] Spin-


# --- math -----------------------------------------------------------------------------
def hamiltonian(flake: Flake, spin_orbit: float, mass: float) -> SparseExtensor:
    """The flake's Hamiltonian, coupling its atoms through multivectors."""
    n_atoms = flake.sublattice.shape[0]
    # Minus one for each hop between neighbours.
    hop = SparseExtensor(mv.scalar(-np.ones((len(flake.first), 1))), *flake.first.T, (n_atoms, n_atoms))   # Scalar+[atoms <- atoms]
    # A hop to a second neighbour runs from the second atom, through the site between, to the first; the
    # commutator of its two steps is the plane it turns in, and its sense is which way the path turns.
    near = flake.positions.batch()[flake.second[:, 0]]                         # [pairs] Point-
    far = flake.positions.batch()[flake.second[:, 1]]                          # [pairs] Point-
    turns = (near - flake.between).commutator(flake.between - far).normalized()   # [pairs] Bivector+
    turn = SparseExtensor(turns, *flake.second.T, (n_atoms, n_atoms))          # Bivector+[atoms <- atoms]
    # Plus one on the atoms of one sublattice, minus one on the other.
    stagger = SparseExtensor.from_diagonal(mv.scalar(flake.sublattice[:, None]).field())   # Scalar+[atoms <- atoms]
    return hop + turn * spin_orbit + stagger * mass                            # Up+[atoms <- atoms]


def levels(energy: SparseExtensor, sector: GAType, n_levels: int) -> tuple[Scalar, Even]:
    """The n_levels energies nearest zero within a sector of spinors, and their states, orthonormal."""
    n_atoms = energy.shape[0]
    unit = SparseExtensor.from_diagonal(mv.scalar(np.ones((n_atoms, 1))).field())   # Scalar+[atoms <- atoms]
    return (energy * sector).eigh(unit * sector, n_levels)                     # [levels] Scalar+, [levels] Even+[atoms]


def rim_share(flake: Flake, states: Even) -> Scalar:
    """How much of each state lies on the flake's rim."""
    density = states.scalar_norm_squared()                                     # [...] Scalar+[atoms]
    return (density * flake.rim).sites.sum() / density.sites.sum()             # [...] Scalar+


def spread(states: Even, energies: Scalar, start: int, spinor: Even, times: np.ndarray) -> Iterator[Spin]:
    """An electron started on one atom with the given spinor, keeping only its part in the given
    states, at each time: its spin density `psi >> up_spin`, as long as the electron's density and
    pointing along its spin."""
    weights = states.batch()[:, start].reverse().scalar_product(spinor)        # [states] Scalar+
    for time in times:
        # Each state turns on its right at the rate of its energy.
        amplitudes = weights * (mv.xy * (-energies * time)).exp()              # [states] Even+
        yield (states * amplitudes).sum(axis=0) >> up_spin                     # Spin-[atoms]


# --- plumbing -------------------------------------------------------------------------
class Flake:
    """A hexagonal graphene flake with armchair edges: the atoms' positions `Point[atoms]` and
    sublattices, plus and minus one; the pairs of neighbours `[pairs, 2]` and of second neighbours,
    both ways round, each with the lattice site between its two atoms, `Point[pairs]`; the atoms on the
    rim; and the atom in the middle of the edge facing x."""

    def __init__(self, positions: Point, sublattice: np.ndarray, first: np.ndarray, second: np.ndarray,
                 between: Point, rim: Scalar, start: int) -> None:
        self.positions = positions
        self.sublattice = sublattice
        self.first = first
        self.second = second
        self.between = between
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
    distance = np.maximum.reduce([np.abs(2 * u + v), np.abs(u + 2 * v), np.abs(v - u)])
    inside = distance <= size + 1e-9
    # Keep only atoms with two neighbours or more.
    inside[inside] = _degrees(i[inside], j[inside], s[inside], size) >= 2
    i, j, s, u, v, distance = i[inside], j[inside], s[inside], u[inside], v[inside], distance[inside]

    lookup = _lookup(i, j, s, size)
    # An atom of the first sublattice neighbours the second in its own cell and in the cells before it
    # along each direction; one of the second, the first in its own cell and the cells after it.
    sign = 1 - 2 * s
    offsets = np.array([[0, 0], [-1, 0], [0, -1]])
    neighbours = lookup(i[:, None] + sign[:, None] * offsets[:, 0], j[:, None] + sign[:, None] * offsets[:, 1], 1 - s[:, None])
    first = _pairs(neighbours)
    # A second neighbour is one cell away on the same sublattice, through the one site of the other
    # sublattice that neighbours both; at the flake's edge that site may hold no atom.
    steps = np.array([[1, 0], [-1, 1], [0, -1], [-1, 0], [1, -1], [0, 1]])
    seconds = lookup(i[:, None] + steps[:, 0], j[:, None] + steps[:, 1], s[:, None])
    second = _pairs(seconds)
    cells = np.stack([i, j], axis=-1)                                          # [atoms, 2]
    ends = cells[:, None, :] + steps                                           # [atoms, steps, 2]
    own = cells[:, None, :] + sign[:, None, None] * offsets                    # [atoms, 3, 2]
    theirs = ends[:, :, None, :] + sign[:, None, None, None] * offsets         # [atoms, steps, 3, 2]
    shared = (own[:, None, :, None, :] == theirs[:, :, None, :, :]).all(axis=-1).any(axis=-1)   # [atoms, steps, 3]
    middles = own[np.arange(len(i))[:, None], shared.argmax(axis=-1)]          # [atoms, steps, 2]

    # The lattice's two directions, the second 60 degrees from the first.
    lattice_u = mv.x.dual()                                                    # [] Point-
    lattice_v = (mv.xy * (-np.pi / 6)).exp() >> lattice_u                      # [] Point-
    positions = (lattice_u * u + lattice_v * v).field()                        # Point-[atoms]
    # The site between each pair of second neighbours, a third of the way along the cell's long diagonal
    # on the second sublattice.
    middle = (middles + (1 - s[:, None, None]) / 3)[seconds >= 0]              # [pairs, 2]
    between = lattice_u * middle[:, 0] + lattice_v * middle[:, 1]              # [pairs] Point-
    facing = np.flatnonzero(2 * u + v == (2 * u + v).max())
    start = facing[np.argmin(np.abs(v[facing]))]
    # One on the atoms of the rim, zero elsewhere.
    edge = mv.scalar((distance > (1 - rim) * size)[:, None]).field()           # Scalar+[atoms]
    return Flake(positions, sign.astype(float), first, second, between, edge, start)


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
    atoms, neighbours = np.nonzero(partners >= 0)
    return np.stack([atoms, partners[atoms, neighbours]], axis=-1)
