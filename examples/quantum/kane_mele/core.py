"""Edge states of a graphene flake with spin-orbit coupling, in the geometric algebra of
three-dimensional space.

An electron on a graphene flake hops between neighbouring carbon atoms. Its state on each atom is
modelled as a spinor, an even multivector, and the Hamiltonian couples the atoms through
multivectors: minus one for each hop to a neighbour, the spin-orbit strength times the unit
bivector of the plane a second-neighbour hop turns in through the site between, and the staggered
mass signed by sublattice on each atom. Because the sense of a hop's turn is the sense of its
plane, the coupling's handedness follows directly from the flake's geometry without manual hop
signs. Every coupling multiplies from the left by a scalar or by a bivector of the xy plane, so
spin along z is conserved: the spin-up spinors `Up` (even multivectors in the xy plane) and the
spin-down spinors `Down` (planes through z) each form an invariant sector.

The staggered mass alone opens a trivial bulk gap. Where spin-orbit coupling outweighs the mass,
the bulk gap holds edge states around the flake's rim and the two spins circulate in opposite
senses.

Energies are measured in units of the nearest-neighbour hop, lengths in lattice constants, and
times in inverse hop energies.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.algebras import VGA3D as ga
from numga.gatype import GAType
from numga.sparse import Sparse, SparseExtensor, spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
# Spin density on each atom is an intrinsic turning-plane patch, like every angular momentum.
Spin = ga.gatype.antibivector()                                                # Antibivector-
# Atom positions and bond displacements have intrinsic orientation.
Point = ga.gatype.antivector()                                                 # Antivector-
Even = ga.gatype.even()                                                        # Even+
# Spin-up spinors lie in the xy plane; spin-down spinors are bivector planes touching z.
Up = Even.derive.within(Algebra("x+y+"))                                       # Even+
Down = Even.derive.touching(Algebra("z+"))                                     # Even+
# Reference spin patch of the unit spinor 1 (spin up in the xy plane).
up_spin = mv.xy.dual()                                                         # [] Spin-


# --- math -----------------------------------------------------------------------------
def hamiltonian(bonds: Topology, next_bonds: Topology, positions: Point, between: Point,
                sublattice: Scalar, spin_orbit: float, mass: float) -> SparseExtensor:
    """Nearest-neighbour hops, second-neighbour turn planes, and the staggered sublattice mass sum
    into the flake's sparse multivector Hamiltonian (`Up+[V <- V]`)."""
    ends = bonds.incidence[1, 0]                                               # [2] Scalar+[E <- V]
    forward_hop = -(ends[0].adjugate() * ends[1])                              # Scalar+[V <- V]
    # A second-neighbour hop steps from the second endpoint through the intermediate lattice site
    # to the first; the commutator of its two steps gives the oriented plane of the turn.
    next_ends = next_bonds.incidence[1, 0]                                     # [2] Scalar+[E_next <- V]
    near, far = next_ends * positions                                          # Point-[E_next], Point-[E_next]
    turns = (near - between).commutator(between - far).normalized()            # Bivector+[E_next]
    forward_turn = next_ends[0].adjugate() * spdiag(turns) * next_ends[1]      # Bivector+[V <- V]
    return (forward_hop + forward_hop.adjugate()
            + (forward_turn + ~forward_turn) * spin_orbit + sublattice * mass)   # Up+[V <- V]


def sector_levels(energy: SparseExtensor, sector: GAType, n_levels: int) -> tuple[Scalar, Even]:
    """The `n_levels` eigenpairs nearest zero energy within `sector` give energies (`[n_levels] Scalar+`)
    and orthonormal spinor fields (`[n_levels] Even+[V]`)."""
    V = energy.shape[0]
    unit = Sparse.selection(np.arange(V), V) * sector                          # (Even+ <- Even+)[V <- V]
    return (energy * sector).eigh(unit, n_levels)                              # [n_levels] Scalar+, [n_levels] Even+[V]


def rim_share(rim: Scalar, states: Even) -> Scalar:
    """The fraction of each state's norm squared lying on the rim atoms (`[...] Scalar+`)."""
    density = states.scalar_norm_squared()                                     # [...] Scalar+[V]
    return (density * rim).sites.sum() / density.sites.sum()                   # [...] Scalar+


def spin_evolution(states: Even, energies: Scalar, start_site: Scalar, spinor: Even,
                   times: np.ndarray) -> Iterator[Spin]:
    """Yields the atomic spin density (`Spin-[V]`) at each step in `times` for an electron started
    on `start_site` with `spinor` and projected onto `states`."""
    weights = (states * start_site).sites.sum().reverse().scalar_product(spinor)   # [...] Scalar+
    for time in times:
        # Each eigenstate turns on its right in the xy plane at the rate of its energy.
        phases = (mv.xy * (-energies * time)).exp()                            # [...] Even+
        state = (states * (weights * phases)).sum(axis=0)                      # Even+[V]
        yield state >> up_spin                                                 # Spin-[V]
