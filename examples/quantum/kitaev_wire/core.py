"""A gate moves the Majorana modes bound at the ends of a topological segment of a Kitaev wire.

Each site holds two real Majorana components as vectors of the two-dimensional algebra. Hopping, pairing, and
chemical potential couple them through a skew field map, the commutator action of one bivector of the Clifford
algebra of all the wire's Majoranas, written site by site in the two-dimensional algebra.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D
from numga.sparse import SparseExtensor, spdiag

ga = VGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                     # Scalar+
# x and y label the two Majorana components at each site, not spatial directions.
Majorana = ga.gatype.vector()                                   # Vector+


# --- math -----------------------------------------------------------------------------
def chain_couplings(n_sites: int, hopping: float, pairing: float) -> SparseExtensor:
    """Nearest-neighbour hopping and p-wave pairing form a skew-adjoint sparse map on the `n_sites` wire sites (`(Majorana+ <- Majorana+)[n_sites <- n_sites]`)."""
    sites = np.arange(n_sites)                                  # [n_sites]
    # Hopping and p-wave pairing add in one cross-site coupling and subtract in the other.
    # A directed bond reads one Majorana component on the next site and returns one on this site.
    bond_coupling = ((hopping + pairing) * mv.y * (mv.x | Majorana)
                     + (pairing - hopping) * mv.x * (mv.y | Majorana))   # [] Majorana+ <- Majorana+
    forward_couplings = SparseExtensor(bond_coupling.broadcast_to((n_sites - 1,)), sites[:-1], sites[1:],
                                       (n_sites, n_sites))      # (Majorana+ <- Majorana+)[n_sites <- n_sites]
    # Reversing a coupling reverses its sign so the full bond map is skew-adjoint.
    return forward_couplings - forward_couplings.adjoint()


def wire_generator(couplings: SparseExtensor, potential: Scalar) -> SparseExtensor:
    """The chemical potential turns each site's two components in `xy` and combines with the bond couplings into the wire's skew map (`[...] (Majorana+ <- Majorana+)[V <- V]`)."""
    local = -potential * mv.xy.commutator(Majorana)             # [...] (Majorana+ <- Majorana+)[V]
    return couplings + spdiag(local)


def excitation_modes(generator: SparseExtensor, n_modes: int) -> tuple[Scalar, Majorana]:
    """Squaring the skew generator against the site identity yields the `n_modes` lowest excitation energies (`[..., n_modes] Scalar+`) and their real, orthonormal quadratures (`[..., n_modes] Majorana+[V]`)."""
    # Squaring the norm of the turning map gives squared excitation energies against the site identity as metric.
    _, states = generator.adjoint()(generator).eigh(0 * generator + Majorana, n_modes)   # [..., n_modes] Majorana+[V]
    # Taking the image norm avoids a square root of roundoff-sized negative eigenvalues at zero modes.
    energies = generator[..., None](states).scalar_norm_squared().sites.sum().square_root()   # [..., n_modes] Scalar+
    return energies, states


def localized_modes(states: Majorana, site_positions: Scalar) -> Majorana:
    """Diagonalizing the site-position map within the two lowest modes recombines them into one mode at each end (`[..., 2] Majorana+[V]`)."""
    # The position along the wire between every pair of the two modes acts as a 2x2 map on the mode subspace.
    position_elements = states[..., :, None].scalar_product(site_positions * states[..., None, :]).sites.sum()   # [..., 2, 2] Scalar+
    position_map = (position_elements * Scalar).field(0, 1)     # [...] (Scalar+ <- Scalar+)[2 <- 2]
    # The eigenvectors of this map select the left and right end combinations.
    _, directions = position_map.eigh()                         # [..., 2] Scalar+[2]
    return (directions.batch() * states[..., None, :]).sum(axis=-1)


def cayley_step(couplings: SparseExtensor, state: Majorana, potential: Scalar, time_step: np.ndarray) -> Majorana:
    """One midpoint Cayley step advances `state` orthogonally across every driving speed while preserving its norm (`[...] Majorana+[V]`)."""
    half_step = wire_generator(couplings, potential) * (time_step[..., None] / 2)   # [...] (Majorana+ <- Majorana+)[V <- V]
    return (Majorana - half_step).solve(state + half_step(state))


def gate_potential(
    sites: np.ndarray, left_end: np.ndarray, right_end: np.ndarray,
    inside_potential: np.ndarray, outside_potential: float, edge_width: float,
) -> Scalar:
    """Two hyperbolic-tangent steps form a smooth chemical-potential well between `left_end` and `right_end` (`[...] Scalar+[V]`)."""
    window = (np.tanh((sites - left_end) / edge_width) - np.tanh((sites - right_end) / edge_width)) / 2
    return mv.scalar((outside_potential + (inside_potential - outside_potential) * window)[..., None]).field()


def boundary_share(modes: Majorana, state: Majorana) -> Scalar:
    """Projecting `state` onto the instantaneous pair of boundary modes gives the weight remaining in the topological subspace (`[...] Scalar+`)."""
    overlap = state[..., None].scalar_product(modes).sites.sum()   # [..., 2] Scalar+
    return overlap.squared().sum(axis=-1)


def smooth_schedule(progress: np.ndarray) -> np.ndarray:
    """A quintic transition curve interpolates from zero to one with zero velocity and acceleration at both endpoints."""
    return progress**3 * (10 + progress * (-15 + 6 * progress))


def transport_states(
    couplings: SparseExtensor, initial: Majorana, midpoint_potentials: Scalar,
    boundary_modes: Majorana, durations: np.ndarray, steps_per_frame: int,
) -> Iterator[tuple[Majorana, Scalar]]:
    """Yields the Majorana field (`[n_speeds] Majorana+[V]`) and boundary-subspace weight (`[n_speeds] Scalar+`) every `steps_per_frame` midpoint Cayley steps."""
    n_speeds = len(durations)
    time_step = durations / len(midpoint_potentials)            # [n_speeds]
    state = initial.broadcast_to((n_speeds,))                   # [n_speeds] Majorana+[V]
    yield state, boundary_share(boundary_modes[0][None], state)
    for frame, block in enumerate(midpoint_potentials.reshape(-1, steps_per_frame), start=1):
        for potential in block:
            state = cayley_step(couplings, state, potential, time_step)
        yield state, boundary_share(boundary_modes[frame][None], state)
