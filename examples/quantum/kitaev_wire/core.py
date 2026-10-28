"""A gate moves the Majorana modes bound at the ends of a topological segment of a Kitaev wire.

Each site's two vectors, `mv.x` and `mv.y`, are its real Majorana components. Hopping, pairing and chemical
potential couple them through a skew field map, the commutator action of one bivector of the Clifford algebra of
all the wire's Majoranas, written site by site in the two-dimensional algebra.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D
from numga.sparse import SparseExtensor

ga = VGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                     # Scalar+
Majorana = ga.gatype.vector()                                   # Vector+


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Wire:
    bonds: SparseExtensor

    @classmethod
    def chain(cls, n_sites: int, hopping: float, pairing: float) -> Wire:
        sites = np.arange(n_sites)
        # Each directed bond reads one Majorana component on the neighbouring site.
        bond_coupling = ((hopping + pairing) * mv.y * (mv.x | Majorana)
                         + (pairing - hopping) * mv.x * (mv.y | Majorana))   # [] Majorana+ <- Majorana+
        forward_couplings = SparseExtensor(bond_coupling.broadcast_to((n_sites - 1,)), sites[:-1], sites[1:],
                                           (n_sites, n_sites))          # Majorana+[sites <- sites]
        # Reversing a coupling reverses its sign: the full generator is skew-adjoint.
        return cls(forward_couplings - forward_couplings.adjoint())

    def generator(self, potential: Scalar) -> SparseExtensor:
        """The wire's map: the turning on each site by its chemical potential, plus the couplings between
        neighbouring sites."""
        local = -potential * mv.xy.commutator(Majorana)
        return self.bonds + SparseExtensor.from_diagonal(local)

    def modes(self, generator: SparseExtensor, n_modes: int) -> tuple[Scalar, Majorana]:
        """The lowest excitation energies and their real, orthonormal quadratures."""
        # The identity on the generator's sites: the metric of a plain eigenproblem.
        identity = 0 * generator + Majorana
        _, modes = generator.adjoint()(generator).eigh(identity, n_modes)
        # The norm of the image gives the frequency without a square root of a
        # roundoff-sized negative eigenvalue at a zero mode.
        energies = generator[..., None](modes).scalar_norm_squared().sites.sum().square_root()
        return energies, modes

    def localized(self, modes: Majorana) -> Majorana:
        """The two lowest modes recombined into one mode at each end: the eigenvectors of the position along the
        wire, as a map on the two modes."""
        site_index = mv.scalar(np.arange(self.bonds.shape[0])[:, None]).field()   # Scalar+[sites]
        # The position along the wire between every pair of the two modes.
        position_elements = modes[..., :, None].scalar_product(site_index * modes[..., None, :]).sites.sum()   # [..., modes, modes] Scalar+
        position_map = (position_elements * Scalar).field(0, 1)       # [...] Scalar+[modes <- modes]
        _, directions = position_map.eigh()
        return (directions.batch() * modes[..., None, :]).sum(axis=-1)

    def step(self, state: Majorana, potential: Scalar, time_step: np.ndarray) -> Majorana:
        """A Cayley step, with the generator evaluated at the time midpoint: orthogonal for a skew generator,
        so it keeps the state's norm."""
        generator = self.generator(potential)
        # The same time step for every site of a case; each case has its own driving speed.
        half_step = generator * (time_step[..., None] / 2)
        return (Majorana - half_step).solve(state + half_step(state))


def gate(
    sites: np.ndarray, left: np.ndarray, right: np.ndarray,
    inside: np.ndarray, outside: float, width: float,
) -> Scalar:
    """A smooth chemical-potential well between two gate boundaries, in site units; the ends and the inside
    potential broadcast against the sites."""
    window = (np.tanh((sites - left) / width) - np.tanh((sites - right) / width)) / 2
    return mv.scalar((outside + (inside - outside) * window)[..., None]).field()


def boundary_share(modes: Majorana, state: Majorana) -> Scalar:
    """The state's weight in the instantaneous pair of boundary modes."""
    overlap = state[..., None].scalar_product(modes).sites.sum()   # [..., modes] Scalar+
    return overlap.squared().sum(axis=-1)


# --- plumbing -------------------------------------------------------------------------
def smooth_progress(progress: np.ndarray) -> np.ndarray:
    """A gate ramp with zero velocity and acceleration at both ends."""
    return progress**3 * (10 + progress * (-15 + 6 * progress))


def transport(
    wire: Wire, state: Majorana, potentials: Scalar, durations: np.ndarray,
    steps_per_frame: int,
) -> Iterator[Majorana]:
    """Yields the state of every case, evolved together through the midpoint gate profiles, every
    `steps_per_frame` steps."""
    time_step = durations / len(potentials)
    state = state.broadcast_to((len(durations),))                  # [cases] Majorana+[sites]
    yield state
    for block in potentials.reshape(-1, steps_per_frame):
        for potential in block:
            state = wire.step(state, potential, time_step)
        yield state
