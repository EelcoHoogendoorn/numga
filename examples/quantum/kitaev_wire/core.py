"""Majorana boundary modes and their transport in a gated Kitaev wire.

The two vector components at each site are real Majorana amplitudes. Hopping,
pairing and chemical potential couple them through a skew field map. Its action
is the vector action of a bivector in the Clifford algebra of all Majoranas.
The local two-dimensional algebra stores the cells of that action.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor

ga = Algebra("x+y+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Majorana = ga.gatype.vector()
Map = ga.gatype((Majorana, Majorana))


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Wire:
    bonds: SparseExtensor
    identity: SparseExtensor

    @classmethod
    def chain(cls, sites: int, hopping: float, pairing: float) -> Wire:
        index = np.arange(sites)
        # Each directed cell reads one Majorana component on the neighbouring site.
        cell = ((hopping + pairing) * mv.y * (mv.x | Majorana)
                + (pairing - hopping) * mv.x * (mv.y | Majorana))
        forward = SparseExtensor(cell.broadcast_to((sites - 1,)), index[:-1], index[1:], (sites, sites))
        # Reversing a coupling reverses its sign: the full generator is skew-adjoint.
        bonds = forward - forward.adjoint()
        unit = SparseExtensor.from_diagonal(mv.scalar(np.ones((sites, 1))).field())
        identity = unit * Majorana
        return cls(bonds, identity)

    def generator(self, potential: Scalar) -> SparseExtensor:
        """On-site turning plus the couplings between neighbouring sites."""
        local = -potential * mv.xy.commutator(Majorana)
        return self.bonds + SparseExtensor.from_diagonal(local)

    def modes(self, generator: SparseExtensor, count: int) -> tuple[Scalar, Majorana]:
        """The lowest excitation energies and their real, orthonormal quadratures."""
        _, modes = generator.adjoint()(generator).eigh(self.identity, count)
        # The norm of the image gives the frequency without a square root of a
        # roundoff-sized negative eigenvalue at a zero mode.
        action = SparseExtensor(generator.cells[..., None, :], generator.rows,
                                generator.columns, generator.shape)
        energies = action(modes).scalar_norm_squared().batch().sum(axis=-1).square_root()
        return energies, modes

    def localized(self, modes: Majorana) -> Majorana:
        """Resolve the lowest two-dimensional mode space by mean site index."""
        coordinates = stack([mv.x, mv.y])
        embedder = (modes * (coordinates | Majorana)).sum(axis=-1)
        # Pull the site-index observable back to the two mode coordinates.
        index = mv.scalar(np.arange(self.identity.shape[0])[:, None]).field()
        position = embedder.adjoint()(index * embedder)
        _, directions = position.eigh()
        return embedder[..., None](directions)

    def step(self, state: Majorana, potential: Scalar, dt: np.ndarray) -> Majorana:
        """Cayley step with the generator evaluated at the time midpoint."""
        generator = self.generator(potential)
        # The same dt for every cell of a case; each case has its own driving speed.
        half_step = SparseExtensor(generator.cells * (dt[..., None] / 2),
                                   generator.rows, generator.columns, generator.shape)
        return (self.identity - half_step).solve(state + half_step(state))


def gate(
    sites: np.ndarray, left: np.ndarray, right: np.ndarray,
    inside: np.ndarray, outside: float, width: float,
) -> Scalar:
    """A smooth chemical-potential well between two gate boundaries, in site units."""
    window = (np.tanh((sites - left[..., None]) / width)
              - np.tanh((sites - right[..., None]) / width)) / 2
    return mv.scalar((outside + (inside[..., None] - outside) * window)[..., None]).field()


def boundary_share(modes: Majorana, state: Majorana) -> Scalar:
    """The state's weight in the instantaneous pair of boundary modes."""
    overlap = state[..., None].scalar_product(modes).batch().sum(axis=-1)
    return overlap.squared().sum(axis=-1)


# --- plumbing -------------------------------------------------------------------------
def smooth_progress(progress: np.ndarray) -> np.ndarray:
    """A gate ramp with zero velocity and acceleration at both ends."""
    return progress**3 * (10 + progress * (-15 + 6 * progress))


def transport(
    wire: Wire, state: Majorana, potentials: Scalar, durations: np.ndarray,
    stride: int,
) -> Iterator[Majorana]:
    """Evolve cases together through midpoint gate profiles, yielding sampled fields."""
    steps = len(potentials)
    dt = durations / steps
    state = state.broadcast_to((len(durations),))
    yield state
    for index, potential in enumerate(potentials):
        state = wire.step(state, potential, dt)
        if (index + 1) % stride == 0:
            yield state
