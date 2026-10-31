"""The superconducting Higgs mode: self-consistent Anderson pseudospins in a collisionless s-wave superconductor.

Each pair of opposite-momentum orbitals carries a pseudospin, the state the pair is in, modelled like a spin
as the intrinsic patch of the plane it turns in: its reading by the plane xy, `mv.xy & spin`, is the
occupation minus one half, and its readings by the planes yz and zx are the pairing coherence. The field a
pseudospin turns in is an extrinsic bivector: the dispersion in the plane xy, and the gap in the planes yz
and zx. The pairing interaction is a correlation from pseudospins to fields, `Field <- Spin`,
which reads the mean coherence by pairing, `mv.yz & spin`, into the gap; the gap feeds back into the field
every pseudospin turns in, and the energy pairs each field with its pseudospin, `field & spin`. Energies
are measured from the chemical potential and hbar is one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                           # Scalar+
# A pseudospin, the state of a pair, is the intrinsic patch of the plane it turns in; that field is extrinsic.
Spin = ga.gatype.antibivector()                       # Antibivector-
Field = ga.gatype.bivector()                          # Bivector+
Pairing = ga.gatype((Field, Spin))                    # Field+ <- Spin-
Map = ga.gatype((Spin, Spin))                         # Spin- <- Spin-


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Condensate:
    dispersion: Field                        # Field+[levels]
    pairing: Pairing                         # [] Field+ <- Spin-

    def gap(self, spins: Spin) -> Field:
        """The gap is the pairing field shared by all energy levels, read from the mean pseudospin."""
        return self.pairing(spins.sites.mean())

    def rate(self, spins: Spin) -> Spin:
        """The pseudospins change at the rate of their precession in a field whose gap they supply themselves."""
        field = self.dispersion - self.gap(spins)
        return (-2 * field).commutator(spins)

    def equilibrium(self, seed: Field, n_iterations: int) -> Spin:
        """At zero temperature each pseudospin, of length one half, is square to its field's plane and set against
        it; repeated alignment makes the field and the gap the pseudospins produce consistent."""
        gap = seed
        for _ in range(n_iterations):
            spins = -(self.dispersion - gap).dual().normalized() / 2
            gap = self.gap(spins)
        return spins

    def response(self, spins: Spin) -> tuple[Map, Map]:
        """The linearized motion is two maps, one acting on a local disturbance and one on its mean."""
        field = self.dispersion - self.gap(spins)
        # A disturbance changes both the spin being turned and its turning field.
        local = (-2 * field).commutator(Spin)              # (Spin- <- Spin-)[levels]
        feedback = (2 * self.pairing).commutator(spins)    # (Spin- <- Spin-)[levels]
        return local, feedback

    def energy(self, spins: Spin) -> Scalar:
        """The energy per pair orbital pairs each field with its pseudospin, counting the shared interaction
        only once."""
        mean = spins.sites.mean()
        return 2 * (self.dispersion & spins).sites.mean() - (self.pairing(mean) & mean)


# --- plumbing -------------------------------------------------------------------------
def energy_levels(n_levels: int, cutoff: float) -> Field:
    """The dispersion, by midpoint quadrature, of a constant density of states between two energy cutoffs."""
    energies = cutoff * (2 * (np.arange(n_levels) + 0.5) / n_levels - 1)
    return (mv.xy * energies).field()


def midpoint(state: Spin, rate: Callable[[Spin], Spin], dt: float, n_iterations: int) -> Spin:
    """The implicit midpoint step, solved by fixed-point iteration; dt must resolve the fastest precession.

    At convergence this preserves quadratic invariants, including spin lengths and
    the reduced BCS energy. The iteration count controls the implicit solve accuracy.
    """
    end = state
    for _ in range(n_iterations):
        end = state + dt * rate((state + end) / 2)
    return end


def evolution(
    state: Spin, rate: Callable[[Spin], Spin], dt: float,
    n_steps: int, steps_per_frame: int, n_iterations: int,
) -> Iterator[Spin]:
    """The initial state and every steps_per_frame-th implicit midpoint step."""
    yield state
    for _ in range(n_steps // steps_per_frame):
        for _ in range(steps_per_frame):
            state = midpoint(state, rate, dt, n_iterations)
        yield state
