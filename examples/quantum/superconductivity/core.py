"""The superconducting Higgs mode: self-consistent Anderson pseudospins in a collisionless s-wave superconductor.

Each pair of opposite-momentum orbitals is a bivector, the plane its pair state turns in: its xy
part is occupation minus one half, and its yz and zx parts carry pairing coherence. The mean of
those coherence parts produces the gap, which feeds back into the plane every spin turns in.
Energies are measured from the chemical potential and hbar is one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Spin = ga.gatype.bivector()
Map = ga.gatype((Spin, Spin))


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Condensate:
    dispersion: Spin                         # Spin[levels]
    pairing: Map                             # [] Spin <- Spin

    def gap(self, spins: Spin) -> Spin:
        """The pairing field shared by all energy levels."""
        return self.pairing(spins.sites.mean())

    def rate(self, spins: Spin) -> Spin:
        """The instantaneous precession, with the field supplied by the spins themselves."""
        field = self.dispersion - self.gap(spins)
        return (-2 * field).commutator(spins)

    def equilibrium(self, seed: Spin, n_iterations: int) -> Spin:
        """Zero-temperature alignment against the field, iterated to self-consistency."""
        gap = seed
        for _ in range(n_iterations):
            spins = -(self.dispersion - gap).normalized() / 2
            gap = self.gap(spins)
        return spins

    def response(self, spins: Spin) -> tuple[Map, Map]:
        """Maps acting on a local disturbance and on its mean, respectively."""
        field = self.dispersion - self.gap(spins)
        # A disturbance changes both the spin being turned and its turning field.
        local = (-2 * field).commutator(Spin)
        feedback = (2 * self.pairing).commutator(spins)
        return local, feedback

    def energy(self, spins: Spin) -> Scalar:
        """Energy per pair orbital, counting the shared interaction only once."""
        mean = spins.sites.mean()
        # Planes pair with their reverses: a plane against itself is then its squared size.
        return 2 * (self.dispersion | spins.reverse()).sites.mean() - (mean | self.pairing(mean).reverse())


# --- plumbing -------------------------------------------------------------------------
def energy_levels(n_levels: int, cutoff: float) -> Spin:
    """Midpoint quadrature for a constant density of states between two energy cutoffs."""
    energies = cutoff * (2 * (np.arange(n_levels) + 0.5) / n_levels - 1)
    return (mv.xy * energies).field()


def midpoint(state: Spin, rate: Callable[[Spin], Spin], dt: float, n_iterations: int) -> Spin:
    """Implicit midpoint by fixed-point iteration; dt must resolve the fastest precession.

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
