"""Self-consistent Anderson pseudospins in a collisionless s-wave superconductor.

Each vector describes a pair of opposite-momentum orbitals. Its z component is
occupation minus one half; its x and y components carry pairing coherence. The
transverse mean produces the gap, which feeds back into every spin's turning field.
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
Spin = ga.gatype.vector()
Map = ga.gatype((Spin, Spin))


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Condensate:
    dispersion: Spin                         # [levels] Spin
    pairing: Map                             # [] Spin <- Spin

    def gap(self, spins: Spin) -> Spin:
        """The pairing field shared by all energy levels."""
        return self.pairing(spins.mean(axis=-1, keepdims=True))

    def rate(self, spins: Spin) -> Spin:
        """The instantaneous precession, with the field supplied by the spins themselves."""
        field = self.dispersion - self.gap(spins)
        return (-2 * mv.xyz * field).commutator(spins)

    def equilibrium(self, seed: Spin, iterations: int) -> Spin:
        """Zero-temperature alignment against the field, iterated to self-consistency."""
        gap = seed
        for _ in range(iterations):
            spins = -(self.dispersion - gap).normalized() / 2
            gap = self.gap(spins)
        return spins

    def response(self, spins: Spin) -> tuple[Map, Map]:
        """Maps acting on a local disturbance and on its mean, respectively."""
        field = self.dispersion - self.gap(spins)
        # A disturbance changes both the spin being turned and its turning field.
        local = (-2 * mv.xyz * field).commutator(Spin)
        feedback = (2 * mv.xyz * self.pairing).commutator(spins)
        return local, feedback

    def energy(self, spins: Spin) -> Scalar:
        """Energy per pair orbital, counting the shared interaction only once."""
        mean = spins.mean(axis=-1)
        return 2 * (self.dispersion | spins).mean(axis=-1) - (mean | self.pairing(mean))


# --- plumbing -------------------------------------------------------------------------
def energy_levels(count: int, cutoff: float) -> Spin:
    """Midpoint quadrature for a constant density of states between two energy cutoffs."""
    energies = cutoff * (2 * (np.arange(count) + 0.5) / count - 1)
    return mv.z * energies


def midpoint(state: Spin, rate: Callable[[Spin], Spin], dt: float, iterations: int) -> Spin:
    """Implicit midpoint by fixed-point iteration; dt must resolve the fastest precession.

    At convergence this preserves quadratic invariants, including spin lengths and
    the reduced BCS energy. The iteration count controls the implicit solve accuracy.
    """
    end = state
    for _ in range(iterations):
        end = state + dt * rate((state + end) / 2)
    return end


def evolution(
    state: Spin, rate: Callable[[Spin], Spin], dt: float,
    steps: int, stride: int, iterations: int,
) -> Iterator[Spin]:
    """The initial state and every stride-th implicit midpoint step."""
    yield state
    for step in range(steps):
        state = midpoint(state, rate, dt, iterations)
        if (step + 1) % stride == 0:
            yield state
