"""Self-consistent Anderson pseudospin dynamics of the superconducting Higgs mode in a collisionless s-wave superconductor.

Each pair of opposite-momentum orbitals is modelled as a pseudospin, represented like a physical spin as the
intrinsic patch of its turning plane (`Spin-`). Pairing with `xy` (`xy & spin`) reads the pair occupation
relative to half filling, while pairing with `yz` and `zx` reads the pairing coherence. Each pseudospin
precesses in an extrinsic bivector field (`Field+`) combining the band dispersion in `xy` with the collective
gap in `yz` and `zx`. A duality map (`Field+ <- Spin-`) maps the band-averaged coherence into that shared gap,
and pairing each field with its pseudospin (`field & spin`) gives the reduced BCS energy with energies measured
from the chemical potential in units where Planck's reduced constant is one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                           # Scalar+
# + extrinsic, - intrinsic
# A pseudospin is the intrinsic patch of the plane it turns in, paired with an extrinsic driving field.
Spin = ga.gatype.antibivector()                       # Antibivector-
Field = ga.gatype.bivector()                          # Bivector+
Pairing = ga.gatype((Field, Spin))                    # Field+ <- Spin-
SpinMap = ga.gatype((Spin, Spin))                     # Spin- <- Spin-


# --- math -----------------------------------------------------------------------------
# Pairing coherence in yz and zx produces a gap field in those same planes, while occupation in xy does not pair.
coherence = mv.yz * (mv.yz & Spin) + mv.zx * (mv.zx & Spin)  # [] Field+ <- Spin-


def ground_state(dispersion: Field, pairing: Pairing, seed_gap: Field, n_iterations: int) -> Spin:
    """Repeated alignment sets each length-1/2 pseudospin square against the self-consistent field (`Spin-[levels]`)."""
    gap = seed_gap                                    # [] Field+
    for _ in range(n_iterations):
        spins = -(dispersion - gap).dual().normalized() / 2  # Spin-[levels]
        gap = pairing(spins.sites.mean())             # [] Field+
    return spins


def precession_rate(spins: Spin, dispersion: Field, pairing: Pairing) -> Spin:
    """Each pseudospin precesses in the effective field whose gap comes from the band-averaged spin (`Spin-[levels]`)."""
    gap = pairing(spins.sites.mean())                 # [] Field+
    field = dispersion - gap                          # Field+[levels]
    return (-2 * field).commutator(spins)             # Spin-[levels]


def bcs_energy(spins: Spin, dispersion: Field, pairing: Pairing) -> Scalar:
    """Pairs each orbital's dispersion with its pseudospin and subtracts the shared pairing interaction once (`[...] Scalar+`)."""
    mean_spin = spins.sites.mean()                    # [...] Spin-
    return 2 * (dispersion & spins).sites.mean() - (pairing(mean_spin) & mean_spin)


def response_maps(dispersion: Field, pairing: Pairing, equilibrium: Spin) -> tuple[SpinMap, SpinMap]:
    """Linearizing about `equilibrium` gives the local precession map and the collective feedback map (`(Spin- <- Spin-)[levels]`)."""
    field = dispersion - pairing(equilibrium.sites.mean())  # Field+[levels]
    # Varying a disturbance turns the disturbed spin locally and shifts the shared pairing field across all levels.
    local_map = (-2 * field).commutator(Spin)         # (Spin- <- Spin-)[levels]
    feedback_map = (2 * pairing).commutator(equilibrium)  # (Spin- <- Spin-)[levels]
    return local_map, feedback_map


# --- plumbing -------------------------------------------------------------------------
def energy_levels(n_levels: int, cutoff: float) -> Field:
    """Midpoint quadrature places `n_levels` equally weighted orbitals across `[-cutoff, cutoff]` (`Field+[levels]`)."""
    energies = cutoff * (2 * (np.arange(n_levels) + 0.5) / n_levels - 1)
    return (mv.xy * energies).field()                 # Field+[levels]


def evolution(
    state: Spin, rate: Callable[[Spin], Spin], dt: float,
    n_steps: int, n_steps_per_frame: int, n_iterations: int,
) -> Iterator[Spin]:
    """Yields the initial state and every `n_steps_per_frame`-th implicit midpoint update (`[...] Spin-[levels]`)."""
    yield state
    for _ in range(n_steps // n_steps_per_frame):
        for _ in range(n_steps_per_frame):
            # Fixed-point iteration evaluates the rate halfway across the step, preserving quadratic invariants.
            end = state
            for _ in range(n_iterations):
                end = state + dt * rate((state + end) / 2)
            state = end
        yield state
