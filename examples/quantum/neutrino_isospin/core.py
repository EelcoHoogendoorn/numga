"""Neutrino flavor oscillations, coherent and decohering, in the geometric algebra of three-dimensional space.

Neutrinos are produced and detected in flavors yet travel as mass states, which advance at slightly different
rates, so an electron neutrino oscillates into a muon neutrino and back along its flight. Over astrophysical
distances the mass states' wavepackets drift apart and the oscillation damps to an incoherent mixture; inside
the Sun, a slowly falling electron density tilts the Hamiltonian gradually enough that an outgoing neutrino
follows the local eigenstate and leaves mostly as a muon neutrino.

A two-flavor density state is modelled as `0.5 * (antiscalar + isospin)`, placing pure flavors on the unit
flavor sphere with the electron flavor at `phase_plane.dual()`. Because an isospin is an intrinsic turning-plane
patch, pairing with the phase plane `phase_plane & isospin` reads its electron-flavor projection directly. The
Hamiltonian is an extrinsic bivector giving the plane of precession, coherent propagation acts via the
commutator `-Bivector.commutator(State)`, matter shifts that bivector along `phase_plane`, and wavepacket
separation dephases the state by a Lindblad half-turn `mass_plane >> State` in the mass plane while preserving
the antiscalar trace.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Bivector = ga.gatype.bivector()                                              # Bivector+
Isospin = ga.gatype.antibivector()                                           # Antibivector-
State = ga.gatype(Isospin + ga.gatype.antiscalar())                          # Antibivector- + Antiscalar-
Rates = ga.gatype((State, State))                                            # State- <- State-
Evolution = ga.gatype((State, State))                                        # State- <- State-
Generator = ga.gatype((State, Bivector, State))                              # State- <- (Bivector+, State-)

one = mv.scalar([1])                                                         # [] Scalar+
antiscalar = one.dual()                                                      # [] Antiscalar-
# Phase plane of a pure flavor state and the plane in which vacuum mixing tilts it.
phase_plane = mv.xy                                                          # [] Bivector+
mixing_plane = mv.xz                                                         # [] Bivector+
# Commutation with a bivector has no antiscalar component; casting embeds its output in the full state space.
coherent_generator = (-Bivector.commutator(State)).cast(Generator)           # [] State- <- (Bivector+, State-)
# Pure electron and muon flavors sit at opposite poles as patches of the phase plane.
flavor_e = phase_plane.dual()                                                # [] Isospin-
flavor_mu = -flavor_e                                                        # [] Isospin-


# --- math -----------------------------------------------------------------------------
def density_state(isospin: Isospin) -> State:
    """Adding `antiscalar` to `isospin` (`[...] Isospin-`) and halving forms the trace-one density state
    (`[...] State-`)."""
    return 0.5 * (antiscalar + isospin)                                      # [...] State-


def flavor_probabilities(isospin: Isospin) -> tuple[Scalar, Scalar]:
    """Pairing `isospin` (`[...] Isospin-`) with `phase_plane` gives the electron and muon flavor probabilities
    (`[...] Scalar+`)."""
    electron_part = phase_plane & isospin                                    # [...] Scalar+
    return 0.5 * (one + electron_part), 0.5 * (one - electron_part)


def vacuum_hamiltonian(omega: float, theta: float) -> Bivector:
    """Tilting `-phase_plane` by mixing angle `theta` in `mixing_plane` gives the vacuum Hamiltonian bivector at
    rate `omega` (`[] Bivector+`)."""
    mixing = (mixing_plane * (-theta)).exp()                                 # [] Rotor+
    return omega * (mixing >> (-phase_plane))                                # [] Bivector+


def relaxation(process: Bivector) -> Rates:
    """Sandwiching a state in `process` and subtracting half the squared reverse norm on each side gives a
    trace-preserving Lindblad rate map (`[] State- <- State-`)."""
    back = process.reverse().symmetric_reverse_product()                     # [] Scalar+
    return (process >> State) - back.anticommutator(State)                   # [] State- <- State-


def step_evolution(rates: Rates, step_size: float) -> Evolution:
    """Truncating the exponential series of `rates * step_size` at fifth order gives the single-step transfer map
    (`[...] State- <- State-`)."""
    small = rates * step_size                                                # [...] State- <- State-
    term = total = State
    for order in range(1, 6):
        term = small(term) / order                                           # [...] State- <- State-
        total = total + term
    return total                                                             # [...] State- <- State-


def channel_isospins(initial_state: State, transfer: Evolution, n_steps: int) -> Iterator[Isospin]:
    """Yields the state's isospin (`[] Isospin-`) across `n_steps` applications of `transfer`."""
    density = initial_state                                                  # [] State-
    for _ in range(n_steps):
        yield 2 * density.select[1]                                          # [] Isospin-
        density = transfer(density)                                          # [] State-


def adiabatic_isospins(initial_state: State, rates: Rates, planes: Bivector, step_size: float
                       ) -> Iterator[tuple[Isospin, Bivector]]:
    """Yields the local isospin and Hamiltonian plane (`[] Isospin-`, `[] Bivector+`) as `rates` (`[n_steps]`)
    varies along the trajectory."""
    n_steps = rates.shape[0]
    density = initial_state                                                  # [] State-
    for step in range(n_steps):
        yield 2 * density.select[1], planes[step]                            # [] Isospin-, [] Bivector+
        density = step_evolution(rates[step], step_size)(density)            # [] State-
