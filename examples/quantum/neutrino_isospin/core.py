"""Neutrino flavor oscillations, coherent and decohering, in the geometric algebra of three-dimensional space.

Neutrinos are produced and detected in flavors but travel as mass states, which run at slightly different
rates, so a neutrino born an electron neutrino turns into a muon neutrino and back as it travels. Over
astrophysical distances the mass states' wavepackets drift apart and the oscillation fades into a fixed
mixture; inside the Sun, the electrons' density tilts the turning slowly enough that a neutrino follows it and
leaves as a muon neutrino.

A two-flavor state is modelled as half the antiscalar, its trace, plus its isospin, `0.5 * (antiscalar +
isospin)`, a point of the flavor sphere with the electron flavor at z. The isospin is intrinsic, like every
spin: the electron flavor is the patch of the phase plane, `phase_plane.dual()`, and a flavor's part
of an isospin is its pairing with that plane, `phase_plane & isospin`. The Hamiltonian is an extrinsic
bivector, the plane the state turns in, and the change of a state under it is the map
`-plane.commutator(State)`; matter shifts that plane, and the drift
of the wavepackets apart dephases the state by the half turn in the mass states' plane. The evolution over a
step is the exponential of the rates, one map from states to states.

In the matrix notation of neutrino oscillations the Hamiltonian on the flavor states reads as
$H = \\frac{\\Delta m^2}{4E} \\begin{pmatrix} -\\cos 2\\theta & \\sin 2\\theta \\\\ \\sin 2\\theta & \\cos 2\\theta \\end{pmatrix}
+ \\begin{pmatrix} V & 0 \\\\ 0 & 0 \\end{pmatrix}$, with the matter potential $V = \\sqrt2 \\, G_F N_e$, and the two-flavor
survival probability in vacuum as $P_{ee} = 1 - \\sin^2 2\\theta \\, \\sin^2(\\Delta m^2 L / 4E)$.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Bivector = ga.gatype.bivector()                                              # Bivector+
# An isospin, intrinsic; a two-flavor state, half the antiscalar, its trace, plus its isospin; its rates of change,
# and its evolution over a step.
Isospin = ga.gatype.vector().derive.flipped_orientation                      # Vector-
State = ga.gatype.odd().derive.flipped_orientation                           # Vector- + Antiscalar-
Rates = ga.gatype((State, State))                                            # State- <- State-
Evolution = ga.gatype((State, State))                                        # State- <- State-
Generator = ga.gatype((State, Bivector, State))                              # State- <- (Bivector+, State-)

one = mv.scalar([1.0])                                                       # [] Scalar+
antiscalar = mv.antiscalar([1.0])                                            # [] Antiscalar-
# The plane a flavor state's phase turns in, and the plane the mixing angle tilts it in.
phase_plane = mv.xy                                                          # [] Bivector+
mixing_plane = mv.xz                                                         # [] Bivector+
# The rates of a state under a plane: minus their commutator, one map with the plane left open. The commutator
# has no part in the antiscalar; the cast embeds it among the states.
coherent_generator = (-Bivector.commutator(State)).cast(Generator)           # [] State- <- (Bivector+, State-)
# The flavors at the poles of the flavor sphere: the electron neutrino along z, the patch of the phase plane, the
# muon neutrino against it.
flavor_e = phase_plane.dual()                                                # [] Isospin-
flavor_mu = -flavor_e                                                        # [] Isospin-


# --- math -----------------------------------------------------------------------------
def state(isospin: Isospin) -> State:
    """The state with the given isospin."""
    return 0.5 * (antiscalar + isospin)                                      # [...] State-


def isospin(rho: State) -> Isospin:
    """The isospin of a state, twice its vector part."""
    return 2.0 * rho.select[1]                                               # [...] Isospin-


def flavor_probabilities(isospin: Isospin) -> tuple[Scalar, Scalar]:
    """The probabilities of the electron and the muon flavor, from the isospin's part along the electron flavor: its
    pairing with the phase plane."""
    electron_part = phase_plane & isospin                                    # [...] Scalar+
    return 0.5 * (one + electron_part), 0.5 * (one - electron_part)


def vacuum_hamiltonian(omega: float, theta: float) -> Bivector:
    """The plane a state turns in through vacuum: the phase plane tilted by the mixing angle, at the rate omega."""
    # The turn from the flavor states to the mass states, by the mixing angle.
    mixing = (mixing_plane * (-theta)).exp()                                 # [] Rotor+
    return omega * (mixing >> (-phase_plane))                                # [] Bivector+


def relaxation(process: Bivector) -> Rates:
    """The rates of a state under a relaxation process: its sandwich of the state, less the process's reverse
    times itself, half on either side."""
    back = process.reverse().symmetric_reverse_product()                     # [] Scalar+
    return (process >> State) - back.anticommutator(State)                   # [] State- <- State-


def evolution(rates: Rates, dx: float) -> Evolution:
    """What becomes of each state over a distance dx: the exponential series of the rates to fifth order."""
    small = rates * dx                                                       # [...] State- <- State-
    term = total = State
    for order in range(1, 6):
        term = small(term) / order                                           # [...] State- <- State-
        total = total + term
    return total                                                             # [...] State- <- State-


def evolve_channel(initial_state: State, transfer: Evolution, n_steps: int) -> Iterator[Isospin]:
    """The isospin at each step of a state carried by one transfer map."""
    rho = initial_state
    for _ in range(n_steps):
        yield isospin(rho)                                                   # [] Isospin-
        rho = transfer(rho)


def adiabatic_channel(initial_state: State, rates: Rates, planes: Bivector, dx: float) -> Iterator[tuple[Isospin, Bivector]]:
    """The isospin and the Hamiltonian's plane at each step of a state carried through rates `[steps]` that change
    along the way."""
    rho = initial_state
    for step in range(rates.shape[0]):
        yield isospin(rho), planes[step]                                     # [] Isospin-, Bivector+
        rho = evolution(rates[step], dx)(rho)
