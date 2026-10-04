"""Neutrino flavor isospin and open quantum systems in 3D Euclidean Geometric Algebra (VGA3D).

Neutrinos are produced and detected as flavor eigenstates, but propagate as mass eigenstates
with distinct dispersion relations. Under plane-wave propagation, the density operator is an element
of the self-reverse subspace `State: 1 x y z`, with isospin vector `P = 2 * rho.select[1]`.

1. State and Superoperators:
   - `State = ga.gatype.self_reverse()` (density operator rho = 0.5 * (1 + P)).
   - `Rates = ga.gatype((State, State))` (Liouvillian superoperator: State <- State).
   - `Evolution = ga.gatype((State, State))` (quantum transfer map: State <- State).

2. Coherent and Dissipative Generators:
   - Coherent rotation in Hamiltonian bivector plane b:
     `turning = - b.commutator(State).cast(Rates)`.
   - Dissipative wavepacket separation / dephasing along mass eigenstate axis l:
     `dephasing = gamma * ((l >> State) - (l * l).anticommutator(State))`.

3. Finite Transfer Maps:
   - Propagation over distance dx is an extensor exponential:
     `transfer = exp(rates * dx)`.
   - Composing transfers across inhomogeneous matter layers is extensor map composition.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()

# Density operators and superoperators (capitalized GATypes):
State = ga.gatype.self_reverse()                                             # 1 x y z
Rates = ga.gatype((State, State))                                            # State <- State
Evolution = ga.gatype((State, State))                                        # State <- State

# Values and basis blades (lowercase):
one = mv.scalar([1.0])
flavor_e = mv.z                                                              # [] Vector (+z: electron neutrino)
flavor_mu = -mv.z                                                            # [] Vector (-z: muon neutrino)

phase_plane = mv.x ^ mv.y                                                    # [] Bivector (quantum phase accumulation)
transition_plane = mv.y ^ mv.z                                               # [] Bivector (flavor conversion)
mixing_plane = mv.x ^ mv.z                                                   # [] Bivector (mass-flavor mixing)


# --- math -----------------------------------------------------------------------------
def state(bloch: Vector) -> State:
    """Construct density operator state: 0.5 * (1 + P)."""
    return 0.5 * (one + bloch)


def isospin(rho: State) -> Vector:
    """Extract isospin vector P from density operator: 2 * rho.select[1]."""
    return 2.0 * rho.select[1]


def flavor_probabilities(p: Vector) -> tuple[Scalar, Scalar]:
    """Electron and muon flavor probabilities as exact scalar projections onto the flavor axis."""
    p_z = p | flavor_e
    return 0.5 * (one + p_z), 0.5 * (one - p_z)


def mixing_rotor(theta: float) -> Rotor:
    """Rotor turning the flavor basis into the mass eigenstate basis by mixing angle theta."""
    return (mixing_plane * (-theta)).exp()


def vacuum_hamiltonian(omega: float, theta: float) -> Bivector:
    """Vacuum Hamiltonian bivector: phase plane tilted by mixing angle theta."""
    rotor = mixing_rotor(theta)
    return omega * (rotor >> (-phase_plane))


def matter_hamiltonian(b_vac: Bivector, potential: float) -> Bivector:
    """Effective Hamiltonian bivector in matter: phase plane shifted by electron potential."""
    return b_vac + potential * phase_plane


def resonance_potential(omega: float, theta: float) -> float:
    """Matter potential v at which the MSW resonance occurs: `omega * np.cos(2 * theta)`."""
    return float(omega * np.cos(2.0 * theta))


def coherent_generator(plane: Bivector) -> Rates:
    """Unitary rotation generator: - plane.commutator(State)."""
    return - plane.commutator(State).cast(Rates)


def relaxation(process: Vector) -> Rates:
    """Lindblad dissipator: (l >> State) - (l * l).anticommutator(State)."""
    back = process.reverse().symmetric_reverse_product()                  # [] State
    return (process >> State) - back.anticommutator(State)                 # State <- State


def dephasing_generator(axis: Vector, gamma: float) -> Rates:
    """Wavepacket decoherence / dephasing along direction axis at rate gamma."""
    return gamma * relaxation(axis)


def evolution(rates: Rates, dx: float) -> Evolution:
    """Quantum transfer map over distance dx: exp(rates * dx) as composed extensor map."""
    small = rates * dx                                                     # State <- State
    term = total = State
    for order in range(1, 6):
        term = small(term) / order                                         # State <- State
        total = total + term
    return total                                                           # State <- State


def evolve_channel(initial_state: State, transfer: Evolution, steps: int) -> Iterator[Vector]:
    """Yield isospin vector of state propagating under constant transfer map."""
    rho = initial_state
    for _ in range(steps):
        yield isospin(rho)
        rho = transfer(rho)


def adiabatic_channel(
    initial_state: State,
    rates_profile: Iterable[tuple[Rates, Bivector]],
    dx: float,
) -> Iterator[tuple[Vector, Bivector]]:
    """Yield isospin vector and Hamiltonian plane under varying generator profile."""
    rho = initial_state
    for rates, plane in rates_profile:
        yield isospin(rho), plane
        step_map = evolution(rates, dx)
        rho = step_map(rho)
