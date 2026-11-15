"""The Dirac electron in the spacetime algebra models the electron's state at a point as an even
multivector built from a density, an angle beta, and a Lorentz rotor.

Sandwiching spacetime vectors with the spinor (`psi >> Vector`) applies its Lorentz rotor and scales
by the density, carrying the observer's frame to the electron's own. The image of the observer's
time direction is the electron's charge current, while the patch of the sandwiched spin plane
(`(psi >> spin_plane).dual()`) is its intrinsic spin plane. Because the pseudoscalar anticommutes
with vectors, the angle beta drops out of vector sandwiches and acts instead on bivectors, where
`psi >> Bivector` differs from the frame's outermorphism by multiplication by
`(pseudoscalar * beta).exp() / density`.

For a plane wave of fixed spatial momentum, the Dirac equation reduces to a linear Hamiltonian map
on spinors whose eigenvalues are plus and minus the relativistic energy, each fourfold degenerate.
Right multiplication by the spin plane (`Spinor * spin_plane`) squares to minus one and commutes
with the Hamiltonian, playing the algebraic role of the quantum phase generator. Positive-energy
states carry `beta == 0` and negative-energy states carry `beta == np.pi`; mixing the two makes the
charge current circulate near the speed of light at twice the energy, the trembling motion known as
Zitterbewegung.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Vector = ga.gatype.vector()                         # Vector+
Bivector = ga.gatype.bivector()                     # Bivector+
Spinor = ga.gatype.even()                           # Spinor+
# Currents, velocities, positions, and sliced spin axes are spacetime directions, intrinsic antivectors.
Antivector = ga.gatype.antivector()                 # Antivector-
# Intrinsic turning-plane patch of the electron's spin in spacetime.
Spin = ga.gatype.antibivector()                     # Spin-
# Spatial wavevectors in the observer's rest slice, perpendicular to the time covector.
Spatial = Vector.derive.within(Algebra("x+y+z+"))   # Vector+
# Linear map carrying spacetime covectors from the observer's frame to the electron's scaled frame.
Frame = ga.gatype((Vector, Vector))                 # Vector+ <- Vector+
# Linear map isolating the spinor's action on bivectors beyond its frame outermorphism.
Duality = ga.gatype((Bivector, Bivector))           # Bivector+ <- Bivector+
# Plane-wave Dirac operator acting linearly on spinors.
Hamiltonian = ga.gatype((Spinor, Spinor))           # Spinor+ <- Spinor+

# Observer time direction pairing to +1 with the time slice; the minus offsets the negative square of t.
time = -mv.t.dual()                                 # [] Antivector-
pseudoscalar = mv.xyzt                              # [] Pseudoscalar+
# Rest-frame spin plane whose right multiplication on spinors squares to minus one.
spin_plane = mv.xy                                  # [] Bivector+
phase_unit = Spinor * spin_plane                    # [] Spinor+ <- Spinor+


def dirac_spinor(density: float, beta: float, rotor: Spinor) -> Spinor:
    """A density, a chiral mixing angle `beta`, and a Lorentz `rotor` combine into an even spacetime
    spinor (`[] Spinor+`)."""
    chiral_factor = (pseudoscalar * (beta / 2)).exp()   # [] Spinor+
    return chiral_factor * rotor * np.sqrt(density)     # [] Spinor+


def bivector_duality(psi: Spinor) -> Duality:
    """Composing the spinor's bivector sandwich with the inverse outermorphism of its vector frame
    isolates multiplication by `(pseudoscalar * beta).exp() / density` (`[...] Bivector+ <- Bivector+`)."""
    frame = psi >> Vector                               # [...] Vector+ <- Vector+
    extended = frame.outermorphism(Bivector)            # [...] Bivector+ <- Bivector+
    own = psi >> Bivector                               # [...] Bivector+ <- Bivector+
    return own(extended.inverse())                      # [...] Bivector+ <- Bivector+


def current_velocity(psi: Spinor) -> Antivector:
    """Dividing the electron's spacetime current by the charge density across the observer's slice
    and subtracting the observer's time direction leaves the spatial drift velocity (`[...] Antivector-`)."""
    flow = psi >> time                                  # [...] Antivector-
    charge_density = mv.t & flow                        # [...] Scalar+
    return flow / charge_density - time                 # [...] Antivector-


def spin_axis(psi: Spinor) -> Antivector:
    """Meeting the intrinsic spin patch `(psi >> spin_plane).dual()` with the observer's slice cuts
    out the spatial spin axis (`[...] Antivector-`)."""
    spin_patch = (psi >> spin_plane).dual()             # [...] Spin-
    return spin_patch ^ mv.t                            # [...] Antivector-


def plane_wave_hamiltonian(momentum: Vector, mass: float) -> Hamiltonian:
    """Left multiplication by the relative momentum and slice reflection scaled by `mass` form the
    plane-wave Dirac Hamiltonian on spinors (`[...] Spinor+ <- Spinor+`)."""
    return (momentum * mv.t) * Spinor + mass * (mv.t * Spinor * mv.t.inverse())


def plane_wave_energy(momentum: Vector, mass: float) -> Scalar:
    """Mass and spatial momentum determine the positive relativistic energy of a free plane wave
    (`[...] Scalar+`)."""
    return (mass**2 + (momentum | momentum)).square_root()


def mixed_spinors(momentum: Vector, mass: float, mixtures: np.ndarray) -> Spinor:
    """Superposing unit-density positive-energy and negative-energy eigenstates of `momentum`
    produces initial spinors (`[n_mixtures] Spinor+`) with negative-energy weights `mixtures`."""
    hamiltonian = plane_wave_hamiltonian(momentum, mass)                        # [] Spinor+ <- Spinor+
    energy = plane_wave_energy(momentum, mass)                                  # [] Scalar+
    identity = mv.rotor()                                                       # [] Spinor+
    # Project unit-spinor seeds onto positive and negative energy, turning in opposite senses.
    electron = 0.5 * (identity + hamiltonian(identity) / energy)                # [] Spinor+
    positron = 0.5 * (mv.tx - hamiltonian(mv.tx) / energy)                      # [] Spinor+
    electron = electron / electron.scalar_norm_squared().square_root()          # [] Spinor+
    positron = positron / (-positron.scalar_norm_squared()).square_root()       # [] Spinor+
    return electron * np.sqrt(1 - mixtures) + positron * np.sqrt(mixtures)      # [n_mixtures] Spinor+


def trembling_states(
    momentum: Vector, mass: float, initial: Spinor, dt: float, n_frames: int, n_substeps: int
) -> Iterator[tuple[Antivector, Spinor]]:
    """Yields `(trail, spinor)` states (`[n_mixtures, ...] Antivector-`, `[n_mixtures] Spinor+`) as
    positive-energy and negative-energy spinor components counter-rotate in `spin_plane`."""
    hamiltonian = plane_wave_hamiltonian(momentum, mass)                        # [] Spinor+ <- Spinor+
    energy = plane_wave_energy(momentum, mass)                                  # [] Scalar+
    positive = 0.5 * (initial + hamiltonian(initial) / energy)                  # [n_mixtures] Spinor+
    negative = initial - positive                                               # [n_mixtures] Spinor+
    step_turn = (spin_plane * (energy * dt)).exp()                              # [] Spinor+
    position = 0 * (initial >> time)                                            # [n_mixtures] Antivector-
    positions: list[Antivector] = []
    for _ in range(n_frames):
        for _ in range(n_substeps):
            psi = positive + negative                                           # [n_mixtures] Spinor+
            positions.append(position)
            position = position + current_velocity(psi) * dt                    # [n_mixtures] Antivector-
            positive = positive * step_turn.reverse()                           # [n_mixtures] Spinor+
            negative = negative * step_turn                                     # [n_mixtures] Spinor+
        yield stack(positions, axis=-1), psi
