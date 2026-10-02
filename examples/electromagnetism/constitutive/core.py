"""Materials map field bivectors to excitation antibivectors: `excitation = medium(field)`.

Electric and magnetic plane weights describe glass, crystals and ferrites;
the identity adds an axion response, and a Lorentz boost sets a medium in motion.
Pairing with an open field gives the constitutive bilinear form. The field and
excitation also determine a stress-energy map from vectors to current antivectors.
Plane waves are bivectors annihilated by both source-free Maxwell equations.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: Algebra("t+x-y-z-"))
# ---------------------------------------------------------------------------
ctx = NumpyContext(STA)
mv = ctx.multivector

# Blade Subspaces:
V = STA.subspace.vector()
B = STA.subspace.bivector()

from numga.extensor.extensor import Extensor

# GATypes:
Scalar = STA.gatype.scalar()
Vector = STA.gatype(V)
Bivector = STA.gatype(B)
Antibivector = STA.gatype.antibivector()
Antivector = STA.gatype.antivector()
Constitutive = STA.gatype((Antibivector, Bivector))
StressEnergy = STA.gatype((Antivector, Vector))
WaveMap = STA.gatype((STA.gatype.odd(), Bivector))
Modes = tuple[np.ndarray, Bivector]

# Canonical Spacetime Basis & Pseudoscalar:
t, x, y, z = mv.vector(np.eye(4))
spatial_axes = stack((x, y, z))
I = mv.txyz


# ---------------------------------------------------------------------------
# 2. Projectors & Media Builders
# ---------------------------------------------------------------------------
def observer_projectors(observer=t) -> tuple[Extensor, Extensor]:
    """Split bivectors into electric and magnetic parts for a unit timelike observer (Bivector <- Bivector)."""
    electric = (Bivector - (observer >> Bivector)) / 2
    magnetic = B - electric
    return electric, magnetic


def isotropic_medium(eps: float, mu: float = 1.0, observer: Vector = t) -> Constitutive:
    """Weight the observer's electric and magnetic planes, then apply the Hodge map."""
    electric, magnetic = observer_projectors(observer)
    return (eps * electric + (1.0 / mu) * magnetic).dual()


def crystal_medium(
    eps_x: float,
    eps_y: float,
    eps_z: float,
    mu: float = 1.0,
    observer: Vector = t,
) -> Constitutive:
    """Anisotropic electric plane response (Antibivector <- Bivector)."""
    _, magnetic = observer_projectors(observer)
    permittivity = np.array([eps_x, eps_y, eps_z])
    planes = spatial_axes ^ observer
    return ((permittivity * planes * (planes | Bivector)).sum(axis=0) + magnetic / mu).dual()


def ferrite_medium(
    eps: float,
    mu_inv_x: float,
    mu_inv_y: float,
    mu_inv_z: float,
    observer: Vector = t,
) -> Constitutive:
    """Anisotropic magnetic plane response (Antibivector <- Bivector)."""
    electric, _ = observer_projectors(observer)
    permeability_inv = np.array([mu_inv_x, mu_inv_y, mu_inv_z])
    planes = (spatial_axes ^ observer).dual()
    return (eps * electric - (permeability_inv * planes * (planes | Bivector)).sum(axis=0)).dual()


def axion_medium(base_medium: Constitutive, alpha: float) -> Constitutive:
    """Add the axion response, proportional to the identity on field planes."""
    return base_medium + alpha * Bivector


def boost_rotor(beta: float, direction=z):
    """Construct the Lorentz boost rotor for relative speed beta in given spatial direction."""
    rapidity = np.arctanh(beta)
    boost_bivector = (direction ^ t) * (rapidity / 2.0)
    return boost_bivector.exp()


def boosted_medium(base_medium: Constitutive, beta: float, direction=z) -> Constitutive:
    """Conjugate a rest constitutive map by a Lorentz boost (Antibivector <- Bivector)."""
    rotor = boost_rotor(beta, direction)
    return rotor >> base_medium(rotor << B)


def stress_energy(field: Bivector, excitation: Antibivector) -> StressEnergy:
    """Electromagnetic stress-energy current (Antivector <- Vector).

    An observer selects a current; `observer & current(observer)` gives its energy
    density for a unit timelike observer. In matter this is the electromagnetic
    Minkowski current, excluding the material's own stress-energy.
    """
    lagrangian = (field & excitation) / 2
    return ((Vector | field) ^ excitation) + lagrangian * Vector.dual()


# ---------------------------------------------------------------------------
# 3. Wave Maps & Dispersion Solvers
# ---------------------------------------------------------------------------
def wave_map(k: Vector, medium: Constitutive) -> WaveMap:
    """Both exterior Maxwell residuals, packed into separate grades (Odd <- Bivector)."""
    # Dualizing the excitation residual keeps the two equations from cancelling.
    return (k ^ Bivector) + (k ^ medium).dual()


def solve_dispersion_scan(
    speeds: np.ndarray,
    medium: Constitutive,
    direction=z,
) -> Scalar:
    """Compute smallest singular values of wave map across a range of trial phase speeds."""
    k = speeds * t + direction
    wave = wave_map(k, medium)
    return wave.svdvals()[..., -1]


def minimum_speeds(
    speeds: np.ndarray,
    svals: Scalar,
    threshold: float = 2e-3,
) -> np.ndarray:
    """Detect local minima in singular value curve corresponding to allowed wave speeds."""
    left, mid, right = svals[:-2], svals[1:-1], svals[2:]
    is_dip = (mid <= left) & (mid <= right) & (mid < threshold)
    return speeds[1:-1][is_dip]


def field_eigenmodes(k: Vector, medium: Constitutive) -> Bivector:
    """Field bivectors in the wave map's nullspace, batched over wave vectors."""
    w = wave_map(k, medium)
    _, _, vh = w.svd()
    return vh[..., -1]


def fresnel_drag_velocities(
    eps: float,
    mu: float,
    betas: np.ndarray,
    speeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute downstream and upstream phase speeds across a range of medium boost speeds."""
    glass = isotropic_medium(eps, mu)
    moving = boosted_medium(glass, betas, direction=z)
    down = solve_dispersion_scan(speeds[:, None], moving, direction=z)
    up = solve_dispersion_scan(speeds[:, None], moving, direction=-z)
    return speeds[down.argmin(axis=0)], speeds[up.argmin(axis=0)]
