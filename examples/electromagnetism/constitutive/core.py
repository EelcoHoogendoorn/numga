"""Materials map field bivectors to excitation bivectors: `G = medium(F)`.

Electric and magnetic plane weights describe glass, crystals and ferrites;
duality adds an axion response, and a Lorentz boost sets a medium in motion.
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


def isotropic_medium(eps: float, mu: float = 1.0, observer=t) -> Extensor:
    """Isotropic dielectric & magnetic medium: `eps * electric + (1 / mu) * magnetic` over the
    observer's projectors (Bivector <- Bivector)."""
    electric, magnetic = observer_projectors(observer)
    return eps * electric + (1.0 / mu) * magnetic


def crystal_medium(
    eps_x: float,
    eps_y: float,
    eps_z: float,
    mu: float = 1.0,
    observer=t,
) -> Extensor:
    """Anisotropic dielectric response along electric bivector planes (Bivector <- Bivector)."""
    _, magnetic = observer_projectors(observer)
    permittivity = np.array([eps_x, eps_y, eps_z])
    planes = spatial_axes ^ observer
    return (permittivity * planes * (planes | Bivector)).sum(axis=0) + magnetic / mu


def ferrite_medium(
    eps: float,
    mu_inv_x: float,
    mu_inv_y: float,
    mu_inv_z: float,
    observer=t,
) -> Extensor:
    """Anisotropic magnetic response along magnetic bivector planes (Bivector <- Bivector)."""
    electric, _ = observer_projectors(observer)
    permeability_inv = np.array([mu_inv_x, mu_inv_y, mu_inv_z])
    planes = (spatial_axes ^ observer).dual()
    return eps * electric - (permeability_inv * planes * (planes | Bivector)).sum(axis=0)


def axion_medium(base_medium: Extensor, alpha: float) -> Extensor:
    """Topological axion electrodynamics: adds alpha * dual to the constitutive extensor (Bivector <- Bivector)."""
    return base_medium + alpha * B.dual()


def boost_rotor(beta: float, direction=z):
    """Construct the Lorentz boost rotor for relative speed beta in given spatial direction."""
    rapidity = np.arctanh(beta)
    boost_bivector = (direction ^ t) * (rapidity / 2.0)
    return boost_bivector.exp()


def boosted_medium(base_medium: Extensor, beta: float, direction=z) -> Extensor:
    """Conjugate a rest constitutive map by a Lorentz boost: `rotor >> base_medium(rotor << B)` (Bivector <- Bivector)."""
    rotor = boost_rotor(beta, direction)
    return rotor >> base_medium(rotor << B)


# ---------------------------------------------------------------------------
# 3. Wave Maps & Dispersion Solvers
# ---------------------------------------------------------------------------
def wave_map(k: Vector, medium: Extensor) -> Extensor:
    """Source-free Maxwell residual, with separate vector and trivector grades (Odd <- Bivector)."""
    return (k | medium) + (k ^ Bivector)


def solve_dispersion_scan(
    speeds: np.ndarray,
    medium: Extensor,
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


def field_eigenmodes(k: Vector, medium: Extensor) -> Bivector:
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
