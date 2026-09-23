"""The constitutive extensor of a medium in Spacetime Algebra: core mathematics.

In a medium, Maxwell's equations split into the field bivector F and the excitation
bivector G, related by a linear map G = χ(F). That map is a Bivector-to-Bivector extensor,
and every medium is built from it: an observer's electric and magnetic projectors give
the isotropic dielectric, permittivity and permeability quadrics lifted through the
observer give a crystal and a ferrite, the pseudoscalar gives the axion term, and
conjugating by a Lorentz boost gives a moving medium.

Plane waves exist where the wave map a -> k · χ(k ∧ a) loses rank, which yields phase speeds,
polarizations, birefringence, and relativistic Fresnel drag.

This module contains pure mathematics: GATypes, constructors, and eigensolvers.
It never imports any plotting library.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import STA

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: R_{1,3}, t+ x- y- z-)
# ---------------------------------------------------------------------------
ctx = NumpyContext(STA)
mv = ctx.multivector

# Blade Subspaces:
V = STA.subspace.vector()
B = STA.subspace.bivector()
Spatial = STA.subspace("x y z")                 # Polarizations in temporal gauge a · t = 0

from numga.extensor.extensor import Extensor

# GATypes:
Scalar = STA.gatype.scalar()
Vector = STA.gatype(V)
Bivector = STA.gatype(B)
SpatialVector = STA.gatype(Spatial)

# Canonical Spacetime Basis & Pseudoscalar:
t, x, y, z = mv.vector(np.eye(4))
I = t ^ x ^ y ^ z


# ---------------------------------------------------------------------------
# 2. Projectors & Media Builders
# ---------------------------------------------------------------------------
def observer_projectors(observer=t) -> tuple[Extensor, Extensor]:
    """Split 6D bivectors into electric and magnetic parts for a given timelike observer (Bivector <- Bivector)."""
    electric = B.commutator(observer).wedge(observer)
    magnetic = B - electric
    return electric, magnetic


def isotropic_medium(eps: float, mu: float = 1.0, observer=t) -> Extensor:
    """Isotropic dielectric & magnetic medium: chi = eps * Pi_E + (1/mu) * Pi_B (Bivector <- Bivector)."""
    electric, magnetic = observer_projectors(observer)
    return eps * electric + (1.0 / mu) * magnetic


def permittivity_tensor(
    eps_x: float,
    eps_y: float,
    eps_z: float,
) -> Extensor:
    """Spatial permittivity quadric (Vector <- Vector)."""
    return -(
        eps_x * x * (x | V) +
        eps_y * y * (y | V) +
        eps_z * z * (z | V)
    )


def crystal_medium(
    eps_x: float,
    eps_y: float,
    eps_z: float,
    mu: float = 1.0,
    observer=t,
) -> Extensor:
    """Anisotropic dielectric crystal lifting a spatial permittivity quadric through the observer (Bivector <- Bivector)."""
    _, magnetic = observer_projectors(observer)
    permittivity = permittivity_tensor(eps_x, eps_y, eps_z)
    return permittivity(B.commutator(observer)).wedge(observer) + (1.0 / mu) * magnetic


def ferrite_medium(
    eps: float,
    mu_inv_x: float,
    mu_inv_y: float,
    mu_inv_z: float,
    observer=t,
) -> Extensor:
    """Anisotropic magnetic ferrite lifting inverse permeability through the dual field (Bivector <- Bivector)."""
    electric, _ = observer_projectors(observer)
    permeability_inv = -(
        mu_inv_x * x * (x | V) +
        mu_inv_y * y * (y | V) +
        mu_inv_z * z * (z | V)
    )  # Vector <- Vector
    return eps * electric + permeability_inv(B.dual().commutator(observer)).wedge(observer).dual_inverse()


def axion_medium(base_medium: Extensor, alpha: float) -> Extensor:
    """Topological axion electrodynamics: adds alpha * dual to the constitutive extensor (Bivector <- Bivector)."""
    return base_medium + mv.scalar([alpha]) * B.dual()


def boost_rotor(beta: float, direction=z):
    """Construct the Lorentz boost rotor for relative speed beta in given spatial direction."""
    rapidity = np.arctanh(beta)
    boost_bivector = (direction ^ t) * (rapidity / 2.0)
    return boost_bivector.exp()


def boosted_medium(base_medium: Extensor, beta: float, direction=z) -> Extensor:
    """Conjugate a rest constitutive map by a Lorentz boost: chi' = L >> chi(L << B) (Bivector <- Bivector)."""
    rotor = boost_rotor(beta, direction)
    return rotor >> base_medium(rotor << B)


# ---------------------------------------------------------------------------
# 3. Wave Maps & Dispersion Solvers
# ---------------------------------------------------------------------------
def wave_map(k, medium: Extensor) -> Extensor:
    """Construct the wave operator W_k: a -> k · chi(k ∧ a) mapping Spatial -> Vector."""
    return k.commutator(medium(k.wedge(Spatial)))


def solve_dispersion_scan(
    speeds: np.ndarray,
    medium: Extensor,
    direction=z,
) -> np.ndarray:
    """Compute smallest singular values of wave map across a range of trial phase speeds."""
    k = speeds * t + direction
    wave = wave_map(k, medium)
    return wave.svdvals()[..., -1]


def minimum_speeds(
    speeds: np.ndarray,
    svals: np.ndarray | Extensor,
    threshold: float = 2e-3,
) -> np.ndarray:
    """Detect local minima in singular value curve corresponding to allowed wave speeds."""
    left, mid, right = svals[:-2], svals[1:-1], svals[2:]
    is_dip = (mid <= left) & (mid <= right) & (mid < threshold)
    return speeds[1:-1][is_dip]


def polarization_eigenmodes(k, medium: Extensor):
    """Extract physical polarization states as right singular vectors of the wave map nullspace."""
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
    v_down = []
    v_up = []
    for beta in betas:
        moving = boosted_medium(glass, beta, direction=z)
        svals_down = solve_dispersion_scan(speeds, moving, direction=z)
        mins_down = minimum_speeds(speeds, svals_down, threshold=3e-3)
        v_down.append(mins_down[0] if len(mins_down) > 0 else np.nan)

        svals_up = solve_dispersion_scan(speeds, moving, direction=-z)
        mins_up = minimum_speeds(speeds, svals_up, threshold=3e-3)
        v_up.append(mins_up[0] if len(mins_up) > 0 else np.nan)
    return np.array(v_down), np.array(v_up)
