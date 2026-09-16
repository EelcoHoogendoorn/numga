"""Maxwell Stress-Energy Extensor, Relativistic Dust Tensor, and Lorentz Force in Spacetime Algebra (STA).

Mathematical formulation of energy-momentum extensors in STA Cl(1, 3):
1. Spacetime Algebra R^{1,3}: (+---) signature with generators gamma_0^2 = +1, gamma_i^2 = -1.
2. Maxwell Energy-Momentum Extensor:
       T_EM(v) = -0.5 * F * v * F = 0.5 * (F >> v)
   with exact tracelessness Tr(T_EM) == 0 and Poynting flux S = E x B. The same map is the
   iterated Lorentz force minus the Lagrangian pressure, F . (F . v) - 0.5 <F^2>_0 v, and for
   a pure travelling wave it collapses to a single null dyad, E^2 k (k . v): massless dust.
3. Relativistic Dust Energy-Momentum Extensor (pressureless matter):
       T_dust(v) = rho_0 * (u . v) * u
   with rank 1, zero spatial stress in rest frame, and invariant trace Tr(T_dust) == rho_0.
4. Particle Cloud Extensor, the same rank-1 quadric summed over a batch of four-velocities:
       T_cloud(v) = sum_i m_i * (u_i . v) * u_i
   whose eigenstructure is an ideal fluid: the timelike eigenvalue is the rest-frame energy
   density, the spatial eigenvalues are minus the pressure, and Tr(T_cloud) = sum_i m_i.
   A cloud of null rays is the massless limit: exactly traceless, so p = rho / 3 exactly.
5. Ideal Relativistic Fluid Extensor:
       T_fluid(v) = (rho_0 + p) * (u . v) * u - p * v
   with Tr(T_fluid) = rho_0 - 3 * p, matched against the cloud.
6. Lorentz 4-Force Density on a current: f = J x F, the commutator.
7. Lorentz Covariance via Rotor Sandwich:
       T' = L >> T(L << V)
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebra import Algebra

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: R_{1,3}, t+ x- y- z-)
# ---------------------------------------------------------------------------
STA = Algebra("t+x-y-z-")
ctx = NumpyContext(STA)
mv = ctx.multivector

V = STA.subspace.vector()
Vector = STA.gatype.vector()
StressEnergy = STA.gatype((V, V))            # momentum flux <= observer


# ---------------------------------------------------------------------------
# 2. Read-outs and Samples
# ---------------------------------------------------------------------------
def normal_stress(T: StressEnergy, n: Vector) -> float:
    """Normal traction on a face with unit spatial normal n (n^2 = -1): sigma(n) = n^{-1} . T(n)."""
    return float((n.inverse() | T(n)).kernel.item())


def principal_values(T: StressEnergy) -> np.ndarray:
    """Eigenvalues of the mixed tensor T^mu_nu, sorted ascending."""
    return np.sort(np.linalg.eigvals(T.kernel).real)


def isotropic_cloud(n: int, speed: float, rng: np.random.Generator) -> Vector:
    """n four-velocities at one speed, directions uniform on the sphere."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    return mv.vector(np.concatenate([np.full((n, 1), gamma), gamma * speed * directions], axis=1))


def null_cloud(n: int, rng: np.random.Generator) -> Vector:
    """n null rays k = t + direction, directions uniform on the sphere."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return mv.vector(np.concatenate([np.ones((n, 1)), directions], axis=1))


# ---------------------------------------------------------------------------
# 3. Demonstration & Algebraic Verifications
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(0)
    # The rest observer with full vector support, so the tensors built on it are endomorphisms.
    t = mv.vector([1.0, 0.0, 0.0, 0.0])

    # -----------------------------------------------------------------------
    # A. Maxwell Energy-Momentum Extensor: plane wave E = Ex x, B = By y along +z
    # -----------------------------------------------------------------------
    Ex, By = 2.0, 2.0
    F = mv.tx * Ex + mv.zx * By
    T_em: StressEnergy = 0.5 * (F >> V)

    # Observer at rest: energy density and Poynting flux
    flux = T_em(t)
    energy = float((t | flux).kernel.item())
    poynting = flux - t * energy
    np.testing.assert_allclose(energy, 0.5 * (Ex**2 + By**2), atol=1e-14)
    np.testing.assert_allclose(poynting.kernel, [0.0, 0.0, 0.0, Ex * By], atol=1e-14)

    # No stress across the field, radiation pressure along the propagation direction
    np.testing.assert_allclose([normal_stress(T_em, mv.x), normal_stress(T_em, mv.y)], 0.0, atol=1e-14)
    np.testing.assert_allclose(normal_stress(T_em, mv.z), -0.5 * (Ex**2 + By**2), atol=1e-14)
    np.testing.assert_allclose(T_em.trace().kernel, 0.0, atol=1e-14)

    # A pure travelling wave has |E| = |B|, and its tensor is one null dyad along the ray
    # k = t + z: the massless counterpart of the dust tensor below.
    k = t + mv.z
    np.testing.assert_allclose(T_em.kernel, (Ex**2 * k * (k | V)).kernel, atol=1e-14)

    # For any field the same map is the Lorentz force generator F . v iterated, minus the
    # Lagrangian <F^2>_0 = E^2 - B^2 as an isotropic pressure.
    F_any = mv.tx * 0.7 + mv.ty * -0.4 + mv.tz * 1.1 + mv.yz * 0.3 + mv.zx * -0.8 + mv.xy * 0.5
    lagrangian = (F_any * F_any).select.scalar()
    np.testing.assert_allclose(
        (0.5 * (F_any >> V)).kernel,
        (F_any.commutator(F_any.commutator(V)) - 0.5 * lagrangian * V).kernel,
        atol=1e-14,
    )

    # -----------------------------------------------------------------------
    # B. Dust Energy-Momentum Extensor: at rest, then boosted
    # -----------------------------------------------------------------------
    rho = 4.0
    T_dust: StressEnergy = rho * t * (t | V)
    np.testing.assert_allclose([normal_stress(T_dust, n) for n in (mv.x, mv.y, mv.z)], 0.0, atol=1e-14)
    np.testing.assert_allclose(T_dust.trace().kernel, rho, atol=1e-14)

    # The moving observer is the rest observer boosted; gamma and beta follow from the rapidity
    zeta = np.arctanh(0.6)
    boost = (mv.zt * (zeta / 2.0)).exp()
    u = boost >> t
    gamma, beta = np.cosh(zeta), np.tanh(zeta)
    T_moving: StressEnergy = rho * u * (u | V)

    # Observed energy density scales with gamma^2, dynamic ram pressure develops along z
    np.testing.assert_allclose(float((t | T_moving(t)).kernel.item()), rho * gamma**2, atol=1e-14)
    np.testing.assert_allclose(normal_stress(T_moving, mv.z), -rho * gamma**2 * beta**2, atol=1e-14)
    np.testing.assert_allclose(T_moving.trace().kernel, rho, atol=1e-14)

    # -----------------------------------------------------------------------
    # C. Particle Cloud: the dust quadric summed over four-velocities is a fluid
    # -----------------------------------------------------------------------
    # Each particle contributes m u (u . v). Summed over an isotropic cloud, the tensor's
    # eigenstructure is that of an ideal fluid at rest: the timelike eigenvalue is the energy
    # density, the three spatial eigenvalues are minus the pressure, and for particles all at
    # one speed the pressure is rho v^2 / 3. The trace is the summed rest mass, exactly.
    speed = 0.5
    u_cloud = isotropic_cloud(2000, speed, rng)
    mass = np.full((2000, 1), 1.0 / 2000)
    T_cloud: StressEnergy = (u_cloud * (u_cloud | V) * mv.scalar(mass)).sum()

    *stresses, energy_cloud = principal_values(T_cloud)
    pressure = -np.mean(stresses)
    np.testing.assert_allclose(T_cloud.trace().kernel, mass.sum(), atol=1e-14)
    np.testing.assert_allclose(pressure, energy_cloud * speed**2 / 3.0, rtol=0.05)
    np.testing.assert_allclose(stresses, -pressure, rtol=0.05)

    # A cloud of null rays is the massless limit. Each k is null, so every dyad is traceless
    # and the trace vanishes to roundoff rather than statistically: p = rho / 3 exactly.
    rays = null_cloud(2000, rng)
    T_light: StressEnergy = (rays * (rays | V) * mv.scalar(mass)).sum()
    *stresses_light, energy_light = principal_values(T_light)
    np.testing.assert_allclose(T_light.trace().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(-np.mean(stresses_light), energy_light / 3.0, atol=1e-12)

    # -----------------------------------------------------------------------
    # D. Ideal Fluid Extensor matches the cloud
    # -----------------------------------------------------------------------
    T_fluid: StressEnergy = (energy_cloud + pressure) * t * (t | V) - pressure * V
    np.testing.assert_allclose(T_fluid.trace().kernel, energy_cloud - 3.0 * pressure, atol=1e-14)
    np.testing.assert_allclose(T_fluid.kernel, T_cloud.kernel, atol=0.02)

    # Conformal radiation fluid: p = rho / 3 gives Tr(T) == 0
    T_radiation: StressEnergy = (rho + rho / 3.0) * t * (t | V) - (rho / 3.0) * V
    np.testing.assert_allclose(T_radiation.trace().kernel, 0.0, atol=1e-14)

    # -----------------------------------------------------------------------
    # E. Lorentz 4-Force Density on a current: f = J x F
    # -----------------------------------------------------------------------
    rho_q = 0.5
    J = u * rho_q
    force = J.commutator(F)
    np.testing.assert_allclose(float((force | mv.x.inverse()).kernel.item()), rho_q * gamma * (Ex - beta * By), atol=1e-14)

    # -----------------------------------------------------------------------
    # F. Lorentz Rotor Sandwich Covariance: T' = L >> T(L << V)
    # -----------------------------------------------------------------------
    np.testing.assert_allclose((boost >> T_em(boost << V)).kernel, (0.5 * ((boost >> F) >> V)).kernel, atol=1e-14)
    np.testing.assert_allclose((boost >> T_dust(boost << V)).kernel, T_moving.kernel, atol=1e-14)


if __name__ == "__main__":
    main()
