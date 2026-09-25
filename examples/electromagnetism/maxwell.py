"""Maxwell Stress-Energy Extensor, Relativistic Dust Tensor, and Lorentz Force in Spacetime Algebra (STA).

Mathematical formulation of energy-momentum extensors in STA, `Algebra("t+x-y-z-")`:
1. Spacetime Algebra: `t * t == 1` and `x * x == y * y == z * z == -1`.
2. Maxwell Energy-Momentum Extensor:
       T_em(v) == -0.5 * F * v * F == 0.5 * (F >> v)
   with exact tracelessness `T_em.trace() == 0` and Poynting flux `T_em(t) - t * (t | T_em(t))`.
   The same map is the iterated Lorentz force minus the Lagrangian pressure,
   `F.commutator(F.commutator(v)) - 0.5 * (F * F).select.scalar() * v`, and for a pure
   travelling wave it collapses to a single null dyad, `Ex ** 2 * k * (k | v)`: massless dust.
3. Relativistic Dust Energy-Momentum Extensor (pressureless matter):
       T_moving(v) == rho * u * (u | v)
   with rank 1, zero spatial stress in rest frame, and invariant trace `T_moving.trace() == rho`.
4. Particle Cloud Extensor, the same rank-1 quadric summed over a batch of four-velocities:
       T_cloud(v) == (u_cloud * (u_cloud | v) * mass).sum()
   whose eigenstructure is an ideal fluid: the timelike eigenvalue is the rest-frame energy
   density, the spatial eigenvalues are minus the pressure, and `T_cloud.trace() == mass.sum()`.
   A cloud of null rays is the massless limit: exactly traceless, so the pressure is exactly a
   third of the energy density.
5. Ideal Relativistic Fluid Extensor:
       T_fluid(v) == (energy_cloud + pressure) * t * (t | v) - pressure * v
   with `T_fluid.trace() == energy_cloud - 3 * pressure`, matched against the cloud.
6. Lorentz 4-Force Density on a current: `force = J.commutator(F)`.
7. Lorentz Covariance via Rotor Sandwich:
       boost >> T_em(boost << Vector) == 0.5 * ((boost >> F) >> Vector)
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import STA

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: Algebra("t+x-y-z-"))
# ---------------------------------------------------------------------------
ctx = NumpyContext(STA)
mv = ctx.multivector

Scalar = STA.gatype.scalar()
Vector = STA.gatype.vector()
# A stress-energy map takes an observer to the momentum flux it sees.
StressEnergy = STA.gatype((Vector, Vector))  # Vector <- Vector

# The rest observer and spatial axes, each with full vector support, so the tensors built on
# them are endomorphisms.
t, x, y, z = mv.vector(np.eye(4))
# Sample size of the particle and ray clouds.
PARTICLES = 2000


# ---------------------------------------------------------------------------
# 2. Read-outs and Samples
# ---------------------------------------------------------------------------
def normal_stress(T: StressEnergy, n: Vector) -> Scalar:
    """Normal traction on a face with unit spatial normal n, `n * n == -1`: `n.inverse() | T(n)`."""
    return n.inverse() | T(n)


def real_sorted(values: Scalar) -> Scalar:
    """Read a real spectrum in ascending order, retaining its scalar type."""
    return mv.scalar(np.sort(values.kernel.real, axis=-2))


def sphere_directions(n: int, rng: np.random.Generator) -> Vector:
    """n unit spatial directions, uniform on the sphere."""
    samples: Vector = mv.vector(np.concatenate([np.zeros((n, 1)), rng.normal(size=(n, 3))], axis=1))
    # A spatial vector squares to minus its length squared.
    return samples / (-(samples | samples)).square_root()


def isotropic_cloud(n: int, speed: float, rng: np.random.Generator) -> Vector:
    """n four-velocities at one speed, directions uniform on the sphere."""
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    return gamma * (t + speed * sphere_directions(n, rng))


def null_cloud(n: int, rng: np.random.Generator) -> Vector:
    """n null rays `t + direction`, directions uniform on the sphere."""
    return t + sphere_directions(n, rng)


# ---------------------------------------------------------------------------
# 3. Demonstration & Algebraic Verifications
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(0)

    # -----------------------------------------------------------------------
    # A. Maxwell Energy-Momentum Extensor: plane wave along +z, electric field Ex along x,
    #    magnetic field By along y
    # -----------------------------------------------------------------------
    Ex, By = 2.0, 2.0
    F = mv.tx * Ex + mv.zx * By
    T_em: StressEnergy = 0.5 * (F >> Vector)

    # Observer at rest: energy density and Poynting flux
    flux = T_em(t)
    energy = t | flux
    poynting = flux - t * energy

    # A pure travelling wave has electric and magnetic fields of equal strength, and its tensor
    # is one null dyad along the ray `k = t + z`: the massless counterpart of the dust tensor below.
    k = t + z

    # For any field the same map is the Lorentz force generator `F.commutator(v)` iterated, minus
    # the Lagrangian `(F * F).select.scalar()`, the squared electric minus the squared magnetic
    # field strength, as an isotropic pressure.
    F_any = mv.tx * 0.7 + mv.ty * -0.4 + mv.tz * 1.1 + mv.yz * 0.3 + mv.zx * -0.8 + mv.xy * 0.5
    lagrangian = (F_any * F_any).select.scalar()
    stress_from_force = F_any.commutator(F_any.commutator(Vector)) - 0.5 * lagrangian * Vector

    # -----------------------------------------------------------------------
    # B. Dust Energy-Momentum Extensor: at rest, then boosted
    # -----------------------------------------------------------------------
    rho = mv.scalar([4.0])
    T_dust: StressEnergy = rho * t * (t | Vector)

    # The moving observer is the rest observer boosted; gamma and beta follow from the rapidity
    zeta = np.arctanh(0.6)
    boost = (mv.zt * (zeta / 2.0)).exp()
    u = boost >> t
    gamma, beta = np.cosh(zeta), np.tanh(zeta)
    T_moving: StressEnergy = rho * u * (u | Vector)

    # -----------------------------------------------------------------------
    # C. Particle Cloud: the dust quadric summed over four-velocities is a fluid
    # -----------------------------------------------------------------------
    # Each particle contributes `mass * u * (u | v)`. Summed over an isotropic cloud, the tensor's
    # eigenstructure is that of an ideal fluid at rest: the timelike eigenvalue is the energy
    # density, the three spatial eigenvalues are minus the pressure, and for particles all at
    # one speed the pressure is `energy_cloud * speed ** 2 / 3`. The trace is the summed rest
    # mass, exactly.
    speed = 0.5
    u_cloud = isotropic_cloud(PARTICLES, speed, rng)
    mass = mv.scalar(np.full((PARTICLES, 1), 1.0 / PARTICLES))
    T_cloud: StressEnergy = (u_cloud * (u_cloud | Vector) * mass).sum()

    spectrum = real_sorted(T_cloud.eigvals())
    stresses, energy_cloud = spectrum[:-1], spectrum[-1]
    pressure = -stresses.mean()

    # A cloud of null rays is the massless limit. Each ray is null, so every dyad is traceless
    # and the trace vanishes to roundoff rather than statistically: the pressure is exactly
    # `energy_light / 3`.
    rays = null_cloud(PARTICLES, rng)
    T_light: StressEnergy = (rays * (rays | Vector) * mass).sum()
    light_spectrum = real_sorted(T_light.eigvals())
    stresses_light, energy_light = light_spectrum[:-1], light_spectrum[-1]

    # -----------------------------------------------------------------------
    # D. Ideal Fluid Extensor matches the cloud
    # -----------------------------------------------------------------------
    T_fluid: StressEnergy = (energy_cloud + pressure) * t * (t | Vector) - pressure * Vector

    # Conformal radiation fluid: pressure `rho / 3` gives `T_radiation.trace() == 0`
    T_radiation: StressEnergy = (rho + rho / 3.0) * t * (t | Vector) - (rho / 3.0) * Vector

    # -----------------------------------------------------------------------
    # E. Lorentz 4-Force Density on a current: `force = J.commutator(F)`
    # -----------------------------------------------------------------------
    rho_q = 0.5
    J = u * rho_q
    force = J.commutator(F)

    # -----------------------------------------------------------------------
    # F. Lorentz Rotor Sandwich Covariance: `boost >> T_em(boost << Vector)`
    # -----------------------------------------------------------------------

    transformed_stress = boost >> T_em(boost << Vector)

    # --- checks -------------------------------------------------------------
    np.testing.assert_allclose(energy.to_array(), 0.5 * (Ex**2 + By**2), atol=1e-14)
    np.testing.assert_allclose(poynting.kernel, [0.0, 0.0, 0.0, Ex * By], atol=1e-14)
    np.testing.assert_allclose([normal_stress(T_em, n).to_array() for n in (x, y)], 0.0, atol=1e-14)
    np.testing.assert_allclose(normal_stress(T_em, z).to_array(), -0.5 * (Ex**2 + By**2), atol=1e-14)
    np.testing.assert_allclose(T_em.trace().kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(T_em.kernel, (Ex**2 * k * (k | Vector)).kernel, atol=1e-14)
    np.testing.assert_allclose(
        (0.5 * (F_any >> Vector)).kernel,
        stress_from_force.kernel,
        atol=1e-14,
    )
    np.testing.assert_allclose([normal_stress(T_dust, n).to_array() for n in (x, y, z)], 0.0, atol=1e-14)
    np.testing.assert_allclose(T_dust.trace().kernel, rho.to_array(), atol=1e-14)
    np.testing.assert_allclose((t | T_moving(t)).to_array(), (rho * gamma**2).to_array(), atol=1e-14)
    np.testing.assert_allclose(normal_stress(T_moving, z).to_array(), (-rho * gamma**2 * beta**2).to_array(), atol=1e-14)
    np.testing.assert_allclose(T_moving.trace().kernel, rho.to_array(), atol=1e-14)
    np.testing.assert_allclose(T_cloud.trace().kernel, mass.sum().kernel, atol=1e-14)
    np.testing.assert_allclose(pressure.kernel, (energy_cloud * speed**2 / 3.0).kernel, rtol=0.05)
    np.testing.assert_allclose(stresses.kernel, np.broadcast_to(-pressure.kernel, stresses.kernel.shape), rtol=0.05)
    np.testing.assert_allclose(T_light.trace().kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(-stresses_light.mean().kernel, (energy_light / 3.0).kernel, atol=1e-12)
    np.testing.assert_allclose(T_fluid.trace().kernel, (energy_cloud - 3.0 * pressure).kernel, atol=1e-14)
    np.testing.assert_allclose(T_fluid.kernel, T_cloud.kernel, atol=0.02)
    np.testing.assert_allclose(T_radiation.trace().kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((force | x.inverse()).to_array(), rho_q * gamma * (Ex - beta * By), atol=1e-14)
    np.testing.assert_allclose(transformed_stress.kernel, (0.5 * ((boost >> F) >> Vector)).kernel, atol=1e-14)
    np.testing.assert_allclose((boost >> T_dust(boost << Vector)).kernel, T_moving.kernel, atol=1e-14)


if __name__ == "__main__":
    main()
