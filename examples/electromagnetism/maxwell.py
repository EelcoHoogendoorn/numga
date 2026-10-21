r"""Stress-energy in the spacetime algebra: of an electromagnetic field, of dust, of a cloud of particles, and
of an ideal fluid, each a map from an observer to the flow of energy and momentum that observer sees.

The stress-energy of a field is half its sandwich of the observer, `0.5 * (field >> Vector)`: the observer's
time picks out the energy density, the rest of the image is the Poynting flux, and the map's trace vanishes.
It is also the Lorentz force applied twice, less the field's Lagrangian as a pressure; and for a wave
running one way it collapses to a single dyad along the ray, the stress-energy of light, massless dust.
Dust moving with a four-velocity is the dyad of that velocity, its trace the rest density. Summed over a
cloud of particles moving every way at one speed, the dyads become an ideal fluid: its timelike eigenvalue is
the energy density, its spatial ones minus the pressure. A cloud of light rays is traceless, so its pressure
is exactly a third of its energy density. Every one of these maps moves under a boost as any map does,
`boost >> stress(boost << Vector)`, and a field's map moves with its field.

In the tensor notation of relativity the stress-energy reads as $T^{\mu}{}_{\nu}$, the field's as
$T^{\mu\nu} = F^{\mu\alpha}F^{\nu}{}_{\alpha} - \tfrac14 g^{\mu\nu}F^{\alpha\beta}F_{\alpha\beta}$, dust's as
$\rho\, u^\mu u^\nu$, the ideal fluid's as $(\epsilon + p)\, u^\mu u^\nu - p\, g^{\mu\nu}$, and the Lorentz force
density as $F^{\mu}{}_{\nu} J^{\nu}$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

ctx = NumpyContext(STA)
mv = ctx.multivector

Scalar = STA.gatype.scalar()
Vector = STA.gatype.vector()
# A stress-energy map takes an observer to the momentum flux it sees.
StressEnergy = STA.gatype((Vector, Vector))  # Vector <- Vector

# The rest observer and spatial axes, each with full vector support, so the maps built on them are maps
# from vectors to vectors.
t, x, y, z = mv.vector(np.eye(4))
# The plane wave's electric and magnetic field strengths; a field of no special form, its coefficients on
# tx, ty, tz, yz, zx and xy.
ELECTRIC, MAGNETIC = 2.0, 2.0
GENERAL_FIELD = np.array([0.7, -0.4, 1.1, 0.3, -0.8, 0.5])
# The dust's density, the speed of the moving dust, and the charge density of the current it carries.
DENSITY, SPEED, CHARGE_DENSITY = 4.0, 0.6, 0.5
# The particle and ray clouds: their samples, the particles' speed, and the seed.
N_PARTICLES, CLOUD_SPEED, SEED = 2000, 0.5, 0


# --- plumbing -------------------------------------------------------------------------
def normal_stress(stress: StressEnergy, normal: Vector) -> Scalar:
    """Normal traction on a face with unit spatial normal, `normal * normal == -1`."""
    return normal.inverse() | stress(normal)


def real_sorted(values: Scalar) -> Scalar:
    """Read a real spectrum in ascending order, retaining its scalar type."""
    return mv.scalar(np.sort(values.kernel.real, axis=-2))


def sphere_directions(n_directions: int, rng: np.random.Generator) -> Vector:
    """Unit spatial directions, uniform on the sphere."""
    samples = mv.vector(np.concatenate([np.zeros((n_directions, 1)), rng.normal(size=(n_directions, 3))], axis=1))
    # A spatial vector squares to minus its length squared.
    return samples / (-(samples | samples)).square_root()


def isotropic_cloud(n_particles: int, speed: float, rng: np.random.Generator) -> Vector:
    """Four-velocities at one speed, directions uniform on the sphere."""
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    return gamma * (t + speed * sphere_directions(n_particles, rng))


def null_cloud(n_rays: int, rng: np.random.Generator) -> Vector:
    """Null rays `t + direction`, directions uniform on the sphere."""
    return t + sphere_directions(n_rays, rng)


# --- math -----------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)

    # A plane wave along +z, its electric field along x and its magnetic field along y, and its
    # stress-energy: for the observer at rest, the energy density and the Poynting flux.
    wave = mv.tx * ELECTRIC + mv.zx * MAGNETIC                                  # [] Bivector
    wave_stress: StressEnergy = 0.5 * (wave >> Vector)                          # [] Vector <- Vector
    flux = wave_stress(t)                                                       # [] Vector
    energy = t | flux                                                           # [] Scalar
    poynting = flux - t * energy                                                # [] Vector
    # A wave running one way, with equal electric and magnetic strengths, is one null dyad along its ray.
    ray = t + z                                                                 # [] Vector

    # For any field the same map is the Lorentz force `field.commutator(v)` applied twice, less the
    # Lagrangian, the squared electric less the squared magnetic strength, as an isotropic pressure.
    field = (stack((mv.tx, mv.ty, mv.tz, mv.yz, mv.zx, mv.xy)) * GENERAL_FIELD).sum(axis=0)   # [] Bivector
    lagrangian = (field * field).select.scalar()                                # [] Scalar
    stress_from_force = field.commutator(field.commutator(Vector)) - 0.5 * lagrangian * Vector   # [] Vector <- Vector

    # Dust at rest, and the same dust moving: the dyad of its four-velocity, the rest observer boosted.
    density = mv.scalar([DENSITY])
    dust: StressEnergy = density * t * (t | Vector)                             # [] Vector <- Vector
    rapidity = np.arctanh(SPEED)
    boost = (mv.zt * (rapidity / 2.0)).exp()                                    # [] Rotor
    velocity = boost >> t                                                       # [] Vector
    gamma, beta = np.cosh(rapidity), np.tanh(rapidity)
    moving_dust: StressEnergy = density * velocity * (velocity | Vector)        # [] Vector <- Vector

    # Each particle contributes its own dyad; summed over a cloud moving every way at one speed, the
    # stress-energy is an ideal fluid at rest, its pressure `energy * speed ** 2 / 3`, its trace the
    # summed rest mass, exactly.
    velocities = isotropic_cloud(N_PARTICLES, CLOUD_SPEED, rng)                 # [particles] Vector
    mass = mv.scalar(np.full((N_PARTICLES, 1), 1.0 / N_PARTICLES))              # [particles] Scalar
    cloud: StressEnergy = (velocities * (velocities | Vector) * mass).sum()     # [] Vector <- Vector
    spectrum = real_sorted(cloud.eigvals())
    stresses, cloud_energy = spectrum[:-1], spectrum[-1]
    pressure = -stresses.mean()

    # A cloud of light rays is the massless limit. Each ray is null, so every dyad is traceless and the
    # trace vanishes to round-off rather than statistically: the pressure is exactly a third of the energy.
    rays = null_cloud(N_PARTICLES, rng)                                         # [rays] Vector
    light: StressEnergy = (rays * (rays | Vector) * mass).sum()                 # [] Vector <- Vector
    light_spectrum = real_sorted(light.eigvals())
    light_stresses, light_energy = light_spectrum[:-1], light_spectrum[-1]

    # The ideal fluid of the cloud's energy and pressure, and the radiation fluid of pressure a third of
    # its density.
    fluid: StressEnergy = (cloud_energy + pressure) * t * (t | Vector) - pressure * Vector
    radiation: StressEnergy = (density + density / 3.0) * t * (t | Vector) - (density / 3.0) * Vector

    # The Lorentz force density on the current the moving dust carries, and the wave's stress-energy moved
    # by the boost as a map.
    current = velocity * CHARGE_DENSITY                                         # [] Vector
    force = current.commutator(wave)                                            # [] Vector
    boosted_stress = boost >> wave_stress(boost << Vector)                      # [] Vector <- Vector

    # checks
    np.testing.assert_allclose(energy.to_array(), 0.5 * (ELECTRIC**2 + MAGNETIC**2), atol=1e-14)
    np.testing.assert_allclose(poynting.kernel, [0.0, 0.0, 0.0, ELECTRIC * MAGNETIC], atol=1e-14)
    np.testing.assert_allclose([normal_stress(wave_stress, normal).to_array() for normal in (x, y)], 0.0, atol=1e-14)
    np.testing.assert_allclose(normal_stress(wave_stress, z).to_array(), -0.5 * (ELECTRIC**2 + MAGNETIC**2), atol=1e-14)
    np.testing.assert_allclose(wave_stress.trace().kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(wave_stress.kernel, (ELECTRIC**2 * ray * (ray | Vector)).kernel, atol=1e-14)
    np.testing.assert_allclose((0.5 * (field >> Vector)).kernel, stress_from_force.kernel, atol=1e-14)
    np.testing.assert_allclose([normal_stress(dust, normal).to_array() for normal in (x, y, z)], 0.0, atol=1e-14)
    np.testing.assert_allclose(dust.trace().kernel, density.to_array(), atol=1e-14)
    np.testing.assert_allclose((t | moving_dust(t)).to_array(), (density * gamma**2).to_array(), atol=1e-14)
    np.testing.assert_allclose(normal_stress(moving_dust, z).to_array(), (-density * gamma**2 * beta**2).to_array(), atol=1e-14)
    np.testing.assert_allclose(moving_dust.trace().kernel, density.to_array(), atol=1e-14)
    np.testing.assert_allclose(cloud.trace().kernel, mass.sum().kernel, atol=1e-14)
    np.testing.assert_allclose(pressure.kernel, (cloud_energy * CLOUD_SPEED**2 / 3.0).kernel, rtol=0.05)
    np.testing.assert_allclose(stresses.kernel, np.broadcast_to(-pressure.kernel, stresses.kernel.shape), rtol=0.05)
    np.testing.assert_allclose(light.trace().kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(-light_stresses.mean().kernel, (light_energy / 3.0).kernel, atol=1e-12)
    np.testing.assert_allclose(fluid.trace().kernel, (cloud_energy - 3.0 * pressure).kernel, atol=1e-14)
    np.testing.assert_allclose(fluid.kernel, cloud.kernel, atol=0.02)
    np.testing.assert_allclose(radiation.trace().kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((force | x.inverse()).to_array(), CHARGE_DENSITY * gamma * (ELECTRIC - beta * MAGNETIC), atol=1e-14)
    np.testing.assert_allclose(boosted_stress.kernel, (0.5 * ((boost >> wave) >> Vector)).kernel, atol=1e-14)
    np.testing.assert_allclose((boost >> dust(boost << Vector)).kernel, moving_dust.kernel, atol=1e-14)


if __name__ == "__main__":
    main()
