r"""Stress-energy in the spacetime algebra: of an electromagnetic field, of dust, of a cloud of particles, and
of an ideal fluid, each a map from a direction in spacetime to the flow of energy and momentum along it.

The stress-energy of a field is half its sandwich of a direction, `0.5 * (field >> Antivector)`: the
observer's direction in time picks out the energy current, which the observer's slice of space reads as the
energy density, `mv.t & stress(time)`; the rest of that current is the Poynting flux, and the map's trace
vanishes. It is also the Lorentz force applied twice, less the field's Lagrangian as a pressure; and for a
wave running one way it collapses to a single dyad along the ray, the stress-energy of light, massless dust.
Dust moving at a velocity is the dyad of that velocity with its own slice of space, `-velocity.dual()`, the
slice that reads it as one; its trace is the rest density. Summed over a cloud of particles moving every way
at one speed, the dyads become an ideal fluid: its timelike eigenvalue is the energy density, its spatial
ones minus the pressure. A cloud of light rays is traceless, so its pressure is exactly a third of its energy
density. Every one of these maps moves under a boost as any map does, `boost >> stress(boost << Antivector)`,
and a field's map moves with its field. The force density on a current reads the field on the plane each
direction sweeps along the current, `field & (direction & current)`.

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
Bivector = STA.gatype.bivector()
# Directions in spacetime: velocities, rays, the normals of faces, and the currents of energy and momentum.
Antivector = STA.gatype.antivector()
# A stress-energy map takes a direction to the current of energy and momentum along it.
StressEnergy = STA.gatype((Antivector, Antivector))  # Antivector <- Antivector

# The observer at rest: its slice of space `mv.t` and its direction in time, which pairs to one with that
# slice; the dual carries the minus of t's square. And the directions of space.
time = -mv.t.dual()
along_x, along_y, along_z = mv.x.dual(), mv.y.dual(), mv.z.dual()
# The unit volume.
pseudoscalar = mv.xyzt
# The plane wave's electric and magnetic field strengths; a field of no special form, its coefficients on
# xt, yt, zt, yz, zx and xy.
electric_strength, magnetic_strength = 2.0, 2.0
general_field = np.array([0.7, -0.4, 1.1, 0.3, -0.8, 0.5])
# The dust's density, the speed of the moving dust, and the charge density of the current it carries.
dust_density, dust_speed, charge_density = 4.0, 0.6, 0.5
# The particle and ray clouds: their samples, the particles' speed, and the seed.
n_particles, cloud_speed, seed = 2000, 0.5, 0


# --- plumbing -------------------------------------------------------------------------
def electric(direction: Antivector) -> Bivector:
    """The unit electric field along a direction: the dual of the plane it sweeps through time."""
    return (direction & time).dual()


def magnetic(direction: Antivector) -> Bivector:
    """The unit magnetic field along a direction: the unit volume times the electric one."""
    return pseudoscalar * electric(direction)


def normal_stress(stress: StressEnergy, normal: Antivector) -> Scalar:
    """The normal traction on a face with a unit spatial normal: the momentum current across the face, read
    by the slice that reads the normal as one, the dual of its inverse."""
    return normal.inverse().dual() & stress(normal)


def real_sorted(values: Scalar) -> Scalar:
    """Read a real spectrum in ascending order, retaining its scalar type."""
    return mv.scalar(np.sort(values.kernel.real, axis=-2))


def sphere_directions(n_directions: int, rng: np.random.Generator) -> Antivector:
    """Unit spatial directions, uniform on the sphere."""
    samples = (stack((along_x, along_y, along_z)) * rng.normal(size=(n_directions, 3))).sum(axis=-1)
    return samples / samples.squared().square_root()


def isotropic_cloud(n_particles: int, speed: float, rng: np.random.Generator) -> Antivector:
    """Velocities at one speed, directions uniform on the sphere."""
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    return gamma * (time + speed * sphere_directions(n_particles, rng))


def null_cloud(n_rays: int, rng: np.random.Generator) -> Antivector:
    """Null rays `time + direction`, directions uniform on the sphere."""
    return time + sphere_directions(n_rays, rng)


def flow(velocities: Antivector) -> StressEnergy:
    """The dyad of each velocity or ray with its own slice of space: the stress-energy of a unit of dust, its
    output embedded among all directions."""
    return (velocities * (-velocities.dual() & Antivector)).cast(StressEnergy)


# --- math -----------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(seed)

    # A plane wave along +z, its electric field along x and its magnetic field along y, and its
    # stress-energy: for the observer at rest, the energy density and the Poynting flux.
    wave = electric_strength * electric(along_x) + magnetic_strength * magnetic(along_y)   # [] Bivector
    wave_stress = 0.5 * (wave >> Antivector)                                   # [] Antivector <- Antivector
    current = wave_stress(time)                                                # [] Antivector
    energy = mv.t & current                                                    # [] Scalar
    poynting = current - time * energy                                         # [] Antivector
    # A wave running one way, with equal electric and magnetic strengths, is one null dyad along its ray.
    ray = time + along_z                                                       # [] Antivector

    # For any field the same map is the Lorentz force `field.commutator(direction)` applied twice, less the
    # Lagrangian, the squared electric less the squared magnetic strength, as an isotropic pressure.
    field = (stack((mv.xt, mv.yt, mv.zt, mv.yz, mv.zx, mv.xy)) * general_field).sum(axis=0)   # [] Bivector
    lagrangian = (field * field).select.scalar()                               # [] Scalar
    stress_from_force = field.commutator(field.commutator(Antivector)) - 0.5 * lagrangian * Antivector   # [] Antivector <- Antivector

    # Dust at rest, and the same dust moving: the dyad of its velocity, the rest observer boosted.
    density = mv.scalar([dust_density])
    dust = density * flow(time)                                                # [] Antivector <- Antivector
    rapidity = np.arctanh(dust_speed)
    boost = (mv.zt * (rapidity / 2.0)).exp()                                   # [] Rotor
    velocity = boost >> time                                                   # [] Antivector
    gamma, beta = np.cosh(rapidity), np.tanh(rapidity)
    moving_dust = density * flow(velocity)                                     # [] Antivector <- Antivector

    # Each particle contributes its own dyad; summed over a cloud moving every way at one speed, the
    # stress-energy is an ideal fluid at rest, its pressure `energy * speed ** 2 / 3`, its trace the
    # summed rest mass, exactly.
    velocities = isotropic_cloud(n_particles, cloud_speed, rng)                # [particles] Antivector
    mass = mv.scalar(np.full((n_particles, 1), 1.0 / n_particles))             # [particles] Scalar
    cloud = (flow(velocities) * mass).sum()                                    # [] Antivector <- Antivector
    spectrum = real_sorted(cloud.eigvals())
    stresses, cloud_energy = spectrum[:-1], spectrum[-1]
    pressure = -stresses.mean()

    # A cloud of light rays is the massless limit. Each ray is null, so every dyad is traceless and the
    # trace vanishes to round-off rather than statistically: the pressure is exactly a third of the energy.
    rays = null_cloud(n_particles, rng)                                        # [rays] Antivector
    light = (flow(rays) * mass).sum()                                          # [] Antivector <- Antivector
    light_spectrum = real_sorted(light.eigvals())
    light_stresses, light_energy = light_spectrum[:-1], light_spectrum[-1]

    # The ideal fluid of the cloud's energy and pressure, and the radiation fluid of pressure a third of
    # its density.
    fluid = (cloud_energy + pressure) * flow(time) - pressure * Antivector      # StressEnergy
    radiation = (density + density / 3.0) * flow(time) - (density / 3.0) * Antivector  # StressEnergy

    # The Lorentz force density on the current the moving dust carries, a form on directions: the field
    # read on the plane each direction sweeps along the current, as charges at rest read it on the plane
    # the direction sweeps through time. And the wave's stress-energy moved by the boost as a map.
    charges = velocity * charge_density                                        # [] Antivector
    force = wave & (Antivector & charges)                                      # [] Scalar <- Antivector
    boosted_stress = boost >> wave_stress(boost << Antivector)                 # [] Antivector <- Antivector

    # checks
    total = 0.5 * (electric_strength**2 + magnetic_strength**2)
    np.testing.assert_allclose(energy.to_array(), total, atol=1e-14)
    np.testing.assert_allclose((stack((mv.x, mv.y, mv.z)) & poynting).to_array(), [0.0, 0.0, electric_strength * magnetic_strength], atol=1e-14)
    np.testing.assert_allclose([normal_stress(wave_stress, normal).to_array() for normal in (along_x, along_y)], 0.0, atol=1e-14)
    np.testing.assert_allclose(normal_stress(wave_stress, along_z).to_array(), -total, atol=1e-14)
    np.testing.assert_allclose(wave_stress.trace().kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((wave_stress - electric_strength**2 * flow(ray)).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((0.5 * (field >> Antivector)).kernel, stress_from_force.kernel, atol=1e-14)
    np.testing.assert_allclose([normal_stress(dust, normal).to_array() for normal in (along_x, along_y, along_z)], 0.0, atol=1e-14)
    np.testing.assert_allclose(dust.trace().kernel, density.to_array(), atol=1e-14)
    np.testing.assert_allclose((mv.t & moving_dust(time)).to_array(), (density * gamma**2).to_array(), atol=1e-14)
    np.testing.assert_allclose(normal_stress(moving_dust, along_z).to_array(), (-density * gamma**2 * beta**2).to_array(), atol=1e-14)
    np.testing.assert_allclose(moving_dust.trace().kernel, density.to_array(), atol=1e-14)
    np.testing.assert_allclose(cloud.trace().kernel, mass.sum().kernel, atol=1e-14)
    np.testing.assert_allclose(pressure.kernel, (cloud_energy * cloud_speed**2 / 3.0).kernel, rtol=0.05)
    np.testing.assert_allclose(stresses.kernel, np.broadcast_to(-pressure.kernel, stresses.kernel.shape), rtol=0.05)
    np.testing.assert_allclose(light.trace().kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(-light_stresses.mean().kernel, (light_energy / 3.0).kernel, atol=1e-12)
    np.testing.assert_allclose(fluid.trace().kernel, (cloud_energy - 3.0 * pressure).kernel, atol=1e-14)
    np.testing.assert_allclose(fluid.kernel, cloud.kernel, atol=0.02)
    np.testing.assert_allclose(radiation.trace().kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(force(along_x).to_array(), charge_density * gamma * (electric_strength - beta * magnetic_strength), atol=1e-14)
    np.testing.assert_allclose(boosted_stress.kernel, (0.5 * ((boost >> wave) >> Antivector)).kernel, atol=1e-14)
    np.testing.assert_allclose((boost >> dust(boost << Antivector)).kernel, moving_dust.kernel, atol=1e-14)


if __name__ == "__main__":
    main()
