"""Unit tests for Maxwell stress-energy extensor, relativistic dust, and fluid tensors."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.electromagnetism.maxwell import (
    Antivector,
    along_x,
    along_y,
    along_z,
    electric,
    flow,
    isotropic_cloud,
    main,
    mv,
    normal_stress,
    null_cloud,
    real_sorted,
    time,
)


def test_maxwell_stress_tracelessness_and_symmetry():
    """Verify Maxwell stress-energy extensor is traceless and symmetric."""
    Ex, By = 2.5, 2.5
    wave = Ex * electric(along_x) + By * mv.zx
    T_em = 0.5 * (wave >> Antivector)
    assert np.isclose(T_em.trace().kernel, 0.0, atol=1e-14)

    rng = np.random.default_rng(123)
    a = (stack((along_x, along_y, along_z, time)) * rng.normal(size=4)).sum(axis=-1)
    b = (stack((along_x, along_y, along_z, time)) * rng.normal(size=4)).sum(axis=-1)
    assert np.isclose((-a.dual() & T_em(b)).kernel.item(), (-b.dual() & T_em(a)).kernel.item(), atol=1e-14)


def test_dust_extensor_invariants_and_ram_pressure():
    """Verify dust extensor has zero rest stress, trace equal to the density, and correct dynamic ram pressure."""
    rho = 5.0
    density = mv.scalar([rho])
    T_rest = density * flow(time)
    assert all(np.isclose(normal_stress(T_rest, n).to_array(), 0.0, atol=1e-14) for n in (along_x, along_y, along_z))
    assert np.isclose(T_rest.trace().kernel, rho, atol=1e-14)

    zeta = np.arctanh(0.6)
    u = (mv.zt * (zeta / 2.0)).exp() >> time
    gamma, beta = np.cosh(zeta), np.tanh(zeta)
    T_moving = density * flow(u)
    assert np.isclose(T_moving.trace().kernel, rho, atol=1e-14)
    assert np.isclose(normal_stress(T_moving, along_z).to_array(), -rho * gamma**2 * beta**2, atol=1e-14)


def test_particle_cloud_is_an_ideal_fluid():
    """The summed rank-1 quadrics of an isotropic cloud have fluid eigenstructure with pressure rho * v**2 / 3."""
    rng = np.random.default_rng(7)
    speed = 0.7
    u = isotropic_cloud(4000, speed, rng)
    mass = mv.scalar(np.full((4000, 1), 1.0 / 4000))
    T = (flow(u) * mass).sum()
    spectrum = real_sorted(T.eigvals())
    stresses, energy = spectrum[:-1], spectrum[-1]
    pressure = -stresses.mean()
    assert np.isclose(T.trace().kernel, 1.0, atol=1e-14)
    assert np.isclose(pressure.kernel, (energy * speed**2 / 3.0).kernel, rtol=0.05)
    assert np.allclose(stresses.kernel, -pressure.kernel, rtol=0.05)


def test_pure_wave_is_a_null_dyad_and_the_commutator_form_holds():
    E = 3.0
    wave = E * electric(along_x) + E * mv.zx
    T = 0.5 * (wave >> Antivector)
    ray = time + along_z
    np.testing.assert_allclose(T.kernel, (E**2 * flow(ray)).kernel, atol=1e-14)

    rng = np.random.default_rng(11)
    c = rng.normal(size=6)
    F = (stack((mv.xt, mv.yt, mv.zt, mv.yz, mv.zx, mv.xy)) * c).sum(axis=0)
    lagrangian = (F * F).select.scalar()
    stress_from_force = F.commutator(F.commutator(Antivector)) - 0.5 * lagrangian * Antivector
    np.testing.assert_allclose((0.5 * (F >> Antivector)).kernel, stress_from_force.kernel, atol=1e-13)
    np.testing.assert_allclose(F.commutator(F.commutator(Antivector)).trace().kernel, (2.0 * lagrangian).kernel, atol=1e-13)


def test_null_cloud_is_exactly_traceless_radiation():
    rng = np.random.default_rng(12)
    rays = null_cloud(3000, rng)
    mass = mv.scalar(np.full((3000, 1), 2.0 / 3000))
    T = (flow(rays) * mass).sum()
    spectrum = real_sorted(T.eigvals())
    stresses, energy = spectrum[:-1], spectrum[-1]
    np.testing.assert_allclose(T.trace().kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(-stresses.mean().kernel, (energy / 3.0).kernel, atol=1e-12)
    assert np.allclose(stresses.kernel, -2.0 / 3.0, rtol=0.08)


def test_fluid_equation_of_state_interpolation():
    """Verify fluid extensor trace matches rho - 3p, yielding dust for p=0 and radiation for p=rho/3."""
    rho = 6.0
    for p in (0.0, 1.5, rho / 3.0):
        pressure = mv.scalar([p])
        T = (rho + pressure) * flow(time) - pressure * Antivector
        assert np.isclose(T.trace().kernel, rho - 3.0 * p, atol=1e-14)


def test_lorentz_4force_density():
    """Verify the Lorentz force density J x F on a moving charge density."""
    Ex, By = 3.0, 2.0
    wave = Ex * electric(along_x) + By * mv.zx
    zeta = np.arctanh(0.5)
    boost = (mv.zt * (zeta / 2.0)).exp()
    velocity = boost >> time
    gamma, beta = np.cosh(zeta), np.tanh(zeta)
    charges = velocity * 0.8
    force = wave & (Antivector & charges)
    assert np.isclose(force(along_x).to_array(), 0.8 * gamma * (Ex - beta * By), atol=1e-14)


def test_lorentz_boost_covariance():
    """Verify GA sandwich covariance: boost >> T(boost << Antivector)."""
    rho = 4.0
    density = mv.scalar([rho])
    T_dust = density * flow(time)
    boost = (mv.zt * 0.4).exp()
    u = boost >> time
    assert np.allclose((boost >> T_dust(boost << Antivector)).kernel, (density * flow(u)).kernel, atol=1e-14)


def test_tutorial_runs():
    main()
