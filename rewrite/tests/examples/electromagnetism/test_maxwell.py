"""Unit tests for Maxwell stress-energy extensor, relativistic dust, and fluid tensors."""

from __future__ import annotations

import numpy as np

from examples.electromagnetism.maxwell import (
    Vector as V,
    isotropic_cloud,
    main,
    null_cloud,
    normal_stress,
    mv,
    real_sorted,
    t,
)


def test_maxwell_stress_tracelessness_and_symmetry():
    """Verify Maxwell stress-energy extensor is traceless and symmetric."""
    Ex, By = 2.5, 2.5
    T_em = 0.5 * ((mv.tx * Ex + mv.zx * By) >> V)
    assert np.isclose(T_em.trace().kernel, 0.0, atol=1e-14)

    rng = np.random.default_rng(123)
    a = mv.vector(rng.normal(size=4))
    b = mv.vector(rng.normal(size=4))
    assert np.isclose((a | T_em(b)).kernel.item(), (b | T_em(a)).kernel.item(), atol=1e-14)


def test_dust_extensor_invariants_and_ram_pressure():
    """Verify dust extensor has zero rest stress, exact trace = rho, and correct dynamic ram pressure."""
    rho = 5.0
    T_rest = rho * t * (t | V)
    assert all(np.isclose(normal_stress(T_rest, n).to_array(), 0.0, atol=1e-14) for n in (mv.x, mv.y, mv.z))
    assert np.isclose(T_rest.trace().kernel, rho, atol=1e-14)

    zeta = np.arctanh(0.6)
    u = (mv.zt * (zeta / 2.0)).exp() >> t
    gamma, beta = np.cosh(zeta), np.tanh(zeta)
    T_moving = rho * u * (u | V)
    assert np.isclose(T_moving.trace().kernel, rho, atol=1e-14)
    assert np.isclose(normal_stress(T_moving, mv.z).to_array(), -rho * gamma**2 * beta**2, atol=1e-14)


def test_particle_cloud_is_an_ideal_fluid():
    """The summed rank-1 quadrics of an isotropic cloud have fluid eigenstructure with p = rho v^2 / 3."""
    rng = np.random.default_rng(7)
    speed = 0.7
    u = isotropic_cloud(4000, speed, rng)
    mass = np.full((4000, 1), 1.0 / 4000)
    T = (u * (u | V) * mv.scalar(mass)).sum()
    *stresses, energy = real_sorted(T.eigvals()).kernel[..., 0]
    pressure = -np.mean(stresses)
    assert np.isclose(T.trace().kernel, 1.0, atol=1e-14)
    assert np.isclose(pressure, energy * speed**2 / 3.0, rtol=0.05)
    assert np.allclose(stresses, -pressure, rtol=0.05)


def test_pure_wave_is_a_null_dyad_and_the_commutator_form_holds():
    E = 3.0
    T = 0.5 * ((mv.tx * E + mv.zx * E) >> V)
    k = t + mv.z
    np.testing.assert_allclose(T.kernel, (E**2 * k * (k | V)).kernel, atol=1e-14)

    rng = np.random.default_rng(11)
    c = rng.normal(size=6)
    F = mv.tx * c[0] + mv.ty * c[1] + mv.tz * c[2] + mv.yz * c[3] + mv.zx * c[4] + mv.xy * c[5]
    lagrangian = (F * F).select.scalar()
    np.testing.assert_allclose((0.5 * (F >> V)).kernel, (F.commutator(F.commutator(V)) - 0.5 * lagrangian * V).kernel, atol=1e-13)
    np.testing.assert_allclose(F.commutator(F.commutator(V)).trace().kernel, 2.0 * lagrangian.kernel, atol=1e-13)


def test_null_cloud_is_exactly_traceless_radiation():
    rng = np.random.default_rng(12)
    rays = null_cloud(3000, rng)
    T = (rays * (rays | V) * mv.scalar(np.full((3000, 1), 2.0 / 3000))).sum()
    *stresses, energy = real_sorted(T.eigvals()).kernel[..., 0]
    np.testing.assert_allclose(T.trace().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(-np.mean(stresses), energy / 3.0, atol=1e-12)
    assert np.allclose(stresses, -2.0 / 3.0, rtol=0.08)


def test_fluid_equation_of_state_interpolation():
    """Verify fluid extensor trace matches rho - 3p, yielding dust for p=0 and radiation for p=rho/3."""
    rho = 6.0
    for p in (0.0, 1.5, rho / 3.0):
        pressure = mv.scalar([p])
        T = (rho + pressure) * t * (t | V) - pressure * V
        assert np.isclose(T.trace().kernel, rho - 3.0 * p, atol=1e-14)


def test_lorentz_4force_density():
    """Verify the Lorentz force density J x F on a moving charge density."""
    Ex, By = 3.0, 2.0
    F = mv.tx * Ex + mv.zx * By
    zeta = np.arctanh(0.5)
    u = (mv.zt * (zeta / 2.0)).exp() >> mv.t
    gamma, beta = np.cosh(zeta), np.tanh(zeta)
    force = (u * 0.8).commutator(F)
    assert np.isclose(float((force | mv.x.inverse()).kernel.item()), 0.8 * gamma * (Ex - beta * By), atol=1e-14)


def test_lorentz_boost_covariance():
    """Verify GA sandwich covariance: T' = L >> T(L << V)."""
    rho = 4.0
    T_dust = rho * t * (t | V)
    boost = (mv.zt * 0.4).exp()
    u = boost >> t
    assert np.allclose((boost >> T_dust(boost << V)).kernel, (rho * u * (u | V)).kernel, atol=1e-14)


def test_tutorial_runs():
    main()
