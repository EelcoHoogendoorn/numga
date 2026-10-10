"""Spinor regularization preserves Kepler geometry and resolves close bound passages."""

import numpy as np

from examples.mechanics.kepler import core, scenarios


def test_lift_reconstructs_velocity_and_exact_motion_in_tilted_planes():
    angles = np.array([0.2, 0.5, -0.3])
    rotation = (core.mv.yz * angles).exp() * (core.mv.xy * (angles / 2)).exp()
    spinor = rotation * np.sqrt(1.8)
    velocity = rotation >> (core.mv.x * 0.13 + core.mv.y * 0.3)
    orbit = core.BoundOrbit.lift(spinor, velocity, 1.0)
    initial = core.physical_state(orbit.spinor, orbit.rate, orbit.energy * 0)
    frequency = (-orbit.energy / 2).square_root()
    phase = core.mv.scalar(np.linspace(0, np.pi, 81)[:, None, None])
    trajectory = orbit.sample(phase / frequency)
    semimajor = -1 / (2 * orbit.energy)
    period = 2 * np.pi * (semimajor ** 3).square_root()

    # checks
    np.testing.assert_allclose((initial.velocity - velocity).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((orbit.rate * core.mv.x * orbit.spinor.reverse()).select[3].kernel, 0, atol=1e-12)
    np.testing.assert_allclose((trajectory.position[-1] - initial.position).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((trajectory.velocity[-1] - initial.velocity).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((trajectory.time[-1] - period).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((trajectory.energy(1.0) - orbit.energy).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((trajectory.momentum() - initial.momentum()).kernel, 0, atol=1e-11)


def test_spinor_gauge_changes_neither_orbit_nor_clock():
    orbit = scenarios.initial_orbit()
    phase = core.mv.scalar(np.linspace(0, 2 * np.pi, 91)[:, None, None])
    gauge = (core.mv.yz * np.array([0.3, -0.5, 0.7])).exp()
    shifted = core.BoundOrbit(orbit.spinor * gauge, orbit.rate * gauge, orbit.energy)
    first, second = orbit.sample(phase), shifted.sample(phase)

    # checks
    np.testing.assert_allclose((first.position - second.position).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((first.velocity - second.velocity).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((first.time - second.time).kernel, 0, atol=1e-10)


def test_physical_clock_inversion_and_inverse_square_acceleration():
    orbit = scenarios.initial_orbit()
    times = core.mv.scalar(np.linspace(0.2, 1.0, 7)[:, None, None])
    delta = 1e-4
    state = orbit.at_time(times, scenarios.CLOCK_ITERATIONS)
    before = orbit.at_time(times - delta, scenarios.CLOCK_ITERATIONS)
    after = orbit.at_time(times + delta, scenarios.CLOCK_ITERATIONS)
    acceleration = (after.velocity - before.velocity) / (2 * delta)
    force = -scenarios.GRAVITY * state.position / state.position.norm() ** 3

    # checks
    np.testing.assert_allclose((state.time - times).kernel, 0, atol=1e-10)
    np.testing.assert_allclose(acceleration.kernel, force.kernel, atol=1e-7, rtol=1e-6)


def test_both_verlet_schemes_converge_and_regularization_resolves_periapsis():
    orbit = scenarios.initial_orbit()
    initial = core.physical_state(orbit.spinor, orbit.rate, orbit.energy * 0)
    frequency = (-orbit.energy / 2).square_root()
    period = 2 * np.pi
    physical_errors, regularized_errors = [], []
    for count in (128, 256):
        physical = core.State.collect(core.physical_verlet(initial, 1.0, period / count, count))
        regularized = core.State.collect(core.regularized_verlet(orbit, np.pi / frequency / count, count))
        physical_errors.append(orbit.position_error(physical, scenarios.CLOCK_ITERATIONS).kernel[:, 0])
        regularized_errors.append(orbit.position_error(regularized, scenarios.CLOCK_ITERATIONS).kernel[:, 0])
        # Both central-force updates preserve angular momentum despite their energy errors.
        np.testing.assert_allclose((physical.momentum() - initial.momentum()).kernel, 0, atol=1e-10)
        np.testing.assert_allclose((regularized.momentum() - initial.momentum()).kernel, 0, atol=1e-10)

    # checks: the resolved orbit has second-order convergence in either clock.
    physical_ratio = physical_errors[1][0] / physical_errors[0][0]
    regularized_ratio = regularized_errors[1] / regularized_errors[0]
    assert 0.2 < physical_ratio < 0.3
    assert np.all((regularized_ratio > 0.2) & (regularized_ratio < 0.3))
    assert regularized_errors[0][-1] < physical_errors[0][-1] / 100
