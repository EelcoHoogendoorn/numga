"""Spinor regularization preserves Kepler geometry; and the passages and the convergence take a step or two and
draw."""

import numpy as np

from examples.mechanics.kepler import core, render, scenarios


def test_lift_reconstructs_velocity_and_exact_motion_in_tilted_planes():
    angles = np.array([0.2, 0.5, -0.3])
    rotation = (core.mv.yz * angles).exp() * (core.mv.xy * (angles / 2)).exp()
    spinor = rotation * np.sqrt(1.8)
    velocity = rotation >> (core.mv.x * 0.13 + core.mv.y * 0.3).dual()
    start = core.State(core.mv.scalar() * np.zeros_like(angles), spinor >> core.mv.x.dual(), velocity)
    orbit = core.BoundOrbit.lift(start, spinor, 1.0)
    initial = core.State.from_spinor(orbit.spinor, orbit.rate, start.time)
    trajectory = orbit.sample(np.linspace(0, np.pi, 81)[:, None] / orbit.frequency)
    semimajor = -1 / (2 * orbit.energy)
    period = 2 * np.pi * (semimajor ** 3).square_root()

    # checks
    np.testing.assert_allclose((initial.velocity - velocity).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((core.placement(orbit.rate, orbit.spinor)
                                - core.placement(orbit.spinor, orbit.rate)).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((trajectory.position[-1] - initial.position).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((trajectory.velocity[-1] - initial.velocity).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((trajectory.time[-1] - period).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((trajectory.energy(1.0) - orbit.energy).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((trajectory.momentum() - initial.momentum()).kernel, 0, atol=1e-11)


def test_spinor_gauge_changes_neither_orbit_nor_clock():
    _, orbit = scenarios.initial_orbit()
    phase = np.linspace(0, 2 * np.pi, 91)[:, None]
    gauge = (core.mv.yz * np.array([0.3, -0.5, 0.7])).exp()
    shifted = core.BoundOrbit(orbit.spinor * gauge, orbit.rate * gauge, orbit.energy)
    first, second = orbit.sample(phase), shifted.sample(phase)

    # checks
    np.testing.assert_allclose((first.position - second.position).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((first.velocity - second.velocity).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((first.time - second.time).kernel, 0, atol=1e-10)


def test_physical_clock_inversion_and_inverse_square_acceleration():
    _, orbit = scenarios.initial_orbit()
    times = np.linspace(0.2, 1.0, 7)[:, None]
    delta = 1e-4
    state = orbit.at_time(times, scenarios.N_CLOCK_ITERATIONS)
    before = orbit.at_time(times - delta, scenarios.N_CLOCK_ITERATIONS)
    after = orbit.at_time(times + delta, scenarios.N_CLOCK_ITERATIONS)
    acceleration = (after.velocity - before.velocity) / (2 * delta)
    force = -scenarios.GRAVITY * state.position / state.position.norm() ** 3

    # checks
    np.testing.assert_allclose((state.time - times).kernel, 0, atol=1e-10)
    np.testing.assert_allclose(acceleration.kernel, force.kernel, atol=1e-7, rtol=1e-6)


def test_the_lift_the_passages_and_the_convergence_draw():
    render.draw_lift(*scenarios.lift(2, 1))
    render.draw_comparison(*scenarios.passage(1, 2))
    step_counts = np.array([1, 2])
    render.draw_errors(step_counts, *scenarios.convergence(step_counts, 1))
    next(render.animate_lift(*scenarios.lift(1, 2)))
