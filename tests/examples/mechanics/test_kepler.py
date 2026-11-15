"""Coverage tests verify tilted-plane spinor lifts, gauge invariance, clock inversion, and Kepler figure rendering."""

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.kepler import core, render, scenarios


def test_tilted_orbit_lift_gauge_invariance_and_clock_inversion():
    """Tilted 3D orbits reconstruct velocity, conserve orbital invariants, ignore gauge spin, and invert physical time."""
    tilt_angles = np.array([0.2, 0.5, -0.3])
    n_tilts = len(tilt_angles)
    n_samples = 5
    rotation = (core.mv.yz * tilt_angles).exp() * (core.mv.xy * (tilt_angles / 2)).exp()  # [n_tilts] Spinor+
    spinor = rotation * np.sqrt(1.8)                                     # [n_tilts] Spinor+
    velocity = rotation >> (core.mv.x * 0.13 + core.mv.y * 0.3).dual()   # [n_tilts] Point-
    initial_position = spinor >> core.mv.x.dual()                        # [n_tilts] Point-
    energy = velocity.scalar_norm_squared() / 2 - 1.0 / initial_position.norm()  # [n_tilts] Scalar+
    rate = core.velocity_lift(velocity, spinor)                          # [n_tilts] Spinor+
    frequency = (-energy / 2).square_root()                              # [n_tilts] Scalar+

    reconstructed_position, reconstructed_velocity = core.physical_state(spinor, rate)  # [n_tilts] Point-, Point-
    parameter = np.linspace(0, np.pi, n_samples)[:, None] / frequency    # [n_samples, n_tilts] Scalar+
    times, positions, velocities = core.exact_orbit(spinor, rate, frequency, parameter)  # [n_samples, n_tilts] Scalar+, Point-, Point-
    semimajor = -1.0 / (2 * energy)                                      # [n_tilts] Scalar+
    period = 2 * np.pi * (semimajor ** 3).square_root()                  # [n_tilts] Scalar+
    trajectory_energies = velocities.scalar_norm_squared() / 2 - 1.0 / positions.norm()  # [n_samples, n_tilts] Scalar+

    gauge_rotor = (core.mv.yz * np.array([0.3, -0.5, 0.7])).exp()        # [n_tilts] Spinor+
    gauged_times, gauged_positions, gauged_velocities = core.exact_orbit(
        spinor * gauge_rotor, rate * gauge_rotor, frequency, parameter)  # [n_samples, n_tilts] Scalar+, Point-, Point-

    inverted_times, inverted_positions, inverted_velocities = core.state_at_time(
        spinor, rate, frequency, times[1:-1], scenarios.n_clock_iterations)  # [3, n_tilts] Scalar+, Point-, Point-

    # checks
    assert ((reconstructed_position - initial_position).norm() < 1e-10).all()
    assert ((reconstructed_velocity - velocity).norm() < 1e-10).all()
    assert ((core.placement(rate, spinor) - core.placement(spinor, rate)).norm() < 1e-10).all()
    assert ((positions[-1] - initial_position).norm() < 1e-10).all()
    assert ((velocities[-1] - velocity).norm() < 1e-10).all()
    assert ((times[-1] - period).abs() < 1e-9).all()
    assert ((trajectory_energies - energy).abs() < 1e-9).all()
    assert (((positions & velocities) - (initial_position & velocity)).norm() < 1e-10).all()
    assert ((gauged_positions - positions).norm() < 1e-10).all()
    assert ((gauged_velocities - velocities).norm() < 1e-9).all()
    assert ((gauged_times - times).abs() < 1e-9).all()
    assert ((inverted_times - times[1:-1]).abs() < 1e-9).all()
    assert ((inverted_positions - positions[1:-1]).norm() < 1e-9).all()
    assert ((inverted_velocities - velocities[1:-1]).norm() < 1e-9).all()


def test_kepler_verlet_and_views():
    """Minimal Verlet steps and scenarios render all static figures and one animation frame without writing files."""
    render.lift_figure(*scenarios.lift_curve(2, 1))
    render.passage_figure(*scenarios.passage(1, 2))
    counts = np.array([1, 2])
    physical_errors, regularized_errors = scenarios.convergence(counts, 2)
    render.convergence_figure(counts, physical_errors, regularized_errors)
    frame = next(render.lift_frames(scenarios.featured_lift_states(2, 2)))
    plt.close("all")

    # checks
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert (regularized_errors >= 0).all()
    assert (physical_errors >= 0).all()
