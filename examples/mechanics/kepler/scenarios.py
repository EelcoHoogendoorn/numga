"""Three eccentric bound Kepler orbits compare physical-clock and spinor-clock Verlet integration against the exact spinor lift."""

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.mechanics.kepler import core

gravity = 1.0
semimajor_axis = 1.0
eccentricities = np.array([0.6, 0.9, 0.97])
n_cases = len(eccentricities)
period = 2 * np.pi * np.sqrt(semimajor_axis ** 3 / gravity)
featured_case = 2
n_steps = 128
n_reference_samples = 1024
n_lift_samples = 64
step_counts = np.array([64, 128, 256, 512])
n_step_counts = len(step_counts)
n_clock_iterations = 44
n_frames = 180
n_revolutions = 2
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def initial_orbits() -> tuple[core.Spinor, core.Spinor, core.Scalar, core.Scalar]:
    """Apoapsis launch conditions initialize the three eccentric orbits as `(spinor, rate, energy, frequency)` across `[n_cases]`."""
    apoapsis_radius = semimajor_axis * (1 + eccentricities)              # [n_cases]
    apoapsis_speed = np.sqrt(gravity / semimajor_axis * (1 - eccentricities) / (1 + eccentricities))  # [n_cases]
    spinor = core.mv.rotor() * np.sqrt(apoapsis_radius)                  # [n_cases] Spinor+
    apoapsis_velocity = core.mv.y.dual() * apoapsis_speed                # [n_cases] Point-
    energy = apoapsis_velocity.scalar_norm_squared() / 2 - gravity / spinor.scalar_norm_squared()  # [n_cases] Scalar+
    rate = core.velocity_lift(apoapsis_velocity, spinor)                 # [n_cases] Spinor+
    frequency = (-energy / 2).square_root()                              # [n_cases] Scalar+
    return spinor, rate, energy, frequency


def lift_curve(n_samples: int, revolutions: int) -> tuple[core.Point, core.Spinor]:
    """Uniform regularized-clock sampling evaluates the featured eccentric orbit and its spinor across `n_samples + 1` steps."""
    spinor, rate, energy, frequency = initial_orbits()                   # [n_cases] Spinor+, Spinor+, Scalar+, Scalar+
    parameter = np.linspace(0, revolutions * np.pi, n_samples + 1)[:, None] / frequency  # [n_samples + 1, n_cases] Scalar+
    spinors = core.oscillator_spinor(spinor, rate, frequency, parameter)  # [n_samples + 1, n_cases] Spinor+
    return (spinors >> core.mv.x.dual())[:, featured_case], spinors[:, featured_case]


def passage(n_orbit_steps: int, n_samples: int) -> tuple[core.Point, core.Point, core.Point]:
    """Equal-step Verlet runs in both clocks integrate one orbital period alongside an `n_samples + 1` exact reference path."""
    spinor, rate, energy, frequency = initial_orbits()                   # [n_cases] Spinor+, Spinor+, Scalar+, Scalar+
    initial_time = 0 * spinor.scalar_norm_squared()                      # [n_cases] Scalar+
    initial_positions, initial_velocities = core.physical_state(spinor, rate)  # [n_cases] Point-, Point-
    parameter = np.linspace(0, np.pi, n_samples + 1)[:, None] / frequency  # [n_samples + 1, n_cases] Scalar+
    reference_positions = core.exact_orbit(spinor, rate, frequency, parameter)[1]  # [n_samples + 1, n_cases] Point-
    physical_samples = list(core.physical_verlet(
        initial_positions, initial_velocities, initial_time, gravity, period / n_orbit_steps, n_orbit_steps))
    regularized_samples = list(core.regularized_verlet(
        spinor, rate, energy, initial_time, np.pi / frequency / n_orbit_steps, n_orbit_steps))
    physical_positions = stack([sample[1] for sample in physical_samples])      # [n_orbit_steps + 1, n_cases] Point-
    regularized_positions = stack([sample[1] for sample in regularized_samples])  # [n_orbit_steps + 1, n_cases] Point-
    return reference_positions, physical_positions, regularized_positions


def convergence(counts: np.ndarray, n_iterations: int) -> tuple[core.Scalar, core.Scalar]:
    """Step-count sweeps evaluate normalized position errors (`[n_counts, n_cases] Scalar+`) across `counts` steps per orbit in both clocks."""
    n_counts = len(counts)
    spinor, rate, energy, frequency = initial_orbits()                   # [n_cases] Spinor+, Spinor+, Scalar+, Scalar+
    initial_time = 0 * spinor.scalar_norm_squared()                      # [n_cases] Scalar+
    initial_positions, initial_velocities = core.physical_state(spinor, rate)  # [n_cases] Point-, Point-
    # Every resolution advances both second-order integrators through one nominal orbital period.
    physical_errors = stack([
        core.position_error(
            spinor, rate, frequency,
            core.physical_verlet(initial_positions, initial_velocities, initial_time,
                                 gravity, period / int(n_orbit_steps), int(n_orbit_steps)),
            n_iterations)
        for n_orbit_steps in counts
    ])                                                                   # [n_counts, n_cases] Scalar+
    regularized_errors = stack([
        core.position_error(
            spinor, rate, frequency,
            core.regularized_verlet(spinor, rate, energy, initial_time,
                                    np.pi / frequency / int(n_orbit_steps), int(n_orbit_steps)),
            n_iterations)
        for n_orbit_steps in counts
    ])                                                                   # [n_counts, n_cases] Scalar+
    return physical_errors / semimajor_axis, regularized_errors / semimajor_axis


def featured_lift_states(n_orbit_frames: int, revolutions: int) -> Iterator[tuple[core.Point, core.Spinor]]:
    """Yields `(position, spinor)` states for the featured eccentric orbit over `revolutions` orbital periods."""
    spinor, rate, energy, frequency = initial_orbits()                   # [n_cases] Spinor+, Spinor+, Scalar+, Scalar+
    return core.lift_states(spinor[featured_case], rate[featured_case], frequency[featured_case],
                            revolutions, n_orbit_frames)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Scenario execution writes all Kepler figures and the spinor-lift animation to `plots/` and verifies orbital invariants."""
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.kepler import render

    lift_positions, lift_spinors = lift_curve(n_lift_samples, 1)
    save_figure(render.lift_figure(lift_positions, lift_spinors), "kepler_spinor_lift")
    reference_positions, physical_positions, regularized_positions = passage(n_steps, n_reference_samples)
    save_figure(render.passage_figure(reference_positions, physical_positions, regularized_positions), "kepler_passage")
    physical_errors, regularized_errors = convergence(step_counts, n_clock_iterations)
    save_figure(render.convergence_figure(step_counts, physical_errors, regularized_errors), "kepler_convergence")

    revolving_states = []

    def streamed_states() -> Iterator[tuple[core.Point, core.Spinor]]:
        for state in featured_lift_states(n_frames, n_revolutions):
            revolving_states.append(state)
            yield state

    save_animation(render.lift_frames(streamed_states()), "kepler_spinor_lift", frame_duration_ms)

    # checks
    assert ((lift_positions[-1] - lift_positions[0]).norm() < 1e-10).all()
    assert ((lift_spinors[-1] + lift_spinors[0]).norm() < 1e-10).all()
    assert ((reference_positions[-1] - reference_positions[0]).norm() < 1e-10).all()
    assert (regularized_errors[-1] < regularized_errors[-2] / 3.5).all()
    assert regularized_errors[-1, featured_case] < physical_errors[-1, featured_case] / 100
    assert ((revolving_states[n_frames // 2][0] - revolving_states[0][0]).norm() < 1e-10).all()
    assert ((revolving_states[n_frames // 2][1] + revolving_states[0][1]).norm() < 1e-10).all()
    assert ((revolving_states[-1][1] - revolving_states[0][1]).norm() < 1e-10).all()


if __name__ == "__main__":
    main()
