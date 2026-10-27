"""Spinor regularization of eccentric bound orbits: their spinor lift, and equal-step integration comparisons."""

import numpy as np

from numga import stack
from examples.mechanics.kepler import core

GRAVITY = 1.0
SEMIMAJOR_AXIS = 1.0
ECCENTRICITIES = np.array([0.6, 0.9, 0.97])
PERIOD = 2 * np.pi * np.sqrt(SEMIMAJOR_AXIS ** 3 / GRAVITY)
FEATURED_CASE = 2
N_STEPS = 128
N_REFERENCE_SAMPLES = 1024
N_LIFT_SAMPLES = 64
STEP_COUNTS = np.array([64, 128, 256, 512])
N_CLOCK_ITERATIONS = 44
N_FRAMES = 120
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def initial_orbit() -> tuple[core.State, core.BoundOrbit]:
    """Start three eccentricities at apoapsis at time zero, with a common semimajor axis and orbital period."""
    apoapsis_radius = SEMIMAJOR_AXIS * (1 + ECCENTRICITIES)              # [cases]
    speed = np.sqrt(GRAVITY / SEMIMAJOR_AXIS * (1 - ECCENTRICITIES) / (1 + ECCENTRICITIES))  # [cases]
    spinor = core.mv.rotor() * np.sqrt(apoapsis_radius)                  # [cases] Spinor+
    velocity = core.mv.y.dual() * speed                                  # [cases] Point-
    time = core.mv.scalar() * np.zeros_like(apoapsis_radius)             # [cases] Scalar+
    start = core.State(time, spinor >> core.mv.x.dual(), velocity)       # [cases] State
    return start, core.BoundOrbit.lift(start, spinor, GRAVITY)


def lift(n_samples: int, n_revolutions: int) -> tuple[core.Point, core.Spinor]:
    _, orbit = initial_orbit()                                           # [cases] BoundOrbit
    parameter = np.linspace(0, n_revolutions * np.pi, n_samples + 1)[:, None] / orbit.frequency  # [samples + 1, cases] Scalar+
    spinors = orbit.spinor_at(parameter)                                 # [samples + 1, cases] Spinor+
    return (spinors >> core.mv.x.dual())[:, FEATURED_CASE], spinors[:, FEATURED_CASE]


def passage() -> tuple[core.State, core.State, core.State]:
    start, orbit = initial_orbit()                                       # [cases] State, BoundOrbit
    parameter = np.linspace(0, np.pi, N_REFERENCE_SAMPLES + 1)[:, None] / orbit.frequency  # [reference_samples + 1, cases] Scalar+
    reference = orbit.sample(parameter)                                  # [reference_samples + 1, cases] State
    physical = core.State.collect(core.physical_verlet(start, GRAVITY, PERIOD / N_STEPS, N_STEPS))  # [steps + 1, cases] State
    regularized = core.State.collect(core.regularized_verlet(
        orbit, start.time, np.pi / orbit.frequency / N_STEPS, N_STEPS))      # [steps + 1, cases] State
    return reference, physical, regularized


def convergence() -> tuple[core.Scalar, core.Scalar]:
    start, orbit = initial_orbit()                                       # [cases] State, BoundOrbit
    # Every resolution advances both second-order methods through one nominal orbit.
    physical = stack([orbit.position_error(core.State.collect(core.physical_verlet(
        start, GRAVITY, PERIOD / count, count)), N_CLOCK_ITERATIONS) for count in STEP_COUNTS])  # [counts, cases] Scalar+
    regularized = stack([orbit.position_error(core.State.collect(core.regularized_verlet(
        orbit, start.time, np.pi / orbit.frequency / count, count)), N_CLOCK_ITERATIONS) for count in STEP_COUNTS])  # [counts, cases] Scalar+
    return physical / SEMIMAJOR_AXIS, regularized / SEMIMAJOR_AXIS


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.kepler import render

    positions, spinors = lift(N_LIFT_SAMPLES, 1)
    save_figure(render.draw_lift(positions, spinors), "kepler_spinor_lift")
    reference, physical, regularized = passage()
    save_figure(render.draw_comparison(reference, physical, regularized), "kepler_passage")
    physical_error, regularized_error = convergence()
    save_figure(render.draw_errors(STEP_COUNTS, physical_error, regularized_error), "kepler_convergence")
    positions, spinors = lift(N_FRAMES, 2)
    save_animation(render.animate_lift(positions, spinors), "kepler_spinor_lift", DURATION_MS)


if __name__ == "__main__":
    main()
