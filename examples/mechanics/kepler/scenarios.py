"""Eccentric bound orbits, their spinor lift, and equal-step integration comparisons."""

import numpy as np

from numga import stack
from examples.mechanics.kepler import core

GRAVITY = 1.0
SEMIMAJOR_AXIS = 1.0
ECCENTRICITIES = np.array([0.6, 0.9, 0.97])
LABELS = ("Eccentricity 0.6", "Eccentricity 0.9", "Eccentricity 0.97")
FEATURED_CASE = 2
STEPS = 128
REFERENCE_SAMPLES = 1024
LIFT_SAMPLES = 64
COUNTS = np.array([64, 128, 256, 512])
CLOCK_ITERATIONS = 44
ANIMATION_FRAMES = 120
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def initial_orbit() -> core.BoundOrbit:
    """Start three eccentricities at apoapsis, with a common semimajor axis and orbital period."""
    distance = SEMIMAJOR_AXIS * (1 + ECCENTRICITIES)
    speed = np.sqrt(GRAVITY / SEMIMAJOR_AXIS * (1 - ECCENTRICITIES) / (1 + ECCENTRICITIES))
    spinor = core.mv.rotor() * np.sqrt(distance)                            # [cases] Spinor
    velocity = core.mv.y * speed                                          # [cases] Vector
    return core.BoundOrbit.lift(spinor, velocity, GRAVITY)


def lift(samples: int, revolutions: int) -> tuple[core.Vector, core.Spinor]:
    orbit = initial_orbit()
    frequency = (-orbit.energy / 2).square_root()
    phase = np.linspace(0, revolutions * np.pi, samples + 1)
    spinors = orbit.spinor * np.cos(phase[:, None]) + orbit.rate / frequency * np.sin(phase[:, None])
    return (spinors >> core.mv.x)[:, FEATURED_CASE], spinors[:, FEATURED_CASE]


def passage() -> tuple[core.State, core.State, core.State]:
    orbit = initial_orbit()
    frequency = (-orbit.energy / 2).square_root()
    period = 2 * np.pi * np.sqrt(SEMIMAJOR_AXIS ** 3 / GRAVITY)
    parameter = core.mv.scalar(np.linspace(0, np.pi, REFERENCE_SAMPLES + 1)[:, None, None]) / frequency
    reference = orbit.sample(parameter)
    initial = core.physical_state(orbit.spinor, orbit.rate, orbit.energy * 0)
    physical = core.State.collect(core.physical_verlet(initial, GRAVITY, period / STEPS, STEPS))
    regularized = core.State.collect(core.regularized_verlet(orbit, np.pi / frequency / STEPS, STEPS))
    return reference, physical, regularized


def convergence() -> tuple[core.Scalar, core.Scalar]:
    orbit = initial_orbit()
    frequency = (-orbit.energy / 2).square_root()
    period = 2 * np.pi * np.sqrt(SEMIMAJOR_AXIS ** 3 / GRAVITY)
    initial = core.physical_state(orbit.spinor, orbit.rate, orbit.energy * 0)
    # Every resolution advances both second-order methods through one nominal orbit.
    physical = stack([orbit.position_error(core.State.collect(core.physical_verlet(
        initial, GRAVITY, period / count, count)), CLOCK_ITERATIONS) for count in COUNTS])
    regularized = stack([orbit.position_error(core.State.collect(core.regularized_verlet(
        orbit, np.pi / frequency / count, count)), CLOCK_ITERATIONS) for count in COUNTS])
    return physical / SEMIMAJOR_AXIS, regularized / SEMIMAJOR_AXIS


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.kepler import render

    positions, spinors = lift(LIFT_SAMPLES, 1)
    save_figure(render.draw_lift(positions, spinors), "kepler_spinor_lift")
    reference, physical, regularized = passage()
    save_figure(render.draw_comparison(reference, physical, regularized, LABELS), "kepler_passage")
    physical_error, regularized_error = convergence()
    save_figure(render.draw_errors(COUNTS, physical_error, regularized_error, LABELS), "kepler_convergence")
    positions, spinors = lift(ANIMATION_FRAMES, 2)
    save_animation(render.animate_lift(positions, spinors), "kepler_spinor_lift", DURATION_MS)


if __name__ == "__main__":
    main()
