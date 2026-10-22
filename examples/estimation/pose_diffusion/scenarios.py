"""Station keeping in gusts: a simulated set of bodies next to the predicted covariance of their
deviations.

Both scenes use the same controller and the same disturbances, which are much larger sideways
than forward. In the first the body does not turn, and its deviations stay mostly sideways. In
the second it turns at a constant rate. The rotation turns sideways disturbances into forward
deviations and back, so the deviations spread more evenly over all directions.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.estimation.pose_diffusion import core

mv, Line = core.mv, core.Line
# The controller gain, and the body's rotation rate in each scene.
RELAXATION = 0.6                                                              # 1/s
RATES = {"still": mv.xy * 0.0, "turning": mv.xy * 2.0}
# The length of a run and its time step, in seconds, and the number of simulated bodies.
SECONDS = 8.0
DT = 0.01
N_BODIES = 3000
# The steps between frames of the animation, and the seed of the disturbances.
N_STEPS_PER_FRAME = 5
SEED = 0
# The half-width of the square each cloud is drawn in, and the time per frame of the animation.
EXTENT = 1.6
DURATION_MS = 50


# --- plumbing -------------------------------------------------------------------------
def shaping(strengths: np.ndarray) -> core.Gusts:
    """The map from white noise to disturbance twists: a dyad for each push direction in the body
    frame, sideways, forward and in yaw, at its standard deviation per square root of a second."""
    pushes = stack([mv.yw, mv.wx, mv.xy])                                     # [pushes] Twist
    return (pushes * strengths * (pushes & Line)).sum(axis=-1)                # [] Twist <- Line


# Mostly sideways.
GUSTS = shaping(np.array([0.35, 0.08, 0.15]))                                             # [] Twist <- Line


# --- math -----------------------------------------------------------------------------
def diffuse(rate: core.Twist, gusts: core.Gusts, seconds: float, dt: float, n_bodies: int, every: int, seed: int):
    """Simulate many bodies holding the commanded pose, and integrate their predicted covariance.

    All bodies start at the commanded pose and the predicted covariance starts at zero. In each
    step every body receives an independent disturbance, and the prediction is advanced with the
    same dynamics. The steady-state covariance is computed once at the start. Every few steps this
    yields the bodies' poses, the predicted covariance and the steady-state covariance.
    """
    dynamics, covariance = core.drift(rate, RELAXATION), core.covariance(gusts)
    limit = core.settled(dynamics, covariance)
    rng = np.random.default_rng(seed)
    # All bodies at the commanded pose, and zero covariance at the start.
    errors = mv.bivector(np.zeros((n_bodies, 3)))                               # [bodies] Twist
    predicted = covariance * 0.0                                              # [] Twist <- Line
    for count in range(int(seconds / dt)):
        if count % every == 0:
            # The exponential of a twist is a motor.
            yield (errors * 0.5).exp(), predicted, limit
        # White noise integrated over the step.
        white = mv.antibivector(rng.normal(size=(n_bodies, 3))) * np.sqrt(dt)   # [bodies] Line
        errors, predicted = core.step(dynamics, covariance, errors, predicted, gusts(white), dt)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.estimation.pose_diffusion import render

    runs = {name: list(diffuse(rate, GUSTS, SECONDS, DT, N_BODIES, N_STEPS_PER_FRAME, SEED)) for name, rate in RATES.items()}
    save_animation(render.animate_clouds(runs, EXTENT), "pose_diffusion", DURATION_MS)
    save_figure(render.draw_settled({name: states[-1] for name, states in runs.items()}, EXTENT), "pose_diffusion")

    # checks
    # By the last frame the integrated prediction has converged to the steady-state covariance.
    for states in runs.values():
        _, predicted, limit = states[-1]
        np.testing.assert_allclose(predicted.kernel, limit.kernel, atol=0.05 * np.abs(limit.kernel).max())


if __name__ == "__main__":
    main()
