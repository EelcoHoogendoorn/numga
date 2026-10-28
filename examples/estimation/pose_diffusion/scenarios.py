"""Scenes for station keeping in gusts simulate a fleet of vessels holding a set point alongside the predicted covariance of their deviations.

Both scenes use the same controller and the same disturbances, which push much harder sideways than forward. In the first scene the body holds a fixed heading and its deviations stay mostly sideways; in the second it turns at a constant rate, rotating sideways disturbances into forward deviations and back so the spread becomes more isotropic.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.estimation.pose_diffusion.core import Covariance, Gusts, Line, Motor, Twist, covariance, mv, settled

# The controller relaxation gain (1/s), and the body's rotation rate in each scene.
RELAXATION = 0.6
rates = {"still": mv.xy * 0.0, "turning": mv.xy * 2.0}
# The body-frame push directions (sideways, forward, and in yaw) and their standard deviations per square
# root of a second, mostly sideways.
pushes = stack([mv.yw, mv.wx, mv.xy])                                           # [pushes] Twist+
STRENGTHS = np.array([0.35, 0.08, 0.15])
gusts = (pushes * STRENGTHS * (pushes & Line)).sum(axis=-1)                     # [] Twist+ <- Line-
# The length of a run and its time step in seconds, and the number of simulated bodies.
SECONDS = 8.0
DT = 0.01
N_BODIES = 3000
# The steps between frames of the animation, and the seed of the disturbances.
N_STEPS_PER_FRAME = 5
SEED = 0
# The half-width of the square each cloud is drawn in, and the frame duration in milliseconds.
EXTENT = 1.6
DURATION_MS = 50


# --- math -----------------------------------------------------------------------------
def diffuse(
    rate: Twist, gusts: Gusts, seconds: float, dt: float, n_bodies: int, every: int, seed: int,
) -> Iterator[tuple[Motor, Covariance, Covariance]]:
    """The bodies' poses, predicted covariance, and steady-state covariance every `every` steps of a fleet holding the commanded pose."""
    # If the body turns on the spot, the error turns with it by the commutator; the controller shrinks it at RELAXATION.
    dynamics = Twist.commutator(rate) - RELAXATION * Twist                      # [] Twist+ <- Twist+
    noise = covariance(gusts)                                                   # [] Twist+ <- Line-
    limit = settled(dynamics, noise)                                            # [] Twist+ <- Line-
    rng = np.random.default_rng(seed)
    # All bodies start at the commanded pose with zero covariance.
    errors = mv.bivector(np.zeros((n_bodies, 3)))                               # [bodies] Twist+
    predicted = noise * 0.0                                                     # [] Twist+ <- Line-
    for count in range(int(seconds / dt)):
        # White noise integrated over the step, and the dynamics acting on the covariance's twist output and line input.
        white = mv.antibivector(rng.normal(size=(n_bodies, 3))) * np.sqrt(dt)   # [bodies] Line-
        errors = errors + dynamics(errors) * dt + gusts(white)                  # [bodies] Twist+
        growth = dynamics(predicted) + predicted(dynamics.adjugate()) + noise   # [] Twist+ <- Line-
        predicted = predicted + growth * dt                                     # [] Twist+ <- Line-
        if (count + 1) % every == 0:
            # The exponential of half a twist is its motor.
            yield (errors * 0.5).exp(), predicted, limit


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.estimation.pose_diffusion import render

    runs = {name: list(diffuse(rate, gusts, SECONDS, DT, N_BODIES, N_STEPS_PER_FRAME, SEED)) for name, rate in rates.items()}
    save_animation(render.animate_clouds(runs, EXTENT), "pose_diffusion", DURATION_MS)
    save_figure(render.draw_settled({name: states[-1] for name, states in runs.items()}, EXTENT), "pose_diffusion")

    # checks
    # By the last frame the integrated prediction has converged to the steady-state covariance.
    for states in runs.values():
        _, predicted, limit = states[-1]
        np.testing.assert_allclose(predicted.kernel, limit.kernel, atol=0.05 * np.abs(limit.kernel).max())


if __name__ == "__main__":
    main()
