"""Scenes for a pose held against random disturbances: a simulated set of bodies next to the
predicted covariance of their deviations.

Both scenes use the same controller and the same disturbances, which are much larger sideways
than forward. In the first the body does not turn, and its deviations stay mostly sideways. In
the second it turns at a constant rate. The rotation turns sideways disturbances into forward
deviations and back, so the deviations spread more evenly over all directions.
"""

from __future__ import annotations

import numpy as np

from examples.geometry.pose_diffusion import core

mv, Line = core.mv, core.Line
RELAXATION = 0.6                                                              # controller gain, per second
RATES = {"still": mv.xy * 0.0, "turning": mv.xy * 2.0}                        # the body's rotation rate in each scene


# --- plumbing -------------------------------------------------------------------------
def shaping(sideways: float, forward: float, turning: float) -> core.Kicks:
    """The map from white noise to disturbance twists.

    The arguments are the standard deviations per square root of a second of the sideways, forward
    and rotational disturbance, in the body frame.
    """
    return mv.yw * sideways * (mv.yw & Line) + mv.wx * forward * (mv.wx & Line) + mv.xy * turning * (mv.xy & Line)


KICKS = shaping(0.35, 0.08, 0.15)                                             # mostly sideways


# --- math -----------------------------------------------------------------------------
def diffuse(rate: core.Twist, kicks: core.Kicks, seconds: float, dt: float, bodies: int, every: int, seed: int):
    """Simulate many bodies holding the commanded pose, and integrate their predicted covariance.

    All bodies start at the commanded pose and the predicted covariance starts at zero. In each
    step every body receives an independent disturbance, and the prediction is advanced with the
    same dynamics. The steady-state covariance is computed once at the start. Every few steps this
    yields the bodies' poses, the predicted covariance and the steady-state covariance.
    """
    dynamics, covariance = core.drift(rate, RELAXATION), core.covariance(kicks)
    limit = core.settled(dynamics, covariance)
    rng = np.random.default_rng(seed)
    errors = mv.bivector(np.zeros((bodies, 3)))                               # [bodies] Twist: all bodies at the commanded pose
    predicted = covariance * 0.0                                              # zero covariance at the start
    for count in range(int(seconds / dt)):
        if count % every == 0:
            yield (errors * 0.5).exp(), predicted, limit                   # the exponential of a twist is a motor
        white = mv.vector(rng.normal(size=(bodies, 3))) * np.sqrt(dt)         # [bodies] Line: white noise integrated over the step
        errors, predicted = core.step(dynamics, covariance, errors, predicted, kicks(white), dt)

    # --- checks
    # The steady-state covariance has zero rate of change, the integrated prediction has converged
    # to it, and the sample covariance of the simulated bodies matches it within sampling error.
    np.testing.assert_allclose(core.growth(dynamics, limit, covariance).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose(predicted.kernel, limit.kernel, atol=0.05 * np.abs(limit.kernel).max())
    empirical = (errors * (errors & Line)).mean(axis=0)                       # Twist <- Line
    np.testing.assert_allclose(empirical.kernel, limit.kernel, atol=0.25 * np.abs(limit.kernel).max())


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.geometry.pose_diffusion import render

    runs = {name: list(diffuse(rate, KICKS, 8.0, 0.01, 3000, 5, 0)) for name, rate in RATES.items()}
    save_animation(render.animate_clouds(runs, 0.05, 1.6), "pose_diffusion", 50)
    save_figure(render.draw_settled({name: states[-1] for name, states in runs.items()}, 1.6), "pose_diffusion")
