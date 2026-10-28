"""Scenes for pose-graph odometry recover the most likely trajectory around a closed lap by Gauss-Newton from dead reckoning, in the plane and in space.

The core module is instantiated for the algebra of the true steps, so the same functions run on a planar lap in `PGA2D` and on a spatial lap in `PGA3D`.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from numga import concatenate
from numga.algebras import PGA2D, PGA3D
from examples import instantiate

CORE = "examples.estimation.odometry.core"
# The steps of a lap in the plane, and of a short one in space, and the length of every step in metres.
N_PLANE_STEPS = 36
N_SPACE_STEPS = 6
STEP_LENGTH = 0.7
# How much the turn rate varies in the plane, and how far a step slips sideways, sways out of the plane and
# rises or falls.
TURN_VARIATION = 0.4
SLIP = 0.1
SWAY = 0.1
RISE = 0.05
# How far a step reading and the closing reading, recognising the start, are trusted, how well the first
# pose is known, and how vaguely the others are known before any reading, as translation and rotation
# standard deviations.
READING_STD = (0.03, 0.01)
CLOSING_STD = (0.001, 0.0003)
KNOWN_STD = (0.01, 0.01)
VAGUE_STD = (100.0, 30.0)
# The full Gauss-Newton steps, the share of each correction a damped step takes and their count, the seed of
# the readings' errors, and the duration of a frame in milliseconds.
N_FULL_STEPS = 3
DAMPING = 0.2
N_DAMPED_STEPS = 20
SEED = 9
FRAME_DURATION = 80
# How far out the uncertainty ellipsoids lie, in standard deviations.
SIGMAS = 2.0


# --- math -----------------------------------------------------------------------------
def damped(steps, damping: float, n_iterations: int, seed: int):
    """Damped Gauss-Newton from dead reckoning yields the poses with each pose's covariance across the iterations."""
    core = instantiate(CORE, steps.algebra)
    truth, dead_reckoning, readings, noises, tails, heads, anchors, priors = survey(core, steps, seed)
    weights, anchor_weights = noises.inverse(), priors.inverse()               # [readings] Forque- <- Twist+, [poses] Forque- <- Twist+
    reckoned_uncertainty = core.reckon(dead_reckoning, noises, priors)         # [poses] Twist+ <- Forque-
    poses = core.gauss_newton(dead_reckoning, readings, weights, tails, heads, anchors, anchor_weights, damping, n_iterations)
    iterates = core.uncertainties(poses, reckoned_uncertainty, damping, noises, tails, heads, priors, weights, anchor_weights)
    return truth, dead_reckoning, reckoned_uncertainty, iterates


def closing(steps, n_iterations: int, seed: int):
    """Full Gauss-Newton steps from dead reckoning reach the most likely poses, with their marginal covariances."""
    core = instantiate(CORE, steps.algebra)
    truth, dead_reckoning, readings, noises, tails, heads, anchors, priors = survey(core, steps, seed)
    weights, anchor_weights = noises.inverse(), priors.inverse()               # [readings] Forque- <- Twist+, [poses] Forque- <- Twist+
    reckoned_uncertainty = core.reckon(dead_reckoning, noises, priors)         # [poses] Twist+ <- Forque-
    *_, poses = core.gauss_newton(dead_reckoning, readings, weights, tails, heads, anchors, anchor_weights, 1.0, n_iterations)  # [poses] Motor+
    uncertainty = core.marginals(poses, noises, tails, heads, priors, weights, anchor_weights)  # [poses] Twist+ <- Forque-
    return truth, dead_reckoning, reckoned_uncertainty, poses, uncertainty


def gradients(steps, poses, seed: int):
    """The pose-graph gradient at the dead-reckoned poses and at the given poses."""
    core = instantiate(CORE, steps.algebra)
    truth, dead_reckoning, readings, noises, tails, heads, anchors, priors = survey(core, steps, seed)
    weights, anchor_weights = noises.inverse(), priors.inverse()               # [readings] Forque- <- Twist+, [poses] Forque- <- Twist+
    return (core.gradient(dead_reckoning, readings, weights, tails, heads, anchors, anchor_weights),
            core.gradient(poses, readings, weights, tails, heads, anchors, anchor_weights))


# --- plumbing -------------------------------------------------------------------------
def survey(core: ModuleType, steps, seed: int):
    """A simulated lap along the true steps gives the true and dead-reckoned poses, the noisy step and loop-closing readings with their covariances and endpoint indices, and every pose's anchor and prior covariance."""
    rng = np.random.default_rng(seed)
    # Each step composes on the right in the robot's own frame; a running right product equals the
    # reverse of the running left product of the reverses.
    truth = concatenate([core.mv.rotor()[None], steps.reverse().cumprod(axis=0).reverse()])  # [poses] Motor+
    tails = np.append(np.arange(len(steps)), 0)                                # [readings]
    heads = np.append(np.arange(1, len(steps) + 1), len(steps))                # [readings]
    noises = concatenate([core.isotropic(*READING_STD) * np.ones(len(steps)), core.isotropic(*CLOSING_STD)[None]])  # [readings] Twist+ <- Forque-
    noise_modes = core.modes(noises)                                           # [readings, modes] Twist+
    errors = (noise_modes * rng.normal(size=noise_modes.shape)).sum(axis=-1)   # [readings] Twist+
    readings = (truth[tails].inverse() * truth[heads]) * (errors * 0.5).exp()  # [readings] Motor+
    dead_reckoning = concatenate([truth[:1], truth[0] * readings[:len(steps)].reverse().cumprod(axis=0).reverse()])  # [poses] Motor+
    anchors = concatenate([truth[:1], dead_reckoning[1:]])                     # [poses] Motor+
    priors = concatenate([core.isotropic(*KNOWN_STD)[None], core.isotropic(*VAGUE_STD) * np.ones(len(steps))])  # [poses] Twist+ <- Forque-
    return truth, dead_reckoning, readings, noises, tails, heads, anchors, priors


def lap_in_plane():
    """The true step motors of a closed lap of `N_PLANE_STEPS` steps in the plane."""
    mv = instantiate(CORE, PGA2D).mv
    phase = 2 * np.pi * np.arange(N_PLANE_STEPS) / N_PLANE_STEPS
    turn = mv.xy * (1 + TURN_VARIATION * np.sin(2 * phase)) * 2 * np.pi / N_PLANE_STEPS  # [steps] Twist+
    slip = mv.yw * SLIP * np.cos(2 * phase)                                    # [steps] Twist+
    return ((turn + slip + mv.xw * STEP_LENGTH) * 0.5).exp()                   # [steps] Motor+


def lap_in_space():
    """The true step motors of a closed lap of `N_SPACE_STEPS` steps in space."""
    mv = instantiate(CORE, PGA3D).mv
    phase = 2 * np.pi * np.arange(N_SPACE_STEPS) / N_SPACE_STEPS
    turn = mv.xy * 2 * np.pi / N_SPACE_STEPS                                   # [] Twist+
    sway = (mv.zx * np.cos(phase) + mv.yz * np.sin(phase)) * SWAY              # [steps] Twist+
    slip = mv.yw * SLIP * np.cos(2 * phase) + mv.zw * RISE * np.sin(2 * phase)  # [steps] Twist+
    return ((turn + sway + slip + mv.xw * STEP_LENGTH) * 0.5).exp()            # [steps] Motor+


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.estimation.odometry import render

    # Dead reckoning drifts half a metre over the lap.
    plane = instantiate(CORE, PGA2D)
    truth, dead_reckoning, reckoned_uncertainty, poses, uncertainty = closing(lap_in_plane(), N_FULL_STEPS, SEED)
    reckoned_ellipsoids = plane.ellipsoids(dead_reckoning, reckoned_uncertainty, SIGMAS)  # [poses] Plane+ <- Point-
    save_figure(render.draw(truth, dead_reckoning, reckoned_ellipsoids, poses, plane.ellipsoids(poses, uncertainty, SIGMAS)), "odometry")
    # Closing the lap a fifth of the way at a time.
    *_, iterates = damped(lap_in_plane(), DAMPING, N_DAMPED_STEPS, SEED)
    save_animation(render.animate(truth, dead_reckoning, reckoned_ellipsoids,
                                  ((poses, plane.ellipsoids(poses, uncertainty, SIGMAS)) for poses, uncertainty in iterates)),
                   "odometry", FRAME_DURATION)

    # checks
    # The gradient falls a millionfold from dead reckoning to the most likely poses, in the plane and, with
    # the same core, in space.
    *_, poses_in_space, _ = closing(lap_in_space(), N_FULL_STEPS, SEED)
    for steps, most_likely in ((lap_in_plane(), poses), (lap_in_space(), poses_in_space)):
        dead_gradient, optimum_gradient = gradients(steps, most_likely, SEED)
        np.testing.assert_allclose(optimum_gradient.kernel, 0.0, atol=1e-6 * np.abs(dead_gradient.kernel).max())


if __name__ == "__main__":
    main()
