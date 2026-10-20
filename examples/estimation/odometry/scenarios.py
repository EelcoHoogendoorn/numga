"""The most likely trajectory from odometry: the poses most likely given every reading, the steps and
then the one that closes the loop, by Gauss-Newton from dead reckoning; in the plane, and in space.

The core is instantiated for the algebra of the true steps, so the same scenes run on a lap in PGA2D
and on one in PGA3D.
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
READING = (0.03, 0.01)
CLOSING = (0.001, 0.0003)
KNOWN = (0.01, 0.01)
VAGUE = (100.0, 30.0)
# The full Gauss-Newton steps, the share of each correction a damped step takes and their count, the seed of
# the readings' errors, and the duration of a frame in milliseconds.
N_FULL_STEPS = 3
DAMPING = 0.2
N_DAMPED_STEPS = 20
SEED = 9
FRAME_DURATION = 80


# --- math -----------------------------------------------------------------------------
def damped(steps, damping: float, n_iterations: int, seed: int):
    """Gauss-Newton with every reading from the dead-reckoned poses, damped by the given share, and each
    pose's uncertainty along it from dead reckoning's. Returns the true and dead-reckoned poses, dead
    reckoning's uncertainty, and the iterations, yielding the poses and their uncertainties after each."""
    core = instantiate(CORE, steps.algebra)
    truth, dead, readings, noises, tails, heads, anchors, priors = survey(core, steps, seed)
    weights, anchor_weights = noises.inverse(), priors.inverse()               # [readings] Information, [poses] Information
    reckoned = core.reckon(dead, noises, priors)                               # [poses] Covariance
    poses = core.gauss_newton(dead, readings, weights, tails, heads, anchors, anchor_weights, damping, n_iterations)
    iterates = core.uncertainties(poses, reckoned, damping, noises, tails, heads, priors, weights, anchor_weights)
    return truth, dead, reckoned, iterates


def closing(steps, n_iterations: int, seed: int):
    """The most likely poses given every reading, by the given full Gauss-Newton iterations. Returns the
    true and dead-reckoned poses, dead reckoning's uncertainty, and the most likely poses with theirs."""
    core = instantiate(CORE, steps.algebra)
    truth, dead, readings, noises, tails, heads, anchors, priors = survey(core, steps, seed)
    weights, anchor_weights = noises.inverse(), priors.inverse()               # [readings] Information, [poses] Information
    reckoned = core.reckon(dead, noises, priors)                               # [poses] Covariance
    *_, poses = core.gauss_newton(dead, readings, weights, tails, heads, anchors, anchor_weights, 1.0, n_iterations)   # [poses] Motor
    uncertainty = core.marginals(poses, noises, tails, heads, priors, weights, anchor_weights)   # [poses] Covariance
    return truth, dead, reckoned, poses, uncertainty


def gradients(steps, poses, seed: int):
    """The gradient of the objective of a lap along the true steps at dead reckoning and at the given
    poses."""
    core = instantiate(CORE, steps.algebra)
    truth, dead, readings, noises, tails, heads, anchors, priors = survey(core, steps, seed)
    weights, anchor_weights = noises.inverse(), priors.inverse()               # [readings] Information, [poses] Information
    return (core.gradient(dead, readings, weights, tails, heads, anchors, anchor_weights),
            core.gradient(poses, readings, weights, tails, heads, anchors, anchor_weights))


# --- plumbing -------------------------------------------------------------------------
def survey(core: ModuleType, steps, seed: int):
    """A lap along the true steps. Returns the true and dead-reckoned poses, and the readings with their
    covariances and the poses they link: every step, then the last pose read from the first; and every
    pose's anchor and prior, the first known, the others only vaguely, near where dead reckoning puts
    them."""
    rng = np.random.default_rng(seed)
    # Each step composes on the right, in the robot's own frame: a running product in that order is the
    # reverse of the running product of the reverses.
    truth = concatenate([core.mv.rotor()[None], steps.reverse().cumprod(axis=0).reverse()])   # [poses] Motor
    tails = np.append(np.arange(len(steps)), 0)                                # [readings]
    heads = np.append(np.arange(1, len(steps) + 1), len(steps))                # [readings]
    noises = concatenate([core.isotropic(*READING) * np.ones(len(steps)), core.isotropic(*CLOSING)[None]])   # [readings] Covariance
    spread = core.modes(noises)                                                # [readings, modes] Twist
    errors = (spread * rng.normal(size=spread.shape)).sum(axis=-1)             # [readings] Twist
    readings = (truth[tails].inverse() * truth[heads]) * (errors * 0.5).exp()  # [readings] Motor
    dead = concatenate([truth[:1], truth[0] * readings[:len(steps)].reverse().cumprod(axis=0).reverse()])   # [poses] Motor
    anchors = concatenate([truth[:1], dead[1:]])                               # [poses] Motor
    priors = concatenate([core.isotropic(*KNOWN)[None], core.isotropic(*VAGUE) * np.ones(len(steps))])   # [poses] Covariance
    return truth, dead, readings, noises, tails, heads, anchors, priors


def lap_in_plane():
    """A lap of N_PLANE_STEPS steps in the plane, turning faster and slower twice a lap, and slipping
    sideways a little, one way and back twice a lap, which leaves the lap closed."""
    mv = instantiate(CORE, PGA2D).mv
    phase = 2 * np.pi * np.arange(N_PLANE_STEPS) / N_PLANE_STEPS
    turn = mv.xy * (1 + TURN_VARIATION * np.sin(2 * phase)) * 2 * np.pi / N_PLANE_STEPS   # [steps] Twist
    slip = mv.yw * SLIP * np.cos(2 * phase)                                    # [steps] Twist
    return ((turn + slip + mv.xw * STEP_LENGTH) * 0.5).exp()                   # [steps] Motor


def lap_in_space():
    """A short lap in space, a turn of N_SPACE_STEPS steps, pitching and rolling once a lap, and slipping
    sideways and rising and falling twice a lap, which leaves the lap closed."""
    mv = instantiate(CORE, PGA3D).mv
    phase = 2 * np.pi * np.arange(N_SPACE_STEPS) / N_SPACE_STEPS
    turn = mv.xy * 2 * np.pi / N_SPACE_STEPS                                   # [] Twist
    sway = (mv.zx * np.cos(phase) + mv.yz * np.sin(phase)) * SWAY              # [steps] Twist
    slip = mv.yw * SLIP * np.cos(2 * phase) + mv.zw * RISE * np.sin(2 * phase) # [steps] Twist
    return ((turn + sway + slip + mv.xw * STEP_LENGTH) * 0.5).exp()            # [steps] Motor


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.estimation.odometry import render

    # Dead reckoning drifts half a metre over the lap.
    truth, dead, reckoned, poses, uncertainty = closing(lap_in_plane(), N_FULL_STEPS, SEED)
    save_figure(render.draw(truth, dead, reckoned, poses, uncertainty), "odometry")
    # Closing the lap a fifth of the way at a time.
    *_, iterates = damped(lap_in_plane(), DAMPING, N_DAMPED_STEPS, SEED)
    save_animation(render.animate(truth, dead, reckoned, iterates), "odometry", FRAME_DURATION)

    # checks
    # The gradient falls a millionfold from dead reckoning to the most likely poses, in the plane and, with
    # the same core, in space.
    *_, poses_in_space, _ = closing(lap_in_space(), N_FULL_STEPS, SEED)
    for steps, most_likely in ((lap_in_plane(), poses), (lap_in_space(), poses_in_space)):
        at_dead, at_optimum = gradients(steps, most_likely, SEED)
        np.testing.assert_allclose(at_optimum.kernel, 0.0, atol=1e-6 * np.abs(at_dead.kernel).max())


if __name__ == "__main__":
    main()
