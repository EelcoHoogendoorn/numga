"""Scenes for the lap: the most likely poses given every reading, the steps and then the one that
closes the loop, by Gauss-Newton from dead reckoning; in the plane, and in space.

The core is instantiated for the algebra of the true steps, so the same scenes run on a lap in PGA2D
and on one in PGA3D.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from numga import concatenate
from numga.algebras import PGA2D, PGA3D
from examples import instantiate

CORE = "examples.geometry.odometry.core"
# The steps of a lap in the plane, and of a short one in space.
PLANE_STEPS = 36
SPACE_STEPS = 6
# How far a step reading and the closing reading, recognising the start, are trusted, how well the first
# pose is known, and how vaguely the others are known before any reading, as translation and rotation
# standard deviations.
READING = (0.03, 0.01)
CLOSING = (0.001, 0.0003)
KNOWN = (0.01, 0.01)
VAGUE = (100.0, 30.0)


# --- math -----------------------------------------------------------------------------
def damped(steps, damping: float, iterations: int, seed: int):
    """Gauss-Newton with every reading from the dead-reckoned poses, damped by the given share, and each
    pose's uncertainty along it from dead reckoning's. Returns the true and dead-reckoned poses, dead
    reckoning's uncertainty, and the iterations, yielding the poses and their uncertainties after each."""
    core = instantiate(CORE, steps.algebra)
    truth, dead, readings, noises, tails, heads, anchors, priors = survey(core, steps, seed)
    weights, anchor_weights = noises.inverse(), priors.inverse()               # [readings] Information, [poses] Information
    reckoned = core.reckon(dead, noises, priors)                               # [poses] Covariance
    poses = core.gauss_newton(dead, readings, weights, tails, heads, anchors, anchor_weights, damping, iterations)
    iterates = core.uncertainties(poses, reckoned, damping, noises, tails, heads, priors, weights, anchor_weights)
    return truth, dead, reckoned, iterates


def closing(steps, iterations: int, seed: int):
    """The most likely poses given every reading, by the given full Gauss-Newton iterations. Returns the
    true and dead-reckoned poses, dead reckoning's uncertainty, and the most likely poses with theirs."""
    truth, dead, reckoned, iterates = damped(steps, 1.0, iterations, seed)
    *_, (poses, uncertainty) = iterates                                        # [poses] Motor, [poses] Covariance

    # --- checks
    # The gradient falls a millionfold from dead reckoning to the most likely poses.
    core = instantiate(CORE, steps.algebra)
    _, _, readings, noises, tails, heads, anchors, priors = survey(core, steps, seed)
    weights, anchor_weights = noises.inverse(), priors.inverse()               # [readings] Information, [poses] Information
    at_dead = core.gradient(dead, readings, weights, tails, heads, anchors, anchor_weights)
    at_optimum = core.gradient(poses, readings, weights, tails, heads, anchors, anchor_weights)
    np.testing.assert_allclose(at_optimum.kernel, 0.0, atol=1e-6 * np.abs(at_dead.kernel).max())
    return truth, dead, reckoned, poses, uncertainty


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
    """A lap of PLANE_STEPS steps of 0.7 m in the plane, turning faster and slower twice a lap,
    and slipping sideways a little, one way and back twice a lap, which leaves the lap closed."""
    mv = instantiate(CORE, PGA2D).mv
    phase = 2 * np.pi * np.arange(PLANE_STEPS) / PLANE_STEPS
    turn = mv.xy * (1 + 0.4 * np.sin(2 * phase)) * 2 * np.pi / PLANE_STEPS       # [steps] Twist
    slip = mv.yw * 0.1 * np.cos(2 * phase)                                     # [steps] Twist
    return ((turn + slip + mv.xw * 0.7) * 0.5).exp()                           # [steps] Motor


def lap_in_space():
    """A short lap in space, a turn of SPACE_STEPS steps of 0.7 m, pitching and rolling once a lap, and
    slipping sideways and rising and falling twice a lap, which leaves the lap closed."""
    mv = instantiate(CORE, PGA3D).mv
    phase = 2 * np.pi * np.arange(SPACE_STEPS) / SPACE_STEPS
    turn = mv.xy * 2 * np.pi / SPACE_STEPS                                      # [] Twist
    sway = (mv.zx * np.cos(phase) + mv.yz * np.sin(phase)) * 0.1               # [steps] Twist
    slip = mv.yw * 0.1 * np.cos(2 * phase) + mv.zw * 0.05 * np.sin(2 * phase)  # [steps] Twist
    return ((turn + sway + slip + mv.xw * 0.7) * 0.5).exp()                    # [steps] Motor


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.geometry.odometry import render

    # Dead reckoning drifts half a metre over the lap.
    truth, dead, reckoned, poses, uncertainty = closing(lap_in_plane(), 5, 9)
    save_figure(render.draw(truth, dead, reckoned, poses, uncertainty), "odometry")
    # Closing the lap a fifth of the way at a time.
    *_, iterates = damped(lap_in_plane(), 0.2, 20, 9)
    save_animation(render.animate(truth, dead, reckoned, iterates), "odometry", 80)
