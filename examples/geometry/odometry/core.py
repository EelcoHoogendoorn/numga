"""The most likely poses of a robot's lap, and how uncertain each one is.

A robot drives a lap and reads, at every stop, how far it moved since the last one; each reading is a
little off. Back at the start it recognises it, and reads one more relative pose, closing the loop.
Wanted are the poses most likely given all the readings, and how uncertain each one is.

A small error of a pose is a twist on its right, `pose * (twist * 0.5).exp()`, and its uncertainty a
covariance, `Covariance = Twist <- Line`, whose inverse is a quadric on twists. A reading measures the
twist at its head minus the twist at its tail carried there by the motor between them,
`relative << twist`. Mismatches at the readings and at the anchors pull on the poses: each weighted by
the inverse of its covariance, a reading's at its head, and carried to its tail with the opposite sign.
The pull of the mismatches is the gradient; the pull of what a correction measures is the information
applied to it, the curvature, never assembled. Conjugate gradients solve it for the correction,
Gauss-Newton moves the poses by it; each pose's uncertainty is the solve of the pull of every unit error
a reading or an anchor allows, the outer products of the solutions summed at each pose.

In the notation of nonlinear least squares the curvature reads as the normal matrix $J^\\top W J$,
applied without forming it.

The algebra is not fixed here. `ga` is supplied per instance, by
`examples.instantiate("examples.geometry.odometry.core", PGA2D)` for a lap in the plane or PGA3D
for one in space, and the same module serves both.
"""

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, concatenate

# Supplied by examples.instantiate.
ga: Algebra
mv = NumpyContext(ga).multivector
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
# A line reads out a twist: `line & twist`.
Line = ga.gatype.antibivector()
Covariance = ga.gatype((Twist, Line))             # Twist <- Line
# The inverse of a covariance: a quadric on twists, `twist & information(twist)`.
Information = ga.gatype((Line, Twist))            # Line <- Twist


# --- problem --------------------------------------------------------------------------
def mismatch(reading: Motor, relative: Motor) -> Twist:
    """How far a relative motor is from what was read: a twist on the right of the reading."""
    return (reading.inverse() * relative).log() * 2                            # [readings] or [poses] Twist


def reckon(poses: Motor, noises: Covariance, priors: Covariance) -> Covariance:
    """Each pose's uncertainty under dead reckoning, along the readings from each pose to the next: the
    first pose's prior and each reading's noise, entering at the pose it reaches, summed in the world
    frame from the first pose on, and read at every pose."""
    entering = concatenate([priors[:1], noises[:len(poses) - 1]])              # [poses] Covariance
    summed = (poses >> entering(poses << Line)).cumsum(axis=0)                 # [poses] Covariance
    return poses << summed(poses >> Line)                                      # [poses] Covariance


def pull(relative: Motor, at_readings: Twist, at_anchors: Twist, weights: Information, tails: np.ndarray,
         heads: np.ndarray, anchor_weights: Information) -> Line:
    """What mismatches at the readings and at the anchors pull on every pose: each weighted by the inverse
    of its covariance, a reading's at its head, and carried to its tail with the opposite sign; an
    anchor's at its pose."""
    weighted = weights(at_readings)                                            # [..., readings] Line
    anchored = anchor_weights(at_anchors)                                      # [..., poses] Line
    # A reading pulls its head, and its tail the opposite way, carried there by the motor between them.
    at_tails = -(relative >> weighted)                                         # [..., readings] Line
    pulled = anchored.at[..., heads].add(weighted)                             # [..., poses] Line
    return pulled.at[..., tails].add(at_tails)                                 # [..., poses] Line


def gradient(poses: Motor, readings: Motor, weights: Information, tails: np.ndarray, heads: np.ndarray,
             anchors: Motor, anchor_weights: Information) -> Line:
    """The gradient, per pose, of half the squared mismatches of every reading and of every pose from
    its anchor, each measured by its covariance: their pull."""
    relative = poses[tails].inverse() * poses[heads]                          # [readings] Motor
    return pull(relative, mismatch(readings, relative), mismatch(anchors, poses), weights, tails, heads, anchor_weights)


def curvature(poses: Motor, twists: Twist, weights: Information, tails: np.ndarray, heads: np.ndarray,
              anchor_weights: Information) -> Line:
    """The information applied to a correction of every pose: the pull of what the readings and the
    anchors measure of it."""
    relative = poses[tails].inverse() * poses[heads]                          # [readings] Motor: the head, from the tail
    measured = twists[..., heads] - (relative << twists[..., tails])           # [..., readings] Twist
    return pull(relative, measured, twists, weights, tails, heads, anchor_weights)   # [..., poses] Line


def curvature_alone(poses: Motor, weights: Information, tails: np.ndarray, heads: np.ndarray,
                    anchor_weights: Information) -> Information:
    """How sharply the objective curves as each pose moves alone, every other pose held."""
    relative = poses[tails].inverse() * poses[heads]                          # [readings] Motor: the head, from the tail
    # A reading measures its head's twist as it is, and its tail's carried to the head: its head, moved
    # alone, meets the reading's weight, and its tail the weight carried back to it.
    at_tails = relative >> weights(relative << Twist)                          # [readings] Line <- Twist
    # Each pose's anchor, then every reading at its head and at its tail.
    alone = anchor_weights.at[heads].add(weights)                              # [poses] Line <- Twist
    return alone.at[tails].add(at_tails)                                       # [poses] Line <- Twist


# --- solvers --------------------------------------------------------------------------
def conjugate_gradients(poses: Motor, right: Line, weights: Information, tails: np.ndarray, heads: np.ndarray,
                        anchor_weights: Information) -> Iterator[Twist]:
    """The corrections the information sends to the given lines, by conjugate gradients preconditioned by
    each pose's curvature alone, one solve for each leading index. Yields the corrections after each
    iteration."""
    alone = curvature_alone(poses, weights, tails, heads, anchor_weights)      # [poses] Line <- Twist
    correction = 0 * alone.solve(right)                                        # [..., poses] Twist
    residual = right                                                           # [..., poses] Line
    direction = alone.solve(residual)                                          # [..., poses] Twist
    aligned = (residual & direction).sum(axis=-1)                              # [...] Scalar
    # Exact after as many steps as there are unknowns.
    for _ in range(len(Twist.output_subspace) * len(poses)):
        pushed = curvature(poses, direction, weights, tails, heads, anchor_weights)   # [..., poses] Line
        step = (aligned / (pushed & direction).sum(axis=-1))[..., None]        # [..., 1] Scalar
        correction = correction + direction * step                             # [..., poses] Twist
        residual = residual - pushed * step                                    # [..., poses] Line
        preconditioned = alone.solve(residual)                                 # [..., poses] Twist
        realigned = (residual & preconditioned).sum(axis=-1)                   # [...] Scalar
        direction = preconditioned + direction * (realigned / aligned)[..., None]   # [..., poses] Twist
        aligned = realigned
        yield correction


def gauss_newton(poses: Motor, readings: Motor, weights: Information, tails: np.ndarray, heads: np.ndarray,
                 anchors: Motor, anchor_weights: Information, damping: float,
                 iterations: int) -> Iterator[Motor]:
    """Gauss-Newton, damped: each iteration solves the information at the current poses for the
    correction the gradient asks for, by conjugate gradients, and moves the poses by
    the damping's share of it. Yields the poses after each iteration."""
    for _ in range(iterations):
        right = -gradient(poses, readings, weights, tails, heads, anchors, anchor_weights)   # [poses] Line
        *_, correction = conjugate_gradients(poses, right, weights, tails, heads, anchor_weights)   # [poses] Twist
        poses = poses * (correction * (0.5 * damping)).exp()                   # [poses] Motor
        yield poses


# --- uncertainty ----------------------------------------------------------------------
def marginals(poses: Motor, noises: Covariance, tails: np.ndarray, heads: np.ndarray, priors: Covariance,
              weights: Information, anchor_weights: Information) -> Covariance:
    """Each pose's own uncertainty: the correction the pull of every unit error a reading or an anchor
    allows asks for, the outer products summed at each pose."""
    relative = poses[tails].inverse() * poses[heads]                          # [readings] Motor
    sites = concatenate([modes(noises), modes(priors)])                        # [readings + poses, modes] Twist
    units = sites[..., None] * np.eye(len(sites))[:, None]                     # [readings + poses, modes, readings + poses] Twist
    right = pull(relative, units[..., :len(tails)], units[..., len(tails):], weights, tails, heads, anchor_weights)   # [readings + poses, modes, poses] Line
    *_, errors = conjugate_gradients(poses, right, weights, tails, heads, anchor_weights)   # [readings + poses, modes, poses] Twist
    return (errors * (errors & Line)).sum(axis=(0, 1))                         # [poses] Covariance


def uncertainties(iterates: Iterator[Motor], uncertainty: Covariance, damping: float, noises: Covariance,
                  tails: np.ndarray, heads: np.ndarray, priors: Covariance, weights: Information,
                  anchor_weights: Information) -> Iterator[tuple[Motor, Covariance]]:
    """Each pose's uncertainty along damped Gauss-Newton, from the given one: the poses of each step keep
    the rest of their error, so the rest's square of their uncertainty beyond what every reading allows
    there. Yields the poses of each step with it."""
    for poses in iterates:
        posterior = marginals(poses, noises, tails, heads, priors, weights, anchor_weights)   # [poses] Covariance
        uncertainty = posterior + (uncertainty - posterior) * (1 - damping)**2    # [poses] Covariance
        yield poses, uncertainty


# --- plumbing -------------------------------------------------------------------------
def modes(uncertainty: Covariance) -> Twist:
    """Twists whose outer products sum to each uncertainty: along lines orthonormal in its form."""
    spread = Line & uncertainty                                                # [...] Scalar <- (Line, Line)
    _, lines = spread.eigh(spread)                                             # [..., modes] Line
    return uncertainty[..., None](lines)                                       # [..., modes] Twist


def isotropic(translation_std: float, rotation_std: float) -> Covariance:
    """The covariance of a reading with isotropic translation noise and rotation noise about the head."""
    twists = mv(Twist, np.eye(len(Twist.output_subspace)))                     # [twists] Twist
    # A basis twist turns if its reverse product is one, and slides if it is zero.
    turning = twists.scalar_norm_squared()                                     # [twists] Scalar
    variances = translation_std**2 + (rotation_std**2 - translation_std**2) * turning   # [twists] Scalar
    return (twists * (twists & Line) * variances).sum(axis=0)                  # [] Twist <- Line
