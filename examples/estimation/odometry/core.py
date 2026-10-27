"""Pose-graph odometry recovers the most likely trajectory of a robot around a lap and the uncertainty of every pose from noisy step readings and a loop-closing reading.

A robot drives a lap and reads, at every stop, how far it moved since the stop before. Each step reading is slightly off, so chaining them by dead reckoning drifts; when the robot returns to the start and recognises it, one more relative pose reading links the end of the lap back to the start and closes the loop.

A pose is a motor, an error in it a twist on its right, and what pulls on a pose a forque, which pairs with a twist, `forque & twist`, into the change the twist makes. A reading's uncertainty is a covariance, a map from forques to twists, `Twist <- Forque`, and its inverse is the information that weighs a mismatch; a motor carries both between frames by its sandwich. Every reading pulls only on the two poses it links, so the Gauss-Newton operator is applied reading by reading, with no system matrix assembled, and preconditioned conjugate gradients solve it both for each step's corrections and, in one batched solve over all noise modes, for every pose's marginal covariance.

In the matrix notation of pose-graph nonlinear least squares on $\\mathrm{SE}(d)$, each reading $z_k$ between poses $x_i$ and $x_j$ defines a Lie-algebra residual $r_k = \\log(z_k^{-1} x_i^{-1} x_j)^\\vee$, the objective is $\\tfrac{1}{2} \\sum_k r_k^{\\mathsf T} \\Sigma_k^{-1} r_k$, and each Gauss-Newton step solves the normal equations $(J^{\\mathsf T} W J)\\, \\delta = -J^{\\mathsf T} W r$.

The algebra is not fixed here. `ga` is supplied per instance, by `examples.instantiate("examples.estimation.odometry.core", PGA2D)` for a lap in the plane or `PGA3D` for one in space, and the same module serves both.
"""

from collections.abc import Iterator

import numpy as np

from numga import Algebra, concatenate
from numga.backend.context import Context

# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector
Motor = ga.gatype.rotor()                           # Rotor+
Twist = ga.gatype.bivector()                        # Bivector+
# What pulls on a pose: a forque, which paired with a twist, `forque & twist`, is the change the twist makes.
Forque = ga.gatype.antibivector()                   # Antibivector-
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Covariance = ga.gatype((Twist, Forque))
Information = ga.gatype((Forque, Twist))
Quadric = ga.gatype((Plane, Point))
# The point each pose carries: the origin, dual to the weight direction w.
origin = mv.w.dual()                                # [] Point-


# --- math -----------------------------------------------------------------------------
def reckon(poses: Motor, noises: Covariance, priors: Covariance) -> Covariance:
    """Under dead reckoning each pose's covariance is the first pose's prior grown by the noise of every step before it."""
    step_noises = concatenate([priors[:1], noises[:len(poses) - 1]])           # [poses] Twist+ <- Forque-
    world_uncertainty = (poses >> step_noises(poses << Forque)).cumsum(axis=0)  # [poses] Twist+ <- Forque-
    return poses << world_uncertainty(poses >> Forque)                         # [poses] Twist+ <- Forque-


def pull(relative_motors: Motor, reading_twists: Twist, anchor_twists: Twist, weights: Information,
         tails: np.ndarray, heads: np.ndarray, anchor_weights: Information) -> Forque:
    """The forque on every pose is the sum of those its readings' and its anchor's twists exert on it."""
    head_forques = weights(reading_twists)                                     # [..., readings] Forque-
    anchor_forques = anchor_weights(anchor_twists)                             # [..., poses] Forque-
    # A reading pulls on its head, and on its tail the opposite way carried across the relative motor.
    tail_forques = -(relative_motors >> head_forques)                          # [..., readings] Forque-
    pose_forques = anchor_forques.at[..., heads].add(head_forques)             # [..., poses] Forque-
    return pose_forques.at[..., tails].add(tail_forques)                       # [..., poses] Forque-


def gradient(poses: Motor, readings: Motor, weights: Information, tails: np.ndarray, heads: np.ndarray,
             anchors: Motor, anchor_weights: Information) -> Forque:
    """The gradient of the pose-graph objective is the forque its mismatches exert on every pose."""
    relative_motors = poses[tails].inverse() * poses[heads]                    # [readings] Motor+
    reading_mismatches = (readings.inverse() * relative_motors).log() * 2      # [readings] Twist+
    anchor_mismatches = (anchors.inverse() * poses).log() * 2                  # [poses] Twist+
    return pull(relative_motors, reading_mismatches, anchor_mismatches, weights, tails, heads, anchor_weights)


def curvature(poses: Motor, twists: Twist, weights: Information, tails: np.ndarray, heads: np.ndarray,
              anchor_weights: Information) -> Forque:
    """The Gauss-Newton information operator takes a twist perturbation of every pose to the forques it causes."""
    # Each reading measures the twist at its head minus the twist at its tail carried to the head.
    relative_motors = poses[tails].inverse() * poses[heads]                    # [readings] Motor+
    reading_twists = twists[..., heads] - (relative_motors << twists[..., tails])  # [..., readings] Twist+
    return pull(relative_motors, reading_twists, twists, weights, tails, heads, anchor_weights)


def curvature_alone(poses: Motor, weights: Information, tails: np.ndarray, heads: np.ndarray,
                    anchor_weights: Information) -> Information:
    """The per-pose diagonal block of the Gauss-Newton information operator is what a pose's perturbation alone causes."""
    relative_motors = poses[tails].inverse() * poses[heads]                    # [readings] Motor+
    # Moving a reading's head alone meets the reading's weight directly; moving its tail alone
    # meets the weight carried back across the relative motor.
    tail_weights = relative_motors >> weights(relative_motors << Twist)        # [readings] Forque- <- Twist+
    pose_weights = anchor_weights.at[heads].add(weights)                       # [poses] Forque- <- Twist+
    return pose_weights.at[tails].add(tail_weights)                            # [poses] Forque- <- Twist+


def conjugate_gradients(poses: Motor, forques: Forque, weights: Information, tails: np.ndarray, heads: np.ndarray,
                        anchor_weights: Information) -> Iterator[Twist]:
    """Preconditioned conjugate gradients yield successive twist corrections converging on the solution of the Gauss-Newton normal equations."""
    preconditioner = curvature_alone(poses, weights, tails, heads, anchor_weights)  # [poses] Forque- <- Twist+
    correction = 0 * preconditioner.solve(forques)                             # [..., poses] Twist+
    residual = forques                                                         # [..., poses] Forque-
    direction = preconditioner.solve(residual)                                 # [..., poses] Twist+
    residual_energy = (residual & direction).sum(axis=-1)                      # [...] Scalar+
    # Conjugate gradients converge in at most as many steps as there are scalar unknowns.
    for _ in range(len(Twist.output_subspace) * len(poses)):
        direction_forques = curvature(poses, direction, weights, tails, heads, anchor_weights)  # [..., poses] Forque-
        step = (residual_energy / (direction_forques & direction).sum(axis=-1))[..., None]
        correction = correction + direction * step                             # [..., poses] Twist+
        residual = residual - direction_forques * step                         # [..., poses] Forque-
        preconditioned_residual = preconditioner.solve(residual)               # [..., poses] Twist+
        next_energy = (residual & preconditioned_residual).sum(axis=-1)        # [...] Scalar+
        direction = preconditioned_residual + direction * (next_energy / residual_energy)[..., None]  # [..., poses] Twist+
        residual_energy = next_energy
        yield correction


def gauss_newton(poses: Motor, readings: Motor, weights: Information, tails: np.ndarray, heads: np.ndarray,
                 anchors: Motor, anchor_weights: Information, damping: float,
                 n_iterations: int) -> Iterator[Motor]:
    """Damped Gauss-Newton iterations yield the poses after each step."""
    for _ in range(n_iterations):
        descent = -gradient(poses, readings, weights, tails, heads, anchors, anchor_weights)  # [poses] Forque-
        *_, correction = conjugate_gradients(poses, descent, weights, tails, heads, anchor_weights)  # [poses] Twist+
        poses = poses * (correction * (0.5 * damping)).exp()                   # [poses] Motor+
        yield poses


def marginals(poses: Motor, noises: Covariance, tails: np.ndarray, heads: np.ndarray, priors: Covariance,
              weights: Information, anchor_weights: Information) -> Covariance:
    """Each pose's marginal covariance sums the pose errors that every unit mode of reading and anchor noise induces."""
    relative_motors = poses[tails].inverse() * poses[heads]                    # [readings] Motor+
    noise_modes = concatenate([modes(noises), modes(priors)])                  # [sites, modes] Twist+
    unit_modes = noise_modes[..., None] * np.eye(len(noise_modes))[:, None]    # [sites, modes, sites] Twist+
    mode_forques = pull(relative_motors, unit_modes[..., :len(tails)], unit_modes[..., len(tails):], weights, tails, heads, anchor_weights)  # [sites, modes, poses] Forque-
    *_, pose_errors = conjugate_gradients(poses, mode_forques, weights, tails, heads, anchor_weights)  # [sites, modes, poses] Twist+
    return (pose_errors * (pose_errors & Forque)).sum(axis=(0, 1))             # [poses] Twist+ <- Forque-


def uncertainties(iterates: Iterator[Motor], uncertainty: Covariance, damping: float, noises: Covariance,
                  tails: np.ndarray, heads: np.ndarray, priors: Covariance, weights: Information,
                  anchor_weights: Information) -> Iterator[tuple[Motor, Covariance]]:
    """Across damped Gauss-Newton steps the poses' covariance relaxes towards their marginals; the steps yield the poses with it."""
    for poses in iterates:
        posterior = marginals(poses, noises, tails, heads, priors, weights, anchor_weights)  # [poses] Twist+ <- Forque-
        uncertainty = posterior + (uncertainty - posterior) * (1 - damping)**2  # [poses] Twist+ <- Forque-
        yield poses, uncertainty


def ellipses(poses: Motor, uncertainty: Covariance, sigmas: float) -> Quadric:
    """The quadric of each pose's position vanishes on its confidence ellipse, `sigmas` standard deviations out."""
    positions = poses >> origin                                                # [...] Point-
    # A twist moves the position by its commutator with it: a map from twists to displacements.
    shift = Twist.commutator(positions)(poses >> Twist)                        # [...] Point- <- Twist+
    # Its adjugate pulls each plane back to the forque that reads the same displacement from the twist.
    readout = shift.adjugate()                                                 # [...] Forque- <- Plane+
    # The covariance pushed forward through the motion adds to the position's own dyad: its second moment.
    moment = positions * (Plane & positions) + shift(uncertainty(readout))     # [...] Point- <- Plane+
    return moment.inverse() - (1 + sigmas**2) * mv.w * (mv.w & Point)          # [...] Plane+ <- Point-


# --- plumbing -------------------------------------------------------------------------
def modes(uncertainty: Covariance) -> Twist:
    """The modes of a covariance are twists whose outer products sum back to it."""
    spread = Forque & uncertainty                                              # [...] Scalar+ <- (Forque-, Forque-)
    _, forques = spread.eigh(spread)                                           # [..., modes] Forque-
    return uncertainty[..., None](forques)                                     # [..., modes] Twist+


def isotropic(translation_std: float, rotation_std: float) -> Covariance:
    """A body-frame covariance is isotropic in translation and in rotation about the origin."""
    basis_twists = mv(Twist, np.eye(len(Twist.output_subspace)))               # [modes] Twist+
    # A basis twist generates a rotation if its reverse product is one, and a translation if it is zero.
    rotational = basis_twists.scalar_norm_squared()                            # [modes] Scalar+
    variances = translation_std**2 + (rotation_std**2 - translation_std**2) * rotational  # [modes] Scalar+
    return (basis_twists * (basis_twists & Forque) * variances).sum(axis=0)    # [] Twist+ <- Forque-
