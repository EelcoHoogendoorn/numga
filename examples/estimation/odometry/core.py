"""Pose-graph odometry recovers the most likely trajectory of a robot around a lap and the uncertainty of every pose from noisy step readings and a loop-closing reading.

A robot drives a lap and reads, at every stop, how far it moved since the stop before. Each step reading is slightly off, so chaining them by dead reckoning drifts; when the robot returns to the start and recognises it, one more relative pose reading links the end of the lap back to the start and closes the loop.

Each pose is modelled as a motor, an error in it as a twist perturbation on its right, and the pull on a pose as an intrinsic forque paired with a twist through `twist & forque`. Where the term ***duality map*** here means a linear map between a space and its dual readout space, a reading's uncertainty is a covariance duality map `Twist <- Forque`, and its inverse `Forque <- Twist` is the information map that weighs a mismatch; a motor transports both between frames by its sandwich. Every reading pulls only on the two poses it links, so the Gauss-Newton information operator acts reading by reading without assembling a system matrix, and preconditioned conjugate gradients solve it both for each step's pose corrections and, in one batched solve over all noise modes, for every pose's marginal covariance.

The algebra is left unbound here and supplied per instance via `examples.instantiate("examples.estimation.odometry.core", PGA2D)` for a planar lap or `PGA3D` for a spatial lap, so one module serves both dimensions.
"""

from collections.abc import Iterator

import numpy as np

from numga import Algebra, concatenate
from numga.backend.context import Context
from numga.topology import Topology

# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector
Motor = ga.gatype.rotor()                           # Rotor+
Twist = ga.gatype.bivector()                        # Bivector+
# An intrinsic forque pairs with a twist via `twist & forque` to read the objective change the twist makes.
Forque = ga.gatype.antibivector()                   # Antibivector-
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Covariance = ga.gatype((Twist, Forque))             # Twist+ <- Forque-
Information = ga.gatype((Forque, Twist))            # Forque- <- Twist+
Quadric = ga.gatype((Plane, Point))                 # Plane+ <- Point-
n_twist_modes = len(Twist.output_subspace)
# Unit-weight point at the origin carried by each pose motor.
origin = mv.w.dual()                                # [] Point-


# --- math -----------------------------------------------------------------------------
def dead_reckoning_covariances(poses: Motor, noises: Covariance, priors: Covariance) -> Covariance:
    """Propagating `priors[:1]` forward through each step's noise in `noises` gives the dead-reckoned body-frame covariances (`[V] Twist+ <- Forque-`)."""
    V = len(poses)
    step_noises = concatenate([priors[:1], noises[:V - 1]])                     # [V] Twist+ <- Forque-
    world_uncertainty = (poses >> step_noises(poses << Forque)).cumsum(axis=0)  # [V] Twist+ <- Forque-
    return poses << world_uncertainty(poses >> Forque)                          # [V] Twist+ <- Forque-


def pose_forques(
    relative_motors: Motor,
    reading_twists: Twist,
    anchor_twists: Twist,
    weights: Information,
    topology: Topology,
    anchor_weights: Information,
) -> Forque:
    """Weighing `reading_twists` (`[..., E] Twist+`) and `anchor_twists` (`[..., V] Twist+`) across the pose graph `topology` sums the net forques (`[..., V] Forque-`) on every pose."""
    tails, heads = topology.elements[1].T                                       # [E], [E]
    head_forques = weights(reading_twists)                                      # [..., E] Forque-
    anchor_forques = anchor_weights(anchor_twists)                              # [..., V] Forque-
    # Each reading pulls on its head directly and on its tail with the opposite forque carried across the relative motor.
    tail_forques = -(relative_motors >> head_forques)                           # [..., E] Forque-
    total_forques = anchor_forques.at[..., heads].add(head_forques)             # [..., V] Forque-
    return total_forques.at[..., tails].add(tail_forques)                       # [..., V] Forque-


def objective_gradient(
    poses: Motor,
    readings: Motor,
    weights: Information,
    topology: Topology,
    anchors: Motor,
    anchor_weights: Information,
) -> Forque:
    """Logarithmic mismatches between `poses` (`[V] Motor+`), `readings` (`[E] Motor+`), and `anchors` (`[V] Motor+`) exert the pose-graph objective gradient (`[V] Forque-`)."""
    tails, heads = topology.elements[1].T                                       # [E], [E]
    relative_motors = poses[tails].inverse() * poses[heads]                     # [E] Rotor+
    reading_mismatches = (readings.inverse() * relative_motors).log() * 2       # [E] Twist+
    anchor_mismatches = (anchors.inverse() * poses).log() * 2                   # [V] Twist+
    return pose_forques(relative_motors, reading_mismatches, anchor_mismatches, weights, topology, anchor_weights)


def information_product(
    poses: Motor,
    twists: Twist,
    weights: Information,
    topology: Topology,
    anchor_weights: Information,
) -> Forque:
    """Applying the Gauss-Newton information operator at `poses` (`[V] Motor+`) maps pose perturbations `twists` (`[..., V] Twist+`) to the resulting forques (`[..., V] Forque-`)."""
    tails, heads = topology.elements[1].T                                       # [E], [E]
    # Each reading sees the twist perturbation at its head minus the perturbation at its tail carried to the head.
    relative_motors = poses[tails].inverse() * poses[heads]                     # [E] Rotor+
    reading_twists = twists[..., heads] - (relative_motors << twists[..., tails])  # [..., E] Twist+
    return pose_forques(relative_motors, reading_twists, twists, weights, topology, anchor_weights)


def diagonal_information(
    poses: Motor,
    weights: Information,
    topology: Topology,
    anchor_weights: Information,
) -> Information:
    """Summing each pose's own incident reading and anchor weights gives the per-pose block-diagonal preconditioner (`[V] Forque- <- Twist+`)."""
    tails, heads = topology.elements[1].T                                       # [E], [E]
    relative_motors = poses[tails].inverse() * poses[heads]                     # [E] Rotor+
    # Perturbing a reading's head alone meets its weight directly; perturbing its tail alone meets the weight carried back across the relative motor.
    tail_weights = relative_motors >> weights(relative_motors << Twist)         # [E] Forque- <- Twist+
    pose_weights = anchor_weights.at[heads].add(weights)                        # [V] Forque- <- Twist+
    return pose_weights.at[tails].add(tail_weights)                             # [V] Forque- <- Twist+


def conjugate_gradient_solves(
    poses: Motor,
    forques: Forque,
    weights: Information,
    topology: Topology,
    anchor_weights: Information,
) -> Iterator[Twist]:
    """Yields preconditioned conjugate-gradient twist corrections (`[..., V] Twist+`) converging on the Gauss-Newton normal-equation solution."""
    V, E = topology.n_elements
    preconditioner = diagonal_information(poses, weights, topology, anchor_weights)  # [V] Forque- <- Twist+
    correction = 0 * preconditioner.solve(forques)                              # [..., V] Twist+
    residual = forques                                                          # [..., V] Forque-
    direction = preconditioner.solve(residual)                                  # [..., V] Twist+
    residual_energy = (direction & residual).sum(axis=-1)                       # [...] Scalar+
    # Conjugate gradients reach the exact linear solution in at most the number of scalar unknowns.
    for _ in range(n_twist_modes * V):
        direction_forques = information_product(poses, direction, weights, topology, anchor_weights)  # [..., V] Forque-
        step = (residual_energy / (direction & direction_forques).sum(axis=-1))[..., None]
        correction = correction + direction * step                              # [..., V] Twist+
        residual = residual - direction_forques * step                          # [..., V] Forque-
        preconditioned_residual = preconditioner.solve(residual)                # [..., V] Twist+
        next_energy = (preconditioned_residual & residual).sum(axis=-1)         # [...] Scalar+
        direction = preconditioned_residual + direction * (next_energy / residual_energy)[..., None]  # [..., V] Twist+
        residual_energy = next_energy
        yield correction


def gauss_newton_iterates(
    poses: Motor,
    readings: Motor,
    weights: Information,
    topology: Topology,
    anchors: Motor,
    anchor_weights: Information,
    damping: float,
    n_iterations: int,
) -> Iterator[Motor]:
    """Yields the updated pose trajectory (`[V] Motor+`) across `n_iterations` damped Gauss-Newton steps."""
    for _ in range(n_iterations):
        descent = -objective_gradient(poses, readings, weights, topology, anchors, anchor_weights)  # [V] Forque-
        *_, correction = conjugate_gradient_solves(poses, descent, weights, topology, anchor_weights)  # [V] Twist+
        motor_step = (correction * (0.5 * damping)).exp()                       # [V] Rotor+
        poses = poses * motor_step                                              # [V] Rotor+
        yield poses


def marginal_covariances(
    poses: Motor,
    noises: Covariance,
    topology: Topology,
    priors: Covariance,
    weights: Information,
    anchor_weights: Information,
) -> Covariance:
    """Solving in one batch for the pose errors induced by every reading and anchor noise mode gives the marginal covariances (`[V] Twist+ <- Forque-`)."""
    V, E = topology.n_elements
    tails, heads = topology.elements[1].T                                       # [E], [E]
    relative_motors = poses[tails].inverse() * poses[heads]                     # [E] Rotor+
    noise_modes = concatenate([covariance_modes(noises), covariance_modes(priors)])  # [E + V, n_twist_modes] Twist+
    unit_modes = noise_modes[..., None] * np.eye(E + V)[:, None]                # [E + V, n_twist_modes, E + V] Twist+
    mode_forques = pose_forques(
        relative_motors, unit_modes[..., :E], unit_modes[..., E:], weights, topology, anchor_weights,
    )                                                                           # [E + V, n_twist_modes, V] Forque-
    *_, pose_errors = conjugate_gradient_solves(poses, mode_forques, weights, topology, anchor_weights)  # [E + V, n_twist_modes, V] Twist+
    return (pose_errors * (pose_errors & Forque)).sum(axis=(0, 1))              # [V] Twist+ <- Forque-


def relaxation_states(
    iterates: Iterator[Motor],
    uncertainty: Covariance,
    damping: float,
    noises: Covariance,
    topology: Topology,
    priors: Covariance,
    weights: Information,
    anchor_weights: Information,
) -> Iterator[tuple[Motor, Covariance]]:
    """Yields `(poses, uncertainty)` (`[V] Rotor+`, `[V] Twist+ <- Forque-`) as the pose covariances relax toward the posterior marginals across damped steps."""
    for poses in iterates:
        posterior = marginal_covariances(poses, noises, topology, priors, weights, anchor_weights)  # [V] Twist+ <- Forque-
        uncertainty = posterior + (uncertainty - posterior) * (1 - damping)**2  # [V] Twist+ <- Forque-
        yield poses, uncertainty


def position_ellipsoids(poses: Motor, uncertainty: Covariance, sigmas: float) -> Quadric:
    """Pushing `uncertainty` (`[...] Twist+ <- Forque-`) through the body-twist motion of `poses >> origin` gives the implicit confidence quadrics (`[...] Plane+ <- Point-`) at `sigmas` standard deviations."""
    positions = poses >> origin                                                 # [...] Point-
    # Body-frame twists carried to world space displace each pose's position by their commutator with it.
    shift = Twist.commutator(positions)(poses >> Twist)                         # [...] Point- <- Twist+
    # Pulling displacement-measuring planes back through that motion gives the corresponding twist readout forques.
    readout = shift.adjugate()                                                  # [...] Forque- <- Plane+
    # Adding the pushed-forward position covariance to the position's own dyad forms its second-moment dual quadric.
    moment = positions * (Plane & positions) + shift(uncertainty(readout))      # [...] Point- <- Plane+
    return moment.inverse() - (1 + sigmas**2) * mv.w * (mv.w & Point)           # [...] Plane+ <- Point-


# --- plumbing -------------------------------------------------------------------------
def covariance_modes(uncertainty: Covariance) -> Twist:
    """Generalized eigenvectors of the readout form `uncertainty & Forque` map through `uncertainty` (`[...] Twist+ <- Forque-`) to twist modes (`[..., n_twist_modes] Twist+`) whose outer products sum back to `uncertainty`."""
    spread = uncertainty & Forque                                               # [...] Scalar+ <- (Forque-, Forque-)
    _, forques = spread.eigh(spread)                                            # [..., n_twist_modes] Forque-
    return uncertainty[..., None](forques)                                      # [..., n_twist_modes] Twist+


def body_covariance(translation_std: float, rotation_std: float) -> Covariance:
    """Translation variance `translation_std**2` and rotation variance `rotation_std**2` about the body origin form an isotropic body-frame covariance (`[] Twist+ <- Forque-`)."""
    basis_twists = mv(Twist, np.eye(n_twist_modes))                             # [n_twist_modes] Twist+
    # A basis twist generates a rotation when its reverse norm squared is one and a translation when it is zero.
    rotational = basis_twists.scalar_norm_squared()                             # [n_twist_modes] Scalar+
    variances = translation_std**2 + (rotation_std**2 - translation_std**2) * rotational  # [n_twist_modes] Scalar+
    return (basis_twists * (basis_twists & Forque) * variances).sum(axis=0)     # [] Twist+ <- Forque-
