"""Reduced-coordinate dynamics of a branching robot in projective 3D space represents each joint axis and link rate as an extrinsic line and each link momentum and forque as an intrinsic line joined from points.

One motor sandwich carries link axes, rates, momenta, forques, and spatial inertias `motor >> inertia(motor << Rate)` between frames. The commutator evaluates both rigid-body cross products, turning a joint axis carried by its link as `rate.commutator(axis)` and transporting momentum under motion as `rate.commutator(momentum)` while keeping motions and forces distinct by orientation.

Summing joint speeds along each tree path from the root body forms a sparse map from joint speeds to link rates, and its adjugate sums subtree forques back onto each joint axis via `axis & forque` while summing all link forques onto the root body. Recursive Newton-Euler inverse dynamics runs those maps forward for rates and accelerations and back for forques, while forward dynamics pulls the link inertias back into the mass matrix and eliminates the root block. A linearly implicit Euler step pulls the damping and stiffness of hinges, joint stops, and ground contacts back through the same map, where each ground contact adds extra link inertia joined at the ball's lowest point.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import jax
import numpy as np

from numga import Extensor, where
from numga.algebras import PGA3D as ga
from numga.backend.jax import JaxContext
from numga.sparse import Sparse, SparseExtensor, spdiag
from numga.topology import Topology
from examples.mechanics.spinning_top import core as contact

jax.config.update("jax_enable_x64", True)
mv = JaxContext(ga, dtype=np.float64).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-
Plane = ga.gatype.vector()                                                   # Vector+
# Joint axis per unit speed and link spatial velocity: an extrinsic line of turn.
Rate = ga.gatype.bivector()                                                  # Bivector+
# Link spatial momentum and applied forque: an intrinsic line joined from points.
Forque = ga.gatype.antibivector()                                            # Antibivector-
Motor = ga.gatype.rotor()                                                    # Rotor+
# Generalized joint momentum and force paired against scalar joint speed.
Force = ga.gatype.antiscalar()                                               # Antiscalar-
Inertia = ga.gatype((Forque, Rate))                                          # Forque- <- Rate+
Hinge = ga.gatype((Force, Scalar))                                           # Force- <- Scalar+
Quadric = ga.gatype((Plane, Point))                                          # Plane+ <- Point-

Tree = tuple[Sparse, Sparse, Sparse, Sparse, Sparse, int]
State = tuple[Motor, Rate, Scalar, Scalar]


# --- math -----------------------------------------------------------------------------
def kinematic_tree(topology: Topology, ball_carriers: np.ndarray, n_generations: int) -> Tree:
    """Root reading (`Scalar+[1 <- V]`), edge tail and head readings (`Scalar+[E <- V]`), root-to-link path matrix (`Scalar+[V <- E]`), ball-carrier reading (`Scalar+[n_balls <- V]`), and tree depth `n_generations` assemble from the directed joint `topology`."""
    V, E = topology.n_elements
    # Read each directed joint's parent tail and child head separately so every sparse map stores only active couplings.
    tails, heads = (topology.reading(ends, 0) for ends in topology.elements[1].T)
    parents = heads.adjugate() * tails                                       # Scalar+[V <- V]
    paths = heads.adjugate()                                                 # Scalar+[V <- E]
    for _ in range(n_generations - 1):
        paths = heads.adjugate() + parents * paths
    root = np.setdiff1d(np.arange(V), topology.elements[1][:, 1])
    return topology.reading(root, 0), tails, heads, paths, topology.reading(ball_carriers, 0), n_generations


@partial(jax.jit, static_argnames=("tree",))
def forward_kinematics(tree: Tree, home_axes: Rate, motor: Motor, angles: Scalar) -> Motor:
    """Link placement motors (`[...] Motor+[V]`) compose the root `motor` with each joint's home-pose turn across the `n_generations` levels of `tree`."""
    root, tails, heads, _, _, n_generations = tree
    turns = (home_axes * (angles / 2)).exp()                                 # [...] Motor+[E]
    root_placement = root.adjugate() * motor[..., None].field()              # [...] Motor+[V]
    motors = root_placement                                                  # [...] Motor+[V]
    for _ in range(n_generations):
        motors = root_placement + heads.adjugate() * ((tails * motors) * turns)
    return motors


def inverse_dynamics(
    tree: Tree,
    home_axes: Rate,
    inertias: Inertia,
    motors: Motor,
    rate: Rate,
    speeds: Scalar,
    acceleration: Rate,
    accelerations: Scalar,
    external_forques: Forque,
) -> tuple[Forque, Force]:
    """Root forque (`[...] Forque-`) and joint generalized forces (`[...] Force-[E]`) balance link inertia and momentum transport at `(rate, speeds)` and `(acceleration, accelerations)` against `external_forques`."""
    _, tails, heads, paths, _, _ = tree
    axes = (tails * motors) >> home_axes                                     # [...] Rate+[E]
    joint_rates = paths * spdiag(axes * Scalar)                              # [...] (Rate+ <- Scalar+)[V <- E]
    world_inertias = motors >> inertias(motors << Rate)                      # [...] (Forque- <- Rate+)[V]
    # Propagate link rates outward; each turning child link adds `rate.commutator(axis) * speed` down its subtree.
    rates = joint_rates(speeds) + rate                                       # [...] Rate+[V]
    velocity_products = (heads * rates).commutator(axes) * speeds            # [...] Rate+[E]
    link_accelerations = joint_rates(accelerations) + paths * velocity_products + acceleration   # [...] Rate+[V]
    momenta = world_inertias(rates)                                          # [...] Forque-[V]
    forques = world_inertias(link_accelerations) + rates.commutator(momenta) - external_forques  # [...] Forque-[V]
    # Sum all link forques onto the root and pull subtree forques back onto each joint axis.
    return forques.sites.sum(), joint_rates.adjugate()(forques)


def rate_pullback(joint_rates: SparseExtensor, link_maps: Inertia) -> tuple[Inertia, Extensor, SparseExtensor]:
    """Link rate-to-forque maps (`[...] (Forque- <- Rate+)[V]`) pull back into the root block (`[...] Forque- <- Rate+`), root-joint coupling (`[...] (Force- <- Rate+)[E]`), and sparse joint block (`[...] (Force- <- Scalar+)[E <- E]`)."""
    return (
        link_maps.sites.sum(),
        joint_rates.adjugate()(link_maps),
        joint_rates.adjugate()(link_maps(joint_rates)),
    )


def coupled_rates(
    root_block: Inertia,
    coupling: Extensor,
    joint_block: SparseExtensor,
    forque: Forque,
    forces: Force,
) -> tuple[Rate, Scalar]:
    """Root rate (`[...] Rate+`) and joint speeds (`[...] Scalar+[E]`) solve the coupled block system against `(forque, forces)` by Schur elimination of `root_block`."""
    # Eliminate the six-degree-of-freedom root block and solve each robot's small joint system densely on JAX.
    reduced = joint_block.to_dense() - coupling(root_block.solve(coupling.adjugate()))   # [...] (Force- <- Scalar+)[E <- E]
    speeds = reduced.solve(forces - coupling(root_block.solve(forque)))                  # [...] Scalar+[E]
    return root_block.solve(forque - coupling.adjugate()(speeds)), speeds


def mass_matrix(tree: Tree, home_axes: Rate, inertias: Inertia, motors: Motor) -> tuple[Inertia, Extensor, SparseExtensor]:
    """World-frame link inertias pull back at `motors` into the root block, root-joint coupling, and sparse joint block of the mass matrix."""
    _, tails, _, paths, _, _ = tree
    axes = (tails * motors) >> home_axes                                     # [...] Rate+[E]
    joint_rates = paths * spdiag(axes * Scalar)                              # [...] (Rate+ <- Scalar+)[V <- E]
    return rate_pullback(joint_rates, motors >> inertias(motors << Rate))


def forward_dynamics(
    tree: Tree,
    home_axes: Rate,
    inertias: Inertia,
    motors: Motor,
    rate: Rate,
    speeds: Scalar,
    forces: Force,
    external_forques: Forque,
) -> tuple[Rate, Scalar]:
    """Root angular-linear acceleration (`[...] Rate+`) and joint accelerations (`[...] Scalar+[E]`) solve the mass matrix against `(forces, external_forques)` minus zero-acceleration bias loads."""
    root_bias, joint_bias = inverse_dynamics(
        tree, home_axes, inertias, motors, rate, speeds, rate * 0, speeds * 0, external_forques
    )
    return coupled_rates(*mass_matrix(tree, home_axes, inertias, motors), -root_bias, forces - joint_bias)


def ground_contact(
    balls: Sparse,
    ground: Plane,
    surfaces: Quadric,
    contact_stiffness: Scalar,
    contact_damping: float,
    friction: float,
    motors: Motor,
    rates: Rate,
) -> tuple[Forque, Inertia, Inertia]:
    """Ground contact at each ball's lowest point produces link contact forques (`[...] Forque-[V]`), contact damping maps (`[...] (Forque- <- Rate+)[V]`), and contact stiffness maps (`[...] (Forque- <- Rate+)[V]`)."""
    up_direction = ground.dual()                                             # [] Point-
    ball_motors = (balls * motors).batch()                                   # [...] Motor+
    world_surfaces = ball_motors >> surfaces(ball_motors << Point)           # [...] Plane+ <- Point-
    bottoms = contact.lowest_point(world_surfaces, up_direction).field()     # [...] Point-[n_balls]
    velocities = (balls * rates).commutator(bottoms)                         # [...] Point-[n_balls]
    normal_part = up_direction * (ground & Point)                            # [] Point- <- Point-
    depths = (-(ground & bottoms)).clip(0, np.inf)                           # [...] Scalar+[n_balls]
    sinking_speeds = -(ground & velocities)                                  # [...] Scalar+[n_balls]
    # Clamp normal forces nonnegative so rising contacts never pull a ball toward the ground.
    pushes = (depths * contact_stiffness * (1 + contact_damping * sinking_speeds)).clip(0, np.inf)   # [...] Scalar+[n_balls]
    forces = up_direction * pushes - (velocities - normal_part(velocities)) * (pushes * friction)    # [...] Point-[n_balls]
    dampers = normal_part * (depths * contact_stiffness * contact_damping) + (Point - normal_part) * (pushes * friction)
    springs = normal_part * where(depths > 0, contact_stiffness, 0)          # [...] (Point- <- Point-)[n_balls]
    damping_maps = bottoms.regressive(dampers(bottoms.commutator(Rate)))     # [...] (Forque- <- Rate+)[n_balls]
    stiffness_maps = bottoms.regressive(springs(bottoms.commutator(Rate)))   # [...] (Forque- <- Rate+)[n_balls]
    return balls.adjugate() * (forces & bottoms), balls.adjugate() * damping_maps, balls.adjugate() * stiffness_maps


def free_step(
    tree: Tree,
    home_axes: Rate,
    inertias: Inertia,
    mass_points: Point,
    gravity: Point,
    hinge_stiffness: Hinge,
    hinge_damping: Hinge,
    state: State,
    dt: float,
) -> State:
    """Linearly implicit Euler advances `state` by `dt` in free flight under gravity and passive hinge springs and dampers."""
    _, tails, _, paths, _, _ = tree
    motor, rate, angles, speeds = state
    motors = forward_kinematics(tree, home_axes, motor, angles)              # [...] Motor+[V]
    axes = (tails * motors) >> home_axes                                     # [...] Rate+[E]
    joint_rates = paths * spdiag(axes * Scalar)                              # [...] (Rate+ <- Scalar+)[V <- E]
    weights = gravity & (motors >> mass_points)                              # [...] Forque-[V]
    drives = -(hinge_stiffness(angles + speeds * dt) + hinge_damping(speeds))   # [...] Force-[E]
    root_bias, joint_bias = inverse_dynamics(
        tree, home_axes, inertias, motors, rate, speeds, rate * 0, speeds * 0, weights
    )
    root_block, coupling, joint_block = rate_pullback(joint_rates, motors >> inertias(motors << Rate))
    hinge_maps = hinge_damping * dt + hinge_stiffness * dt**2                # [...] (Force- <- Scalar+)[E]
    rate_change, speed_changes = coupled_rates(
        root_block, coupling, joint_block + hinge_maps, -root_bias * dt, (drives - joint_bias) * dt
    )
    next_rate, next_speeds = rate + rate_change, speeds + speed_changes
    next_turn = (next_rate * (dt / 2)).exp()                                 # [...] Motor+
    return next_turn * motor, next_rate, angles + next_speeds * dt, next_speeds


def drop_step(
    tree: Tree,
    home_axes: Rate,
    inertias: Inertia,
    mass_points: Point,
    gravity: Point,
    hinge_stiffness: Hinge,
    hinge_damping: Hinge,
    low_stops: Scalar,
    high_stops: Scalar,
    stop_stiffness: Hinge,
    ground: Plane,
    surfaces: Quadric,
    contact_stiffness: Scalar,
    contact_damping: float,
    friction: float,
    state: State,
    dt: float,
) -> State:
    """Linearly implicit Euler advances `state` by `dt` under gravity, hinge springs and dampers, one-sided joint stops, and ground contact."""
    _, tails, _, paths, balls, _ = tree
    motor, rate, angles, speeds = state
    motors = forward_kinematics(tree, home_axes, motor, angles)              # [...] Motor+[V]
    axes = (tails * motors) >> home_axes                                     # [...] Rate+[E]
    joint_rates = paths * spdiag(axes * Scalar)                              # [...] (Rate+ <- Scalar+)[V <- E]
    rates = joint_rates(speeds) + rate                                       # [...] Rate+[V]
    pushes, contact_damping_maps, contact_stiffness_maps = ground_contact(
        balls, ground, surfaces, contact_stiffness, contact_damping, friction, motors, rates
    )
    weights = gravity & (motors >> mass_points)                              # [...] Forque-[V]
    # Shift ground spring forces to end-of-step positions predicted from current link rates.
    external_forques = weights + pushes - contact_stiffness_maps(rates) * dt   # [...] Forque-[V]
    overruns = (angles - high_stops).clip(0, np.inf) + (angles - low_stops).clip(-np.inf, 0)   # [...] Scalar+[E]
    active_stops = where(overruns.abs() > 0, stop_stiffness, 0)              # [...] (Force- <- Scalar+)[E]
    angle_steps = speeds * dt                                                # [...] Scalar+[E]
    drives = -(hinge_stiffness(angles + angle_steps) + active_stops(overruns + angle_steps) + hinge_damping(speeds))
    root_bias, joint_bias = inverse_dynamics(
        tree, home_axes, inertias, motors, rate, speeds, rate * 0, speeds * 0, external_forques
    )
    link_maps = (motors >> inertias(motors << Rate)) + contact_damping_maps * dt + contact_stiffness_maps * dt**2
    root_block, coupling, joint_block = rate_pullback(joint_rates, link_maps)
    hinge_maps = hinge_damping * dt + (hinge_stiffness + active_stops) * dt**2   # [...] (Force- <- Scalar+)[E]
    rate_change, speed_changes = coupled_rates(
        root_block, coupling, joint_block + hinge_maps, -root_bias * dt, (drives - joint_bias) * dt
    )
    next_rate, next_speeds = rate + rate_change, speeds + speed_changes
    next_turn = (next_rate * (dt / 2)).exp()                                 # [...] Motor+
    return next_turn * motor, next_rate, angles + next_speeds * dt, next_speeds


@partial(jax.jit, static_argnames=("step_fn", "n_steps"))
def advance(step_fn, state: State, dt: float, n_steps: int) -> State:
    """Repeated `step_fn` application advances `state` across `n_steps` substeps of `dt` inside a single compiled loop."""
    joint_type = state[2].gatype

    def retyped_state(current: State) -> State:
        """Full carry types stay invariant across `fori_loop` iterations."""
        motor, rate, angles, speeds = current
        return motor.retyped(Motor), rate.retyped(Rate), angles.retyped(joint_type), speeds.retyped(joint_type)

    return jax.lax.fori_loop(0, n_steps, lambda _, current: retyped_state(step_fn(current, dt)), retyped_state(state))


def trajectory(step_fn, state: State, dt: float, n_steps: int, n_frames: int) -> Iterator[State]:
    """Yields `n_frames` robot states (`State`) spaced by `n_steps` substeps of `dt` starting from `state`."""
    for _ in range(n_frames):
        yield state
        state = advance(step_fn, state, dt, n_steps)


@partial(jax.jit, static_argnames=("tree",))
def mechanical_energy(
    tree: Tree,
    home_axes: Rate,
    inertias: Inertia,
    mass_points: Point,
    gravity: Point,
    hinge_stiffness: Hinge,
    low_stops: Scalar,
    high_stops: Scalar,
    stop_stiffness: Hinge,
    ground: Plane,
    surfaces: Quadric,
    contact_stiffness: Scalar,
    state: State,
) -> Scalar:
    """Total mechanical energy (`[...] Scalar+`) sums kinetic energy from the mass matrix blocks with gravitational, hinge-spring, joint-stop, and ground-spring potential energies."""
    _, _, _, _, balls, _ = tree
    motor, rate, angles, speeds = state
    motors = forward_kinematics(tree, home_axes, motor, angles)              # [...] Motor+[V]
    root_block, coupling, joint_block = mass_matrix(tree, home_axes, inertias, motors)
    kinetic = (
        (rate & root_block(rate)) / 2
        + (speeds & coupling(rate)).sites.sum()
        + (speeds & joint_block(speeds)).sites.sum() / 2
    )
    overruns = (angles - high_stops).clip(0, np.inf) + (angles - low_stops).clip(-np.inf, 0)   # [...] Scalar+[E]
    spring_energy = (
        (angles & hinge_stiffness(angles)).sites.sum() / 2
        + (overruns & stop_stiffness(overruns)).sites.sum() / 2
    )
    ball_motors = (balls * motors).batch()                                   # [...] Motor+
    world_surfaces = ball_motors >> surfaces(ball_motors << Point)           # [...] Plane+ <- Point-
    bottoms = contact.lowest_point(world_surfaces, ground.dual()).field()    # [...] Point-[n_balls]
    depths = (-(ground & bottoms)).clip(0, np.inf)                           # [...] Scalar+[n_balls]
    # Dualizing downward gravity gives the upward level plane scaled by gravitational field strength.
    gravitational = (gravity.dual() & (motors >> mass_points)).sites.sum()   # [...] Scalar+
    return kinetic + gravitational + spring_energy + (contact_stiffness * depths * depths).sites.sum() / 2
