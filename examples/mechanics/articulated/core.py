"""Reduced-coordinate dynamics of a branching robot in PGA3D: a free root body with limbs, such as Brax's humanoid,
its state the root's motor and rate and its joints' angles and speeds, every quantity of a link carried between frames
by the link's motor.

A joint's axis is a line, a bivector, and so is a link's rate; the link's momentum and the forques on it are lines
joined from points, antibivectors. One motor sandwich moves them all, and the inertia, a map from rates to forques, as
`motor >> inertia(motor << Rate)`. The commutator is both of the cross products of spatial vector algebra: the change of
a joint's axis as its link turns, `rate.commutator(axis)`, and the change of a momentum under motion,
`rate.commutator(momentum)`, the orientation of each operand keeping motions and forces apart.

The links' rates are the root's rate plus the joints' speeds summed along each path from the root, a sparse map whose
cells are the joints' axes. Its adjugate runs the couplings back, summing the forques of each joint's subtree and
pairing them with the joint's axis, `axis & forque`; the root's share is the sum of all the forques. The mass matrix is
the pullback of the links' inertias through these maps, kept in two blocks, the root's and the joints', since one
sparse map holds cells of one type. Recursive Newton–Euler inverse dynamics is the maps forward, for rates and
accelerations, and back, for forques; forward dynamics solves the mass matrix with the root's block eliminated.
A linearly implicit step solves the same pullback with the damping and stiffness of the hinges, of their stops and of
the ground added: the ground's contact at a ball's lowest point, joined with the point, is extra inertia on its link.
"""

from __future__ import annotations

from functools import partial
from collections.abc import Callable
from typing import NamedTuple

import jax
import numpy as np

from numga import Extensor, where
from numga.algebras import PGA3D as ga
from numga.backend.jax import JaxContext
from numga.sparse import SparseExtensor, spdiag
from examples.mechanics.spinning_top import core as contact
from examples.util.topology import Topology

jax.config.update("jax_enable_x64", True)
mv = JaxContext(ga, dtype=np.float64).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-
Plane = ga.gatype.vector()                                                   # Vector+
# A joint's axis per unit speed, and a link's rate: a line to turn about, scaled.
Rate = ga.gatype.bivector()                                                  # Bivector+
# A link's momentum and the forques on it: lines joined from points.
Forque = ga.gatype.antibivector()                                            # Antibivector-
Motor = ga.gatype.rotor()                                                    # Rotor+
# A joint's momentum and its generalized force: what a joint's speed pairs with into energy and power.
Amount = ga.gatype.antiscalar()                                              # Antiscalar-
Inertia = ga.gatype((Forque, Rate))
Hinge = ga.gatype((Amount, Scalar))
Quadric = ga.gatype((Plane, Point))


class Tree(NamedTuple):
    """The robot's tree: its root, the body that carries the rest, `numbers[1 <- links]`, each joint's parent and child
    link, `numbers[joints <- links]` each, the path from the root to each link, `numbers[links <- joints]`, one wherever
    a joint lies on it, the link that carries each ball the ground can meet, `numbers[balls <- links]`, and how many
    joints the longest path passes."""
    body: SparseExtensor
    tails: SparseExtensor
    heads: SparseExtensor
    paths: SparseExtensor
    balls: SparseExtensor
    n_generations: int


class Robot(NamedTuple):
    """The robot in its home pose, its root at the origin and its joints at angle zero: each joint's axis,
    `Rate[joints]`, and each link's inertia, `(Forque <- Rate)[links]`, and mass point, its centre weighted by its mass,
    `Point[links]`."""
    axes: Rate
    inertias: Inertia
    mass_points: Point


class Physics(NamedTuple):
    """What acts on the robot always: gravity, an acceleration along a direction, and the hinges' springs and dampers,
    their stiffness and damping, `Scalar[joints]` each, the springs pulling towards the home pose."""
    gravity: Point
    hinge_stiffness: Scalar
    hinge_damping: Scalar


class Ground(NamedTuple):
    """A ground the robot can meet: a plane facing up, the surfaces of the robot's balls that can meet it in the home
    pose, solid quadrics `[balls] Plane <- Point`, its push on each ball per depth below it, `Scalar[balls]`, the share
    the push grows by per speed of sinking, and its friction per push and speed of slip."""
    plane: Plane
    balls: Quadric
    stiffness: Scalar
    damping: float
    friction: float


class Stops(NamedTuple):
    """The stops at the ends of the joints' ranges: the ends and the stops' stiffness, `Scalar[joints]` each."""
    low: Scalar
    high: Scalar
    stiffness: Scalar


class State(NamedTuple):
    """The robots' state, `[robots]`: the body's motor and its rate, `Motor` and `Rate`, and the joints' angles and
    speeds, `Scalar[joints]` each."""
    motor: Motor
    rate: Rate
    angles: Scalar
    speeds: Scalar


# --- math -----------------------------------------------------------------------------
def tree(topology: Topology, carriers: np.ndarray, n_generations: int) -> Tree:
    """The tree of a topology of edges, each from a parent link to a child link, the root its only link that is no
    edge's head, and the links that carry the balls, `[balls]`.

    A link's rate is its parent's plus its own joint's, so the path to each link is its own joint, then its parent's
    path: the series `~heads + parents * ~heads + ...`, which ends after as many generations as the tree has."""
    # Each joint's two ends, tail then head, each read on its own, so that every map keeps only the couplings it has.
    tails, heads = (topology.reading(ends) for ends in topology.elements[1].T)
    # The value of each link's parent, at the link: zero at the root, which has none.
    parents = ~heads * tails                                                 # numbers[links <- links]
    paths = ~heads
    for _ in range(n_generations - 1):
        paths = ~heads + parents * paths
    root = np.setdiff1d(np.arange(topology.n_vertices), topology.elements[1][:, 1])
    return Tree(topology.reading(root), tails, heads, paths, topology.reading(carriers), n_generations)


@partial(jax.jit, static_argnames=("tree",))
def forward_kinematics(tree: Tree, robot: Robot, motor: Motor, angles: Scalar) -> Motor:
    """Each link's motor, `Motor[links]`, with the body placed by its motor and the joints at their angles: its parent's
    motor followed by its own joint's turn about its axis in the home pose, one generation of the tree a sweep."""
    turns = (robot.axes * (angles / 2)).exp()                                # Motor+[joints]
    placed = ~tree.body * motor[..., None].field()                          # Motor+[links]
    motors = placed
    for _ in range(tree.n_generations):
        motors = placed + ~tree.heads * ((tree.tails * motors) * turns)
    return motors


def inverse_dynamics(tree: Tree, robot: Robot, motors: Motor, rate: Rate, speeds: Scalar, acceleration: Rate,
                     accelerations: Scalar, external: Forque) -> tuple[Forque, Amount]:
    """The generalized forces, the body's forque and the joints' amounts, `Amount[joints]`, that give the body and the
    joints their accelerations at their rate and speeds, under forques on the links from outside: recursive
    Newton–Euler, rates and accelerations out along every path of the tree and forques back."""
    # Each joint's axis where its parent link has carried it.
    axes = (tree.tails * motors) >> robot.axes                               # Rate+[joints]
    joint_rates = tree.paths * spdiag(axes * Scalar)                         # (Rate+ <- Scalar+)[links <- joints]
    inertia = spdiag(motors >> robot.inertias(motors << Rate))               # (Forque- <- Rate+)[links <- links]
    # Out from the body. A joint's axis turns with its child link, so its speed adds the velocity product,
    # `rate.commutator(axis) * speed`, to the acceleration of that link and of every link beyond it.
    rates = joint_rates(speeds) + rate                                       # Rate+[links]
    velocity_products = (tree.heads * rates).commutator(axes) * speeds      # Rate+[joints]
    link_accelerations = joint_rates(accelerations) + tree.paths * velocity_products + acceleration
    # The forque each link needs: the change of its momentum, by its acceleration and as its own motion turns it,
    # less what acts on it from outside.
    momenta = inertia(rates)                                                 # Forque-[links]
    forques = inertia(link_accelerations) + rates.commutator(momenta) - external
    # Back to the body, which bears all of them, and to each joint: the adjugate sums the forques of its subtree and
    # pairs the sum with its axis.
    return forques.sites.sum(), joint_rates.adjugate()(forques)


def pullback(joint_rates: SparseExtensor, link_maps: Inertia) -> tuple[Inertia, Extensor, SparseExtensor]:
    """Maps from each link's rate to a forque, `(Forque <- Rate)[links]`, pulled back onto the body's rate and the
    joints' speeds, in blocks: the body's, the links' maps summed, as the body's rate reaches every link alike; the
    joints' per rate of the body, `Amount[joints] <- Rate`, whose adjugate is the body's per speed of the joints; and the
    joints' own, `(Amount <- Scalar)[joints <- joints]`, coupling two joints only where one lies on the other's path."""
    return (link_maps.sites.sum(), joint_rates.adjugate()(link_maps),
            joint_rates.adjugate()(spdiag(link_maps)(joint_rates)))


def mass_matrix(tree: Tree, robot: Robot, motors: Motor) -> tuple[Inertia, Extensor, SparseExtensor]:
    """The mass matrix in blocks, the links' inertias where they are pulled back."""
    axes = (tree.tails * motors) >> robot.axes                               # Rate+[joints]
    joint_rates = tree.paths * spdiag(axes * Scalar)                         # (Rate+ <- Scalar+)[links <- joints]
    return pullback(joint_rates, motors >> robot.inertias(motors << Rate))


def solved(whole: Inertia, coupling: Extensor, joints: SparseExtensor, forque: Forque,
           forces: Amount) -> tuple[Rate, Scalar]:
    """The body's rate and the joints' speeds that a map in blocks takes to a forque on the body and forces at the
    joints: the body's block eliminated, the body giving way to the joints' forces through it. The sparse solve runs on
    SciPy alone, so each robot's small block is solved dense."""
    reduced = joints.to_dense() - coupling(whole.solve(coupling.adjugate()))   # (Amount- <- Scalar+)[joints <- joints]
    speeds = reduced.solve(forces - coupling(whole.solve(forque)))           # Scalar+[joints]
    return whole.solve(forque - coupling.adjugate()(speeds)), speeds


def forward_dynamics(tree: Tree, robot: Robot, motors: Motor, rate: Rate, speeds: Scalar, forces: Amount,
                     external: Forque) -> tuple[Rate, Scalar]:
    """The body's acceleration and the joints', under generalized forces at the joints, at their rate and speeds, under
    forques on the links from outside, a forque on the body among them: the mass matrix solved against the forces less
    the bias, the forces inverse dynamics gives at no acceleration."""
    body_bias, joint_bias = inverse_dynamics(tree, robot, motors, rate, speeds, rate * 0, speeds * 0, external)
    return solved(*mass_matrix(tree, robot, motors), -body_bias, forces - joint_bias)


def own_loads(tree: Tree, robot: Robot, physics: Physics, motors: Motor, rates: Rate, angles: Scalar, speeds: Scalar,
              dt: float) -> tuple[Forque, Inertia, Amount, Hinge]:
    """What acts on the robot always over a step: on the links their weights, gravity joined with their mass points,
    and their inertias, `Forque[links]` and `(Forque <- Rate)[links]`; on the joints the hinges' forces, the springs as
    the present speeds will have bent them by the step's end, and the inertia the springs and the dampers carry over
    the step, `Amount[joints]` and `(Amount <- Scalar)[joints]`."""
    weights = physics.gravity & (motors >> robot.mass_points)                # Forque-[links]
    inertias = motors >> robot.inertias(motors << Rate)                      # (Forque- <- Rate+)[links]
    # Each hinge's maps from its angle and its speed to its generalized force, an amount.
    stiffness, damping = physics.hinge_stiffness * Scalar.dual(), physics.hinge_damping * Scalar.dual()
    forces = -(spdiag(stiffness)(angles + speeds * dt) + spdiag(damping)(speeds))   # Amount-[joints]
    return weights, inertias, forces, damping * dt + stiffness * dt**2


def bottoms(tree: Tree, ground: Ground, motors: Motor) -> Point:
    """Each ball's lowest point against the ground's own direction up, `Point[balls]`, the ball where its link has
    carried it."""
    carried = (tree.balls * motors).batch()                                  # [..., balls] Motor+
    return contact.lowest_point(carried >> ground.balls(carried << Point), ground.plane.dual()).field()


def ground_contact(tree: Tree, ground: Ground, motors: Motor, rates: Rate, dt: float) -> tuple[Forque, Inertia]:
    """The ground's forques on the links over a step, `Forque[links]`, and the inertia its contact carries over the
    step, `(Forque <- Rate)[links]`.

    Below the ground a ball's lowest point is pushed along the ground's own direction up, by a spring whose push grows
    with the speed of sinking, damped in proportion to its depth, and rubbed against its slip in proportion to the push.
    Joined with the point as an inertia joins its points with their velocities, the push's damping and friction, at the
    present push, and its stiffness are maps from the link's rate to a forque, which the step carries over its span;
    the spring pushes as the present rates will have pressed it by the step's end."""
    lowest = bottoms(tree, ground, motors)                                   # Point-[balls]
    ball_rates = tree.balls * rates                                          # Rate+[balls]
    velocities = ball_rates.commutator(lowest)                               # Point-[balls]
    # A direction's part along up, and how deep each lowest point lies and how fast it sinks.
    up = ground.plane.dual()                                                 # [] Point-
    along = up * (ground.plane & Point)                                      # [] Point- <- Point-
    depths = (-(ground.plane & lowest)).clip(0, np.inf)                      # Scalar+[balls]
    sinking = -(ground.plane & velocities)                                   # Scalar+[balls]
    # The push never pulls a ball down, however fast it rises.
    pushes = (depths * ground.stiffness * (1 + ground.damping * sinking)).clip(0, np.inf)   # Scalar+[balls]
    forces = up * pushes - (velocities - along(velocities)) * (pushes * ground.friction)   # Point-[balls]
    # How the force falls as the point moves and as it is moved: maps from a velocity and a displacement to a force,
    # the spring's stiffness where a ball presses into the ground and none where it does not.
    dampers = along * (depths * ground.stiffness * ground.damping) + (Point - along) * (pushes * ground.friction)
    springs = along * where(depths > 0, ground.stiffness, 0)                 # (Point- <- Point-)[balls]
    damping = lowest.regressive(dampers(lowest.commutator(Rate)))            # (Forque- <- Rate+)[balls]
    stiffness = lowest.regressive(springs(lowest.commutator(Rate)))          # (Forque- <- Rate+)[balls]
    forques = (forces & lowest) - spdiag(stiffness)(ball_rates) * dt         # Forque-[balls]
    return ~tree.balls * forques, ~tree.balls * (damping * dt + stiffness * dt**2)


def stop_loads(stops: Stops, angles: Scalar, speeds: Scalar, dt: float) -> tuple[Amount, Hinge]:
    """The stops' forces on the joints over a step, `Amount[joints]`, each as the present speed will have pressed it by
    the step's end, and the inertia they carry over the step, `(Amount <- Scalar)[joints]`."""
    # How far each joint lies past its stops, and the stops' stiffness where it does, none where it does not.
    past = (angles - stops.high).clip(0, np.inf) + (angles - stops.low).clip(-np.inf, 0)   # Scalar+[joints]
    pressed = where(past.abs() > 0, stops.stiffness * Scalar.dual(), 0)      # (Amount- <- Scalar+)[joints]
    return -spdiag(pressed)(past + speeds * dt), pressed * dt**2


def drop_loads(tree: Tree, robot: Robot, physics: Physics, ground: Ground, stops: Stops, motors: Motor, rates: Rate,
               angles: Scalar, speeds: Scalar, dt: float) -> tuple[Forque, Inertia, Amount, Hinge]:
    """What acts on the robot dropped onto the ground over a step: what acts always, with the ground on the links and
    the stops on the joints."""
    forques, link_maps, forces, joint_maps = own_loads(tree, robot, physics, motors, rates, angles, speeds, dt)
    pushes, contact_maps = ground_contact(tree, ground, motors, rates, dt)
    stop_forces, stop_maps = stop_loads(stops, angles, speeds, dt)
    return forques + pushes, link_maps + contact_maps, forces + stop_forces, joint_maps + stop_maps


def energy(tree: Tree, robot: Robot, physics: Physics, ground: Ground, stops: Stops, state: State) -> Scalar:
    """The energy of robots dropped onto the ground: the links' momenta paired with their rates, halved, the work done
    against gravity from the level plane through the origin, and the work stored in the hinges' springs, in the stops
    past which the joints lie and in the ground under the balls."""
    motors = forward_kinematics(tree, robot, state.motor, state.angles)
    axes = (tree.tails * motors) >> robot.axes                               # Rate+[joints]
    rates = (tree.paths * spdiag(axes * Scalar))(state.speeds) + state.rate  # Rate+[links]
    link_inertias = motors >> robot.inertias(motors << Rate)                 # (Forque- <- Rate+)[links]
    kinetic = (spdiag(link_inertias)(rates) & rates).sites.sum() / 2
    # Gravity's dual is the level plane scaled by gravity's strength: its pairing with a mass point is the work of
    # lifting that mass to it from the plane through the origin.
    lifted = (physics.gravity.dual() & (motors >> robot.mass_points)).sites.sum()
    past = (state.angles - stops.high).clip(0, np.inf) + (state.angles - stops.low).clip(-np.inf, 0)
    sprung = (physics.hinge_stiffness * state.angles * state.angles + stops.stiffness * past * past).sites.sum() / 2
    depths = (-(ground.plane & bottoms(tree, ground, motors))).clip(0, np.inf)   # Scalar+[balls]
    return kinetic + lifted + sprung + (ground.stiffness * depths * depths).sites.sum() / 2


Loads = Callable[[Motor, Rate, Scalar, Scalar, float], tuple[Forque, Inertia, Amount, Hinge]]


def step(tree: Tree, robot: Robot, state: State, dt: float, loads: Loads) -> State:
    """The state dt later, by a linearly implicit Euler step: the change of the rates solves the step's matrix, the
    inertia everything acting carries over the step pulled back, against the forces over the step, each spring as the
    present rates will have moved it; the placement then moves at the new rates. The loads give what acts, from the
    links' placement and rates and the joints' angles and speeds. The velocity products stay at the step's start."""
    motors = forward_kinematics(tree, robot, state.motor, state.angles)
    axes = (tree.tails * motors) >> robot.axes                               # Rate+[joints]
    joint_rates = tree.paths * spdiag(axes * Scalar)                         # (Rate+ <- Scalar+)[links <- joints]
    rates = joint_rates(state.speeds) + state.rate                           # Rate+[links]
    forques, link_maps, forces, joint_maps = loads(motors, rates, state.angles, state.speeds, dt)
    body_bias, joint_bias = inverse_dynamics(tree, robot, motors, state.rate, state.speeds, state.rate * 0,
                                             state.speeds * 0, forques)
    whole, coupling, joints = pullback(joint_rates, link_maps)
    rate_change, speed_changes = solved(whole, coupling, joints + spdiag(joint_maps), -body_bias * dt,
                                        (forces - joint_bias) * dt)
    rate, speeds = state.rate + rate_change, state.speeds + speed_changes
    # The body turns about its new rate over the step, `motor' == rate * motor / 2`.
    return State((rate * (dt / 2)).exp() * state.motor, rate, state.angles + speeds * dt, speeds)


@partial(jax.jit, static_argnames=("tree", "n_steps", "loads"))
def advance(tree: Tree, robot: Robot, state: State, dt: float, n_steps: int, loads: Loads) -> State:
    """The state n_steps of dt later."""
    joint_type = state.angles.gatype

    def typed(state: State) -> State:
        """The state with each part of its full type, the same at every step."""
        return State(state.motor.retyped(Motor), state.rate.retyped(Rate), state.angles.retyped(joint_type),
                     state.speeds.retyped(joint_type))

    return jax.lax.fori_loop(0, n_steps, lambda index, state: typed(step(tree, robot, state, dt, loads)), typed(state))
