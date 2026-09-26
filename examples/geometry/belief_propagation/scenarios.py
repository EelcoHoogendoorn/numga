"""Scenes for the pose graph: a lap and a bit, first with odometry alone, then with the loop closed
where the robot passes a pose of its first lap again; in the plane, and in space.

The core is instantiated for the algebra of the true steps, so the same scenes run on a lap in PGA2D
and on one in PGA3D.
"""

from __future__ import annotations

from itertools import accumulate
from types import ModuleType

import numpy as np

from numga import concatenate, stack
from numga.algebras import PGA2D, PGA3D
from examples import instantiate

CORE = "examples.geometry.belief_propagation.core"
# A lap and a bit in the plane, and a short one in space: poses, and poses to a lap.
PLANE_POSES, PLANE_LAP = 40, 36
SPACE_POSES, SPACE_LAP = 8, 6
# How far a reading is trusted, and how well the first pose is known, as translation and rotation
# standard deviations. The other poses are known before any reading only vaguely, near where dead
# reckoning puts them: enough to give every belief a covariance before anything has been told.
READING = (0.03, 0.01)
START = (0.01, 0.01)
VAGUE = (100.0, 30.0)
# How far out the quadrics lie, in standard deviations.
SIGMAS = 2.0


# --- math -----------------------------------------------------------------------------
def chain(steps, rounds: int, seed: int):
    """Odometry alone: a chain of readings from the known start, along the true steps. Returns the true
    and dead-reckoned poses, and after every round the poses and the quadrics of their beliefs: ellipses
    in the plane, ellipsoids in space."""
    core = instantiate(CORE, steps.algebra)
    truth, dead, graph = survey(core, steps, np.zeros((0, 2), int), seed)
    poses, informations = settle(core.propagate(graph, dead, graph.untold(), rounds))                         # [rounds, poses] Motor, Information
    quadrics = core.position_quadric(poses, informations.inverse(), origin(core), SIGMAS)   # [rounds, poses] Quadric

    # --- checks
    # Dead reckoning is already the most likely: the gradient vanishes. Once what is told has crossed
    # the chain, each belief's covariance is the exact one.
    np.testing.assert_allclose(core.gradient(graph, poses[-1]).kernel, 0.0, atol=1e-8)
    lines = basis(core)                                                        # [lines] Line
    exact = core.exact_covariance(graph, poses[-1], lines, len(poses[-1]))     # [lines, poses] Twist
    believed = informations[-1].solve(lines[:, None])                      # [lines, poses] Twist
    np.testing.assert_allclose((exact - believed).kernel, 0.0, atol=1e-10)
    return truth, dead, poses, quadrics


def loop(steps, lap: int, rounds: int, seed: int):
    """The loop closed between the last pose and the pose `lap` poses before it, along the true steps.
    Returns the true and dead-reckoned poses, after every round the poses and the quadrics of their
    beliefs, and for each pose the ratio of its believed to its exact variance along each basis line."""
    core = instantiate(CORE, steps.algebra)
    truth, dead, graph = survey(core, steps, np.array([[len(steps), len(steps) - lap]]), seed)
    poses, informations = settle(core.propagate(graph, dead, graph.untold(), rounds))                         # [rounds, poses] Motor, Information
    quadrics = core.position_quadric(poses, informations.inverse(), origin(core), SIGMAS)   # [rounds, poses] Quadric
    lines = basis(core)                                                        # [lines] Line
    exact = core.exact_covariance(graph, poses[-1], lines, rounds)             # [lines, poses] Twist
    believed = informations[-1].solve(lines[:, None])                      # [lines, poses] Twist
    ratios = (lines[:, None] & believed) / (lines[:, None] & exact)            # [lines, poses] Scalar

    # --- checks
    # The poses have settled at the most likely ones: the gradient has fallen a millionfold from dead
    # reckoning.
    start = np.abs(core.gradient(graph, dead).kernel).max()
    np.testing.assert_allclose(core.gradient(graph, poses[-1]).kernel, 0.0, atol=1e-6 * start)
    return truth, dead, poses, quadrics, ratios


def closing(steps, lap: int, rollout: int, rounds: int, seed: int):
    """Odometry alone while what is told rolls out along the chain from the known start, for the given
    rounds; then the loop closed between the last pose and the pose `lap` poses before it, carrying on
    from what was told. Returns the true and dead-reckoned poses, and after every round of both the
    poses and the quadrics of their beliefs."""
    core = instantiate(CORE, steps.algebra)
    truth, dead, chain_graph = survey(core, steps, np.zeros((0, 2), int), seed)
    # The odometry's noise is drawn first, so the closed graph reads the same steps.
    _, _, loop_graph = survey(core, steps, np.array([[len(steps), len(steps) - lap]]), seed)
    poses, informations = settle(rolled_out_then_closed(core, chain_graph, loop_graph, dead, rollout, rounds))
    quadrics = core.position_quadric(poses, informations.inverse(), origin(core), SIGMAS)   # [rounds, poses] Quadric

    # --- checks
    # Closing the loop settles the poses: the gradient has fallen a millionfold from where the rollout
    # left it.
    start = np.abs(core.gradient(loop_graph, poses[rollout - 1]).kernel).max()
    np.testing.assert_allclose(core.gradient(loop_graph, poses[-1]).kernel, 0.0, atol=1e-6 * start)
    return truth, dead, poses, quadrics


def rolled_out_then_closed(core: ModuleType, chain_graph, loop_graph, dead, rollout: int, rounds: int):
    """Every round of belief propagation along the chain, then around the closed loop from what the
    chain's readings have told."""
    poses, told = yield from core.propagate(chain_graph, dead, chain_graph.untold(), rollout)
    yield from core.propagate(loop_graph, poses, loop_graph.continued(told), rounds)


# --- plumbing -------------------------------------------------------------------------
def growing(poses, quadrics, rollout: int, closed: np.ndarray):
    """Frames of the lap: during the rollout, the poses what is told has reached, one more each round;
    after the loop is closed, every pose, after each of the given rounds. Yields the number of poses
    shown, and their poses and quadrics."""
    for index in range(rollout):
        yield index + 2, poses[index, :index + 2], quadrics[index, :index + 2]
    for index in rollout + closed:
        yield poses.shape[-1], poses[index], quadrics[index]


def origin(core: ModuleType):
    """The point each pose carries: the origin, dual to the weight direction w."""
    return core.mv.w.dual()                                                    # [] Point


def lap_in_plane():
    """Steps of 0.7 m in the plane, turning a lap every PLANE_LAP poses, faster and slower twice a lap."""
    mv = instantiate(CORE, PGA2D).mv
    phase = 2 * np.pi * np.arange(PLANE_POSES - 1) / PLANE_LAP
    return ((mv.xy * (1 + 0.4 * np.sin(2 * phase)) * 2 * np.pi / PLANE_LAP + mv.xw * 0.7) * 0.5).exp()   # [poses - 1] Motor


def lap_in_space():
    """A short lap in space, a turn of SPACE_LAP steps of 0.7 m, pitching up and down and rolling side
    to side once a lap."""
    mv = instantiate(CORE, PGA3D).mv
    phase = 2 * np.pi * np.arange(SPACE_POSES - 1) / SPACE_LAP
    turn = mv.xy * 2 * np.pi / SPACE_LAP                                       # [] Twist
    sway = (mv.zx * np.cos(phase) + mv.yz * np.sin(phase)) * 0.1               # [poses - 1] Twist
    return ((turn + sway + mv.xw * 0.7) * 0.5).exp()                           # [poses - 1] Motor


def basis(core: ModuleType):
    """The basis lines, reading out a twist's coefficients."""
    return core.mv(core.Line, np.eye(len(core.Line.output_subspace)))          # [lines] Line


def survey(core: ModuleType, steps, closures: np.ndarray, seed: int):
    """The poses along the true steps, read between consecutive poses and across the given pairs of
    poses. Returns the true poses, the dead-reckoned ones, and the pose graph."""
    rng = np.random.default_rng(seed)
    reading, start, vague = core.information(*READING), core.information(*START), core.information(*VAGUE)
    truth = stack(list(accumulate(steps, lambda pose, step: pose * step, initial=core.mv.rotor())))      # [poses] Motor
    tails, heads = np.concatenate([np.stack([np.arange(len(steps)), np.arange(1, len(steps) + 1)], axis=-1), closures]).T
    # The odometry's noise first, so that closing the loop leaves it as it was.
    noise = concatenate([sample(core, reading.inverse(), rng, len(steps)), sample(core, reading.inverse(), rng, len(closures))])   # [readings] Twist
    readings = (truth[tails].inverse() * truth[heads]) * (noise * 0.5).exp()   # [readings] Motor
    dead = stack(list(accumulate(readings[:len(steps)], lambda pose, step: pose * step, initial=truth[0])))      # [poses] Motor
    anchors = concatenate([truth[:1], dead[1:]])                               # [poses] Motor
    priors = concatenate([start[None], vague * np.ones(len(steps))])           # [poses] Information
    return truth, dead, core.PoseGraph(np.stack([heads, tails]), readings, reading * np.ones(len(tails)), anchors, priors)


def settle(rounds):
    """The poses and their beliefs' information after every round of belief propagation, stacked."""
    history, informations = zip(*rounds)
    return stack(history), stack(informations)


def sample(core: ModuleType, covariance, rng: np.random.Generator, count: int):
    """Twists drawn from a covariance: independent unit draws along lines orthonormal in its form."""
    spread = core.Line & covariance                                            # [] Scalar <- (Line, Line)
    _, lines = spread.eigh(spread)                                             # [modes] Line
    return (covariance(lines) * rng.normal(size=(count,) + lines.shape)).sum(axis=-1)   # [count] Twist


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.geometry.belief_propagation import render

    # A seed whose dead reckoning drifts clearly, near a metre over the lap.
    closed = loop(lap_in_plane(), PLANE_LAP, 400, 13)
    for where, ratios in (("plane", closed[-1]), ("space", loop(lap_in_space(), SPACE_LAP, 100, 0)[-1])):
        ratios = ratios.to_array()
        print(f"loop closed in {where}, believed over exact variance: {ratios.min():.3f} to {ratios.max():.3f}")
    # Each round carries what is told one pose further, so one round per step rolls it out along the
    # whole chain; then the loop is closed, and the frames after it move evenly.
    rollout = PLANE_POSES - 1
    truth, dead, poses, quadrics = closing(lap_in_plane(), PLANE_LAP, rollout, 400, 13)
    save_figure(render.draw_survey({"rolled out, then closed": (truth, dead, poses, quadrics)}), "belief_propagation")
    closed = render.evenly_moving(poses[rollout - 1:], 24) - 1
    save_animation(render.animate_growth(truth, dead, growing(poses, quadrics, rollout, closed)), "belief_propagation", 80)
