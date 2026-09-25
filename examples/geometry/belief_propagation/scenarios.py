"""Scenes for the pose graph: a lap and a bit, first with odometry alone, then with the loop closed
where the robot passes a pose of its first lap again."""

from __future__ import annotations

from itertools import accumulate

import numpy as np

from numga import concatenate, stack
from examples.geometry.belief_propagation import core

mv = core.mv
POSES, PER_LAP = 40, 36
# How far a reading is trusted, and how well the first pose is known. The other poses are known
# before any reading only vaguely, near where dead reckoning puts them: enough to give every belief
# a covariance before anything has been told.
READING = core.information(0.03, 0.01)
START = core.information(0.01, 0.01)
VAGUE = core.information(100.0, 30.0)
# The basis lines, reading out a twist's coefficients.
LINES = mv(core.Line, np.eye(len(core.Line.output_subspace)))                 # [lines] Line
# The ellipses drawn, in standard deviations.
SIGMAS = 2.0


# --- math -----------------------------------------------------------------------------
def chain(rounds: int, seed: int) -> tuple[core.Motor, core.Motor, core.Motor, core.Quadric]:
    """Odometry alone: a chain of readings from the known start. Returns the true and dead-reckoned
    poses, and after every round the poses and the ellipses of their beliefs."""
    truth, dead, graph = survey(np.zeros((0, 2), int), seed)
    poses, informations = settle(graph, dead, rounds)                         # [rounds, poses] Motor, Information
    ellipses = core.position_quadric(poses, informations.inverse(), core.ORIGIN, SIGMAS)   # [rounds, poses] Quadric

    # --- checks
    # Dead reckoning is already the most likely: the gradient vanishes. Once what is told has crossed
    # the chain, each belief's covariance is the exact one.
    np.testing.assert_allclose(core.gradient(graph, poses[-1]).kernel, 0.0, atol=1e-8)
    exact = core.exact_covariance(graph, poses[-1], LINES, POSES)              # [lines, poses] Twist
    believed = informations[-1].inverse()(LINES[:, None])                      # [lines, poses] Twist
    np.testing.assert_allclose((exact - believed).kernel, 0.0, atol=1e-10)
    return truth, dead, poses, ellipses


def loop(rounds: int, seed: int) -> tuple[core.Motor, core.Motor, core.Motor, core.Quadric, core.Scalar]:
    """The loop closed between the last pose and the pose one lap before it. Returns the true and
    dead-reckoned poses, after every round the poses and the ellipses of their beliefs, and for each
    pose the ratio of its believed to its exact variance along each basis line."""
    truth, dead, graph = survey(np.array([[POSES - 1, POSES - 1 - PER_LAP]]), seed)
    poses, informations = settle(graph, dead, rounds)                         # [rounds, poses] Motor, Information
    ellipses = core.position_quadric(poses, informations.inverse(), core.ORIGIN, SIGMAS)   # [rounds, poses] Quadric
    exact = core.exact_covariance(graph, poses[-1], LINES, rounds)             # [lines, poses] Twist
    believed = informations[-1].inverse()(LINES[:, None])                      # [lines, poses] Twist
    ratios = (LINES[:, None] & believed) / (LINES[:, None] & exact)            # [lines, poses] Scalar

    # --- checks
    # The poses have settled at the most likely ones: the gradient has fallen a millionfold from dead
    # reckoning.
    start = np.abs(core.gradient(graph, dead).kernel).max()
    np.testing.assert_allclose(core.gradient(graph, poses[-1]).kernel, 0.0, atol=1e-6 * start)
    return truth, dead, poses, ellipses, ratios


# --- plumbing -------------------------------------------------------------------------
def survey(closures: np.ndarray, seed: int) -> tuple[core.Motor, core.Motor, core.PoseGraph]:
    """A wobbly lap and a bit, read between consecutive poses and across the given pairs of poses.
    Returns the true poses, the dead-reckoned ones, and the pose graph."""
    rng = np.random.default_rng(seed)
    # Steps of 0.7 m, turning a lap every PER_LAP poses, faster and slower twice a lap.
    phase = 2 * np.pi * np.arange(POSES - 1) / PER_LAP
    steps = ((mv.xy * (1 + 0.4 * np.sin(2 * phase)) * 2 * np.pi / PER_LAP - mv.wx * 0.7) * 0.5).exp()   # [poses - 1] Motor
    truth = stack(list(accumulate(steps, lambda pose, step: pose * step, initial=mv.rotor())))           # [poses] Motor
    tails, heads = np.concatenate([np.stack([np.arange(POSES - 1), np.arange(1, POSES)], axis=-1), closures]).T
    # The odometry's noise first, so that closing the loop leaves it as it was.
    noise = concatenate([sample(READING.inverse(), rng, POSES - 1), sample(READING.inverse(), rng, len(closures))])   # [readings] Twist
    readings = (truth[tails].inverse() * truth[heads]) * (noise * 0.5).exp()   # [readings] Motor
    dead = stack(list(accumulate(readings[:POSES - 1], lambda pose, step: pose * step, initial=truth[0])))       # [poses] Motor
    anchors = concatenate([truth[:1], dead[1:]])                               # [poses] Motor
    priors = concatenate([START[None], VAGUE * np.ones(POSES - 1)])            # [poses] Information
    return truth, dead, core.PoseGraph(np.stack([heads, tails]), readings, READING * np.ones(len(tails)), anchors, priors)


def settle(graph: core.PoseGraph, poses: core.Motor, rounds: int) -> tuple[core.Motor, core.Information]:
    """The poses and their beliefs' information after every round of belief propagation."""
    history, informations = zip(*core.propagate(graph, poses, rounds))
    return stack(history), stack(informations)


def sample(covariance: core.Covariance, rng: np.random.Generator, count: int) -> core.Twist:
    """Twists drawn from a covariance: independent unit draws along lines orthonormal in its form."""
    spread = core.Line & covariance                                            # [] Scalar <- (Line, Line)
    _, lines = spread.eigh(spread)                                             # [modes] Line
    return (covariance(lines) * rng.normal(size=(count,) + lines.shape)).sum(axis=-1)   # [count] Twist


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.geometry.belief_propagation import render

    alone = chain(400, 0)
    closed = loop(400, 0)
    ratios = closed[-1].to_array()
    print(f"loop closed, believed over exact variance: {ratios.min():.3f} to {ratios.max():.3f}")
    runs = {"odometry alone": alone, "loop closed": closed[:-1]}
    save_figure(render.draw_survey(runs), "belief_propagation")
    save_animation(render.animate_survey(runs, 60), "belief_propagation", 80)
