"""Points breathing in and out along their rays on a fixed mesh, with the poles of its faces; a mesh
rescued by the nearest weights that keep every dual edge positive."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from scipy.optimize import LinearConstraint, linprog, minimize

from examples.surfaces.weighted_dual import core

# The number of points, how far their directions are nudged off an even spread, the seeds of the nudges
# and of their phases, their spread in distance from the centre, and the frames of one breath and the
# duration of each.
N_POINTS = 32
JITTER = 0.03
DIRECTION_SEED, PHASE_SEED = 0, 3
AMPLITUDE = 0.06
N_FRAMES = 48
DURATION_MS = 80
# The spread and seed of the random distances whose hull gives the mesh the rescue starts from, every point
# on the hull; the bounds on the weights, the share of the widest margin of its conditions the rescue keeps, and the
# frames from the start to the rescue.
MESH_SPREAD, MESH_SEED = 0.06, 57
WEIGHT_BOUNDS = (0.5, 2.0)
MARGIN_SHARE = 0.3
N_SETTLE_FRAMES = 48


# --- math -----------------------------------------------------------------------------
def directions(count: int) -> core.Point:
    """Directions spread evenly over the sphere along a Fibonacci spiral, each nudged a little at random."""
    turns = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * turns / count)
    azimuth = np.pi * (1 + 5**0.5) * turns
    # The pole turned down towards x by its polar angle, then around the pole by its azimuth.
    down = ((core.mv.z ^ core.mv.x) * (-polar / 2)).exp()
    around = ((core.mv.x ^ core.mv.y) * (-azimuth / 2)).exp()
    spiral = (around * down) >> core.mv.z.dual()                              # [points] Point
    nudges = core.mv.vector(np.random.default_rng(DIRECTION_SEED).normal(size=(count, 3))).dual()   # [points] Point
    return (spiral + JITTER * nudges).normalized()                             # [points] Point


def breath() -> Iterator[tuple[core.Point, np.ndarray, core.Point]]:
    """At each frame: the points, each at its own distance from the centre along a fixed direction, the
    faces of the mesh they had on the sphere, held throughout, and the poles of those faces."""
    faces = core.hull(directions(N_POINTS))                                    # [faces, 3]
    phases = np.random.default_rng(PHASE_SEED).uniform(0, 2 * np.pi, N_POINTS)
    radii = 1 + AMPLITUDE * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)[:, None] + phases)   # [frames, points]
    for points in directions(N_POINTS) * radii:                                # [points] Point
        yield points, faces, core.poles(points, faces)                         # [faces] Point


def fold_map(faces: np.ndarray) -> np.ndarray:
    """The folds of every edge as a linear map of the points' weights, one column per point: the folds with
    that point's weight one and every other zero, all points as one batch."""
    return core.folds(directions(N_POINTS), np.eye(N_POINTS), faces).to_array().T   # [edges, points]


def widest(conditions: np.ndarray) -> tuple[np.ndarray, float]:
    """The weights that make the smallest condition, each scaled to unit size, as large as it can be, with
    the weights within their bounds and summing to their count, and that smallest condition: negative when
    no weights meet them all."""
    scaled = conditions / np.linalg.norm(conditions, axis=-1, keepdims=True)
    count = conditions.shape[-1]
    # The weights, then the smallest fold, which is maximised.
    largest = np.concatenate([np.zeros(count), [-1.0]])
    total = np.concatenate([np.ones(count), [0.0]])[None]
    result = linprog(largest, A_ub=np.hstack([-scaled, np.ones((len(scaled), 1))]), b_ub=np.zeros(len(scaled)),
                     A_eq=total, b_eq=[count], bounds=[WEIGHT_BOUNDS] * count + [(None, None)])
    return result.x[:-1], result.x[-1]


def inside_map(faces: np.ndarray) -> np.ndarray:
    """Every pole's signed distances inside its face as a linear map of the points' weights, one column per
    point: the distances with that point's weight one and every other zero, all points as one batch."""
    return np.moveaxis(core.insides(directions(N_POINTS), np.eye(N_POINTS), faces).to_array(), 0, -1)   # [faces, 3, points]


def nearest(conditions: np.ndarray, margin: float, start: np.ndarray) -> np.ndarray:
    """The weights nearest to the start, by the summed squares of the changes, that leave every condition,
    scaled to unit size, at least the given margin: the change shared among many points rather than a few."""
    scaled = conditions / np.linalg.norm(conditions, axis=-1, keepdims=True)
    result = minimize(lambda weights: ((weights - start) ** 2).sum(), start, jac=lambda weights: 2 * (weights - start),
                      method="SLSQP", constraints=[LinearConstraint(scaled, margin, np.inf)], bounds=[WEIGHT_BOUNDS] * len(start))
    return result.x


def settle() -> Iterator[tuple[core.Point, np.ndarray, core.Point]]:
    """The mesh of the hull of the points at random distances along their rays, held, with the points back on
    the sphere, where some of its dual edges cross over; then the weights on the straight line to the nearest
    weights that unfold it with every pole inside its face, closing in on them at the end: the points, faces
    and poles at every frame."""
    radii = np.exp(MESH_SPREAD * np.random.default_rng(MESH_SEED).normal(size=N_POINTS))   # [points]
    faces = core.hull(directions(N_POINTS) * radii)                            # [faces, 3]
    start = np.ones(N_POINTS)                                                  # [points]
    # Every fold outward, and every pole inside its face.
    conditions = np.concatenate([fold_map(faces), inside_map(faces).reshape(-1, N_POINTS)])   # [edges + faces * 3, points]
    rescue = nearest(conditions, MARGIN_SHARE * widest(conditions)[1], start)  # [points]
    for weights in start + timing(N_SETTLE_FRAMES)[:, None] * (rescue - start):   # [points]
        points = core.weighted(directions(N_POINTS), weights)                  # [points] Point
        yield points, faces, core.poles(points, faces)                         # [faces] Point


def timing(frames: int) -> np.ndarray:
    """How far along the way each frame is: at a steady pace for three quarters of the frames, then closing in
    on the end over the last quarter, the way left shrinking by so many e-folds, the pace continuous where it
    starts closing in."""
    closing_start, e_folds = 3 / 4, 3
    rate = e_folds / (1 - closing_start)
    left = 1 / (1 + closing_start * rate)                                      # the way left where closing in starts
    time = np.linspace(0, 1, frames)
    return np.where(time < closing_start, time * left * rate, 1 - left * np.exp(-(time - closing_start) * rate))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.weighted_dual import render

    frames = list(breath())
    save_figure(render.draw_complex(*frames[0]), "weighted_dual")
    save_animation(render.animate(frames), "weighted_dual_breath", DURATION_MS)
    settled = list(settle())
    save_animation(render.animate(settled), "weighted_dual_rescue", DURATION_MS)

    # checks
    for points, faces, duals in frames:
        # Every face's plane, the dual of its pole, pairs to one with each of the face's vertices.
        assert np.abs((duals.dual_inverse()[:, None] & points[faces]).to_array() - 1).max() < 1e-10

    # On the sphere some dual edges cross over, exactly where the polyhedron folds; along the straight line
    # to the rescue they only ever uncross, and at the rescue none is left.
    faces = settled[0][1]
    crossed = [(core.dual_lengths(points, faces).to_array() < 0).sum() for points, _, _ in settled]
    assert np.array_equal(core.dual_lengths(settled[0][0], faces).to_array() < 0,
                          core.folds(directions(N_POINTS), np.ones(N_POINTS), faces).to_array() < 0)
    assert crossed[0] > 0 and crossed[-1] == 0 and (np.diff(crossed) <= 0).all()
    # At the rescue every pole lies inside its face.
    assert (core.sides(*settled[-1]).to_array() > 0).all()


if __name__ == "__main__":
    main()
