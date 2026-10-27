"""Points breathing in and out along their rays on a fixed mesh, with the poles of its faces; a mesh
rescued by the nearest weights that keep every dual edge positive."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from scipy.optimize import LinearConstraint, linprog, minimize

from numga import concatenate
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
    """The directions are spread evenly over the sphere along a Fibonacci spiral, each nudged a little at random."""
    turns = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * turns / count)
    azimuth = np.pi * (1 + 5**0.5) * turns
    # The pole turned down towards x by its polar angle, then around the pole by its azimuth.
    down = (core.mv.zx * (-polar / 2)).exp()                                   # [points] Rotor+
    around = (core.mv.xy * (-azimuth / 2)).exp()                               # [points] Rotor+
    spiral = (around * down) >> core.mv.z.dual()                               # [points] Point-
    nudges = core.mv.vector(np.random.default_rng(DIRECTION_SEED).normal(size=(count, 3))).dual()   # [points] Point-
    return (spiral + JITTER * nudges).normalized()                             # [points] Point-


DIRECTIONS = directions(N_POINTS)


def breath() -> Iterator[tuple[core.Point, np.ndarray, core.Point]]:
    """The breath yields, at each frame, the points along their fixed directions at their own distances, the
    faces of the mesh they had on the sphere, held throughout, and the poles of those faces."""
    faces = core.hull(DIRECTIONS)                                              # [faces, 3]
    phases = np.random.default_rng(PHASE_SEED).uniform(0, 2 * np.pi, N_POINTS)
    radii = 1 + AMPLITUDE * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)[:, None] + phases)   # [frames, points]
    for points in DIRECTIONS * radii:                                          # [points] Point-
        yield points, faces, core.poles(points, faces)


def widest(conditions: core.Scalar) -> tuple[np.ndarray, float]:
    """The widest margin is the largest smallest condition, each scaled to unit size, over the weights within
    their bounds and summing to their count; it is negative when no weights meet every condition. A linear
    program finds it, with its weights, outside the algebra."""
    # The conditions as numbers for the solver, one row per condition and one column per point.
    matrix = conditions.to_array().T                                           # [conditions, points]
    scaled = matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)
    count = matrix.shape[-1]
    # The weights, then the smallest condition, which is maximised.
    largest = np.concatenate([np.zeros(count), [-1.0]])
    total = np.concatenate([np.ones(count), [0.0]])[None]
    result = linprog(largest, A_ub=np.hstack([-scaled, np.ones((len(scaled), 1))]), b_ub=np.zeros(len(scaled)),
                     A_eq=total, b_eq=[count], bounds=[WEIGHT_BOUNDS] * count + [(None, None)])
    return result.x[:-1], result.x[-1]


def nearest(conditions: core.Scalar, margin: float, start: np.ndarray) -> np.ndarray:
    """The nearest weights to the start, by the summed squares of the changes, leave every condition, scaled to
    unit size, at least the given margin, which shares the change among many points rather than a few. A
    quadratic program finds them outside the algebra."""
    # The conditions as numbers for the solver, one row per condition and one column per point.
    matrix = conditions.to_array().T                                           # [conditions, points]
    scaled = matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)
    result = minimize(lambda weights: ((weights - start) ** 2).sum(), start, jac=lambda weights: 2 * (weights - start),
                      method="SLSQP", constraints=[LinearConstraint(scaled, margin, np.inf)], bounds=[WEIGHT_BOUNDS] * len(start))
    return result.x


def settle() -> Iterator[tuple[core.Point, np.ndarray, core.Point]]:
    """The settling yields, at every frame, the points, faces and poles on the straight line in the weights from
    the hull of points at random distances along their rays, held with the points back on the sphere, where
    some of its dual edges cross over, to the nearest weights that unfold it with every pole inside its face,
    closing in on them at the end."""
    radii = np.exp(MESH_SPREAD * np.random.default_rng(MESH_SEED).normal(size=N_POINTS))   # [points]
    faces = core.hull(DIRECTIONS * radii)                                      # [faces, 3]
    start = np.ones(N_POINTS)                                                  # [points]
    # Every fold outward and every pole inside its face, each linear in the weights: at one unit weighting per
    # point, all as one batch, they give the columns of the map from weights to conditions.
    unit_weights = np.eye(N_POINTS)                                            # [points, points]
    conditions = concatenate([core.folds(DIRECTIONS, unit_weights, core.Mesh(N_POINTS, faces)),
                              core.insides(DIRECTIONS, unit_weights, faces).reshape(N_POINTS, -1)], axis=-1)   # [points, conditions] Scalar+
    rescue = nearest(conditions, MARGIN_SHARE * widest(conditions)[1], start)  # [points]
    for points in DIRECTIONS / (start + core.timing(N_SETTLE_FRAMES)[:, None] * (rescue - start)):   # [points] Point-
        yield points, faces, core.poles(points, faces)


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
    mesh = core.Mesh(N_POINTS, settled[0][1])
    crossed = [(core.dual_lengths(points, mesh) < 0).sum() for points, _, _ in settled]
    assert np.array_equal(core.dual_lengths(settled[0][0], mesh) < 0, core.folds(DIRECTIONS, np.ones(N_POINTS), mesh) < 0)
    assert crossed[0] > 0 and crossed[-1] == 0 and (np.diff(crossed) <= 0).all()
    # At the rescue every pole lies inside its face.
    assert (core.sides(*settled[-1]) > 0).all()


if __name__ == "__main__":
    main()
