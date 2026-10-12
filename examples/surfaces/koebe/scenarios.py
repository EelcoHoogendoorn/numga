"""The Koebe polyhedron of an irregular triangulation of the sphere, centred, and carried by boosts."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull

from examples.surfaces.koebe import core

# The number of points, how far their directions are nudged off an even spread, and its seed; the spread and
# seed of the random distances whose hull gives the triangulation, every point on it.
POINTS = 32
JITTER = 0.03
DIRECTION_SEED = 0
MESH_SPREAD, MESH_SEED = 0.06, 57
# The boosts the centred polyhedron is carried through: their largest rapidity, the tilt of the direction
# they turn about the z axis through, the frames of one turn, and the duration of each.
RAPIDITY = 0.6
TILT = 0.4
FRAMES = 48
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def triangulation() -> tuple[core.Vector, np.ndarray, np.ndarray]:
    """Directions spread over the sphere along a Fibonacci spiral, each nudged a little at random, and the
    faces and edges of the hull of the points at random distances along them: a triangulation of the sphere
    with triangles of uneven shape."""
    turns = np.arange(POINTS) + 0.5
    polar = np.arccos(1 - 2 * turns / POINTS)
    azimuth = np.pi * (1 + 5**0.5) * turns
    spiral = np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=-1)   # [points, 3]
    nudged = spiral + JITTER * np.random.default_rng(DIRECTION_SEED).normal(size=(POINTS, 3))
    directions = core.mv("x y z", nudged / np.linalg.norm(nudged, axis=-1, keepdims=True))   # [points] Vector
    faces = hull(directions * np.exp(MESH_SPREAD * np.random.default_rng(MESH_SEED).normal(size=POINTS)))
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=-1), axis=0)
    return directions, faces, edges


def centred() -> tuple[core.Vector, np.ndarray, np.ndarray]:
    """The circles of the Koebe polyhedron of the triangulation, from caps around the directions half as
    wide as the mean angle to their neighbours, and centred; with the faces and edges."""
    directions, faces, edges = triangulation()
    angles = np.arccos(np.clip(scalars(directions[edges[:, 0]] | directions[edges[:, 1]]), -1, 1))   # [edges]
    radii = np.bincount(edges.reshape(-1), np.repeat(angles, 2)) / np.bincount(edges.reshape(-1)) / 2   # [points]
    circles = koebe(core.caps(directions, radii), edges)                       # [points] Vector
    return core.centring(centre(core.touching(circles, edges))) >> circles, faces, edges


def boosted() -> tuple[core.Vector, np.ndarray, np.ndarray]:
    """The centred circles carried by boosts whose direction turns once about the z axis, tilted towards it,
    with a rapidity rising and falling twice per turn."""
    circles, faces, edges = centred()
    turn = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)                   # [frames]
    directions = core.mv("x y z", np.stack([np.cos(turn), np.sin(turn), np.full_like(turn, TILT)], axis=-1)).normalized()   # [frames] Vector
    rotors = core.boost(directions, RAPIDITY * np.sin(turn) ** 2)              # [frames] Rotor
    return rotors[:, None] >> circles, faces, edges                            # [frames, points] Vector


# --- the steps outside the algebra -----------------------------------------------------
def koebe(start: core.Vector, edges: np.ndarray) -> core.Vector:
    """The circles nearest the start, by a least-squares solve, that square to one and touch from outside
    along every edge: inner product minus one."""
    gram = scalars(core.basis()[:, None] | core.basis()[None, :])              # [4, 4] the metric
    count = start.shape[-1]

    def residual(flat: np.ndarray) -> np.ndarray:
        c = flat.reshape(count, 4)
        return np.concatenate([np.einsum("ni,ij,nj->n", c, gram, c) - 1,
                               np.einsum("ei,ij,ej->e", c[edges[:, 0]], gram, c[edges[:, 1]]) + 1])

    def jacobian(flat: np.ndarray) -> np.ndarray:
        c = flat.reshape(count, 4)
        rows = np.zeros((count + len(edges), count, 4))
        rows[np.arange(count), np.arange(count)] = 2 * c @ gram
        rows[count + np.arange(len(edges)), edges[:, 0]] = c[edges[:, 1]] @ gram
        rows[count + np.arange(len(edges)), edges[:, 1]] = c[edges[:, 0]] @ gram
        return rows.reshape(len(rows), -1)

    solved = least_squares(residual, components(start).reshape(-1), jac=jacobian, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15).x
    return core.mv("x y z w", solved.reshape(count, 4))                        # [vertices] Vector


def centre(points: core.Vector) -> core.Vector:
    """The unit timelike vector that minimises the summed logarithms of minus its inner products with the
    null vectors of points of the sphere: the centre seen from which they balance."""
    null = components(points)                                                  # [points, 4]
    gram = scalars(core.basis()[:, None] | core.basis()[None, :])              # [4, 4] the metric

    def timelike(spatial: np.ndarray) -> np.ndarray:
        return np.concatenate([spatial, [np.sqrt(1 + spatial @ spatial)]])

    spatial = minimize(lambda spatial: np.log(-(null @ gram @ timelike(spatial))).sum(), np.zeros(3), method="BFGS").x
    return core.mv("x y z w", timelike(spatial))                               # [] Vector


def hull(points: core.Vector) -> np.ndarray:
    """The faces of the convex hull of points around the centre, each turned so that the volume of its
    vertices turns as x, y, z do: counter-clockwise from outside."""
    faces = ConvexHull(components(points)[..., :3]).simplices                 # [faces, 3]
    a, b, c = (points[faces[:, corner]] for corner in range(3))               # [faces] Vector each
    turning = scalars((a ^ b ^ c) * core.mv.xyz.reverse()) > 0                 # [faces]
    return np.where(turning[:, None], faces, faces[:, [0, 2, 1]])


def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for the solves."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def components(values: core.Vector) -> np.ndarray:
    """x, y, z and w of a batch of vectors, for the solves."""
    return values.cast(core.ga.subspace("x y z w")).kernel


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.surfaces.koebe import render

    circles, faces, edges = centred()
    save_figure(render.draw(circles, faces, edges), "koebe")
    frames, faces, edges = boosted()
    save_animation(render.animate(frames, faces, edges), "koebe_boost", DURATION_MS)

    # checks
    # Every circle squares to one, and the circles at the ends of every edge touch from outside.
    assert np.abs(scalars(frames | frames) - 1).max() < 1e-10
    assert np.abs(scalars(frames[:, edges[:, 0]] | frames[:, edges[:, 1]]) + 1).max() < 1e-10
    # Every face circle is orthogonal to its vertices' circles, and passes through their touching points.
    duals = core.face_circles(frames, faces)                                   # [frames, faces] Vector
    assert np.abs(scalars(duals[..., None] | frames[:, faces]) / scalars(duals | duals)[..., None] ** 0.5).max() < 1e-10
    # Every edge of the polyhedron touches the sphere: its line's distance from the centre is one.
    assert np.abs(render.edge_distances(core.apexes(frames), edges) - 1).max() < 1e-10
    # Centred, the touching points balance about the centre.
    assert np.linalg.norm(render.coordinates(core.touching(circles, edges)).mean(axis=0)) < 1e-4


if __name__ == "__main__":
    main()
