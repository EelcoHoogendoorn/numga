"""The Koebe polyhedron of an irregular triangulation of the sphere, centred, and carried by boosts."""

from __future__ import annotations

import numpy as np

from examples.surfaces.koebe import core

# The number of points, how far their directions are nudged off an even spread, and its seed; the spread and
# seed of the random distances whose hull gives the triangulation, every point on it.
N_POINTS = 32
JITTER = 0.03
DIRECTION_SEED = 0
MESH_SPREAD, MESH_SEED = 0.06, 57
# The Gauss-Newton steps of the Koebe solve, and the iterations finding the centre.
N_NEWTON_STEPS = 6
N_CENTRE_ITERATIONS = 8
# The boosts the centred polyhedron is carried through: their largest rapidity, the tilt of the direction
# they turn about the z axis through, the frames of one turn, and the duration of each.
RAPIDITY = 0.6
TILT = 0.4
N_FRAMES = 48
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def triangulation() -> tuple[core.Vector, np.ndarray, np.ndarray]:
    """Directions spread over the sphere along a Fibonacci spiral, each nudged a little at random, and the
    faces and edges of the hull of the points at random distances along them: a triangulation of the sphere
    with triangles of uneven shape."""
    nudges = core.mv(core.Vector.derive.within("x y z"), np.random.default_rng(DIRECTION_SEED).normal(size=(N_POINTS, 3)))
    directions = (core.spiral(N_POINTS) + nudges * JITTER).normalized()        # [points] Vector
    faces = core.hull(directions * np.exp(MESH_SPREAD * np.random.default_rng(MESH_SEED).normal(size=N_POINTS)))
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=-1), axis=0)
    return directions, faces, edges


def centred() -> tuple[core.Vector, np.ndarray, np.ndarray]:
    """The circles of the Koebe polyhedron of the triangulation, from caps around the directions half as
    wide as the mean angle across an edge, and centred; with the faces and edges."""
    directions, faces, edges = triangulation()
    # The angle across each edge, the size of the turn from one end's direction to the other's.
    angles = (directions[edges[:, 1]] * directions[edges[:, 0]]).log().norm()  # [edges] Scalar
    circles = core.koebe(core.caps(directions, angles.mean() / 2), edges, N_NEWTON_STEPS)   # [points] Vector
    return core.centring(core.centre(core.touching(circles, edges), N_CENTRE_ITERATIONS)) >> circles, faces, edges


def boosted() -> tuple[core.Vector, np.ndarray, np.ndarray]:
    """The centred circles carried by boosts whose direction turns once about the z axis, tilted towards it,
    with a rapidity rising and falling twice per turn."""
    circles, faces, edges = centred()
    turn_angles = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)          # [frames]
    directions = ((core.mv.xy * (-turn_angles / 2)).exp() >> (core.mv.x + core.mv.z * TILT)).normalized()   # [frames] Vector
    rotors = core.boost(directions, RAPIDITY * np.sin(turn_angles) ** 2)       # [frames] Rotor
    return rotors[:, None] >> circles, faces, edges                            # [frames, points] Vector


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
    assert np.abs((frames | frames).to_array() - 1).max() < 1e-10
    assert np.abs((frames[:, edges[:, 0]] | frames[:, edges[:, 1]]).to_array() + 1).max() < 1e-10
    # Every face circle is orthogonal to its vertices' circles, and passes through their touching points.
    duals = core.face_circles(frames, faces)                                   # [frames, faces] Vector
    assert np.abs((duals[..., None] | frames[:, faces]).to_array() / (duals | duals).to_array()[..., None] ** 0.5).max() < 1e-10
    # Every edge of the polyhedron touches the sphere: its line's distance from the centre is one.
    assert np.abs(render.edge_distances(core.apexes(frames), edges) - 1).max() < 1e-10
    # Centred, the touching points balance about the centre.
    assert np.linalg.norm(render.coordinates(core.touching(circles, edges)).mean(axis=0)) < 1e-4


if __name__ == "__main__":
    main()
