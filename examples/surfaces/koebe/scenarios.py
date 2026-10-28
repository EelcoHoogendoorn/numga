"""The Koebe polyhedron of an irregular triangulation of the sphere, centred, and carried by boosts."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.surfaces.koebe import core

# The number of points, how far their directions are turned off an even spread, and its seed; the spread and
# seed of the random distances whose hull gives the triangulation, every point on it.
N_POINTS = 32
JITTER = 0.03
DIRECTION_SEED = 0
MESH_SPREAD, MESH_SEED = 0.06, 57
# The Gauss-Newton steps of the Koebe solve, and the iterations finding the centre.
N_NEWTON_STEPS = 6
N_CENTRE_ITERATIONS = 8
# The boosts the centred polyhedron is carried through: their largest rapidity, the angle by which their plane
# through w is tilted from x towards the z axis it turns about, the frames of one turn, and the duration of each.
RAPIDITY = 0.6
TILT = 0.4
N_FRAMES = 48
FRAME_DURATION = 80


# --- math -----------------------------------------------------------------------------
def triangulation() -> tuple[core.Point, np.ndarray, np.ndarray]:
    """The triangulation is the directions along a Fibonacci spiral, each turned a little at random, with the faces
    and edges of the hull of the points at random distances along them, triangles of uneven shape."""
    coordinate_planes = stack((core.mv.yz, core.mv.zx, core.mv.xy))           # [3] Bivector+
    nudge_weights = np.random.default_rng(DIRECTION_SEED).normal(size=(N_POINTS, 3))   # [points, 3]
    nudge_planes = (coordinate_planes * nudge_weights).sum(axis=-1)            # [points] Bivector+
    nudges = (nudge_planes * (JITTER / 2)).exp()                               # [points] Rotor+
    directions = nudges >> core.spiral(N_POINTS)                               # [points] Point-
    faces = core.hull(directions * np.exp(MESH_SPREAD * np.random.default_rng(MESH_SEED).normal(size=N_POINTS)))
    edges = core.Mesh(N_POINTS, faces).edges                                   # [edges, 2]
    return directions, faces, edges


def centred() -> tuple[core.Vector, np.ndarray, np.ndarray]:
    """The centred circles of the Koebe polyhedron of the triangulation come with its faces and edges."""
    directions, faces, edges = triangulation()
    # The angle across each edge, the size of the turn from one end's direction to the other's.
    angles = (directions[edges[:, 1]] * directions[edges[:, 0]]).log().norm()  # [edges] Scalar+
    # Caps around the directions, half as wide as the mean angle across an edge: each the polar of the apex of the
    # cone touching the sphere along it, out along its direction at the reciprocal of the cosine, scaled so that
    # the circle squares to one.
    radius = angles.mean() / 2                                                 # [] Scalar+
    cap_apexes = (directions + core.origin * radius.cos()) / radius.sin()      # [points] Point-
    start = cap_apexes.dual()                                                  # [points] Vector+
    circles = core.koebe(start, edges, N_NEWTON_STEPS)                         # [points] Vector+
    centre = core.centre(core.touching(circles, edges), N_CENTRE_ITERATIONS)   # [] Point-
    # The boost along the line through both centres carries the balancing centre onto the centre of the sphere.
    centring = (1 - core.origin * centre).normalized()                         # [] Rotor+
    return centring >> circles, faces, edges


def boosted() -> tuple[core.Vector, np.ndarray, np.ndarray]:
    """The centred circles are carried by boosts whose plane through w turns once about the z axis, tilted
    towards it, with a rapidity rising and falling twice per turn."""
    circles, faces, edges = centred()
    turn_angles = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)          # [frames]
    tilted_plane = (core.mv.zx * (TILT / 2)).exp() >> core.mv.xw              # [] Bivector+
    boost_planes = (core.mv.xy * (-turn_angles / 2)).exp() >> tilted_plane     # [frames] Bivector+
    rotors = (boost_planes * (RAPIDITY * np.sin(turn_angles) ** 2 / 2)).exp()  # [frames] Rotor+
    return rotors[:, None] >> circles, faces, edges                            # [frames, points] Vector+


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.koebe import render

    circles, faces, edges = centred()
    save_figure(render.draw(circles, faces, edges), "koebe")
    frames, faces, edges = boosted()
    save_animation(render.animate(frames, faces, edges), "koebe_boost", FRAME_DURATION)

    # checks
    # Every circle squares to one, and the circles at the ends of every edge touch from outside.
    assert np.abs((frames | frames).to_array() - 1).max() < 1e-10
    assert np.abs((frames[:, edges[:, 0]] | frames[:, edges[:, 1]]).to_array() + 1).max() < 1e-10
    # Every face circle is orthogonal to its vertices' circles, and passes through their touching points.
    duals = core.face_circles(frames, faces)                                   # [frames, faces] Vector-
    assert np.abs((duals[..., None] | frames[:, faces]).to_array() / (duals | duals).to_array()[..., None] ** 0.5).max() < 1e-10
    # Every edge of the polyhedron touches the sphere: its line's distance from the centre is one, the line's
    # moment about the centre, its part paired with w, over its direction, its part wedged with w.
    apexes = core.apexes(frames)                                               # [frames, points] Point-
    lines = apexes[..., edges[:, 0]] & apexes[..., edges[:, 1]]                # [frames, edges] Bivector-
    distances = ((lines | core.mv.w).squared() / (lines ^ core.mv.w).squared()).square_root()   # [frames, edges] Scalar+
    assert np.abs(distances.to_array() - 1).max() < 1e-10
    # Centred, the touching points balance about the centre of the sphere: at unit weight they average to a point
    # on the planes x, y and z through it.
    touches = core.touching(circles, edges)                                    # [edges] Point-
    balance = (touches / (core.mv.w & touches)).mean()                         # [] Point-
    assert np.abs((stack((core.mv.x, core.mv.y, core.mv.z)) & balance).to_array()).max() < 1e-4


if __name__ == "__main__":
    main()
