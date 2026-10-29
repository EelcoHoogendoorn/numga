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
N_FRAMES = 96
FRAME_DURATION = 40


# --- math -----------------------------------------------------------------------------
def triangulation() -> tuple[core.Mesh, core.Point]:
    """The triangulation is the mesh of the hull of points at random distances along directions on a Fibonacci
    spiral, each turned a little at random, triangles of uneven shape, with the directions."""
    coordinate_planes = stack((core.mv.yz, core.mv.zx, core.mv.xy))           # [3] Bivector+
    nudge_weights = np.random.default_rng(DIRECTION_SEED).normal(size=(N_POINTS, 3))   # [points, 3]
    nudge_planes = (coordinate_planes * nudge_weights).sum(axis=-1)            # [points] Bivector+
    nudges = (nudge_planes * (JITTER / 2)).exp()                               # [points] Rotor+
    directions = (nudges >> core.spiral(N_POINTS)).field()                     # Point-[points]
    radii = np.exp(MESH_SPREAD * np.random.default_rng(MESH_SEED).normal(size=N_POINTS))   # [points]
    return core.hull((directions.batch() * radii).field()), directions


def centred() -> tuple[core.Mesh, core.Vector]:
    """The triangulation's mesh comes with the centred circles of its Koebe polyhedron."""
    mesh, directions = triangulation()
    # The angle across each edge, the size of the turn from one end's direction to the other's.
    tails, heads = mesh.incidence[1, 0] * directions                           # Point-[edges], Point-[edges]
    angles = (heads * tails).log().norm()                                      # Scalar+[edges]
    # Caps around the directions, half as wide as the mean angle across an edge: each the polar of the apex of the
    # cone touching the sphere along it, out along its direction at the reciprocal of the cosine, scaled so that
    # the circle squares to one.
    radius = angles.sites.mean() / 2                                           # [] Scalar+
    cap_apexes = (directions + core.origin * radius.cos()) / radius.sin()      # Point-[points]
    start = cap_apexes.dual()                                                  # Vector+[points]
    circles = core.koebe(mesh, start, N_NEWTON_STEPS)                          # Vector+[points]
    centre = core.centre(core.touching(mesh, circles), N_CENTRE_ITERATIONS)    # [] Point-
    # The boost along the line through both centres carries the balancing centre onto the centre of the sphere.
    centring = (1 - core.origin * centre).normalized()                         # [] Rotor+
    return mesh, centring >> circles


def boosted() -> tuple[core.Mesh, core.Vector]:
    """The centred circles are carried by boosts whose plane through w turns once about the z axis, tilted
    towards it, with a rapidity rising and falling twice per turn."""
    mesh, circles = centred()
    turn_angles = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)          # [frames]
    tilted_plane = (core.mv.zx * (TILT / 2)).exp() >> core.mv.xw              # [] Bivector+
    boost_planes = (core.mv.xy * (-turn_angles / 2)).exp() >> tilted_plane     # [frames] Bivector+
    rotors = (boost_planes * (RAPIDITY * np.sin(turn_angles) ** 2 / 2)).exp()  # [frames] Rotor+
    return mesh, rotors >> circles                                             # [frames] Vector+[points]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.koebe import render

    mesh, circles = centred()
    save_figure(render.draw(mesh, circles), "koebe")
    mesh, frames = boosted()
    save_animation(render.animate(mesh, frames), "koebe_boost", FRAME_DURATION)

    # checks
    # Every circle squares to one, and the circles at the ends of every edge touch from outside.
    assert np.abs((frames | frames).to_array() - 1).max() < 1e-10
    ends = mesh.incidence[1, 0] * frames[..., None]                            # [frames, ends] Vector+[edges]
    assert np.abs((ends[..., 0] | ends[..., 1]).to_array() + 1).max() < 1e-10
    # Every face circle is orthogonal to its vertices' circles, and passes through their touching points.
    duals = core.face_circles(mesh, frames)                                    # [frames] Vector-[faces]
    corners = mesh.incidence[2, 0] * frames[..., None]                         # [frames, corners] Vector+[faces]
    assert np.abs((duals[..., None] | corners).to_array() / (duals | duals).to_array()[..., None, :] ** 0.5).max() < 1e-10
    # Every edge of the polyhedron touches the sphere: its line's distance from the centre is one, the line's
    # moment about the centre, its part paired with w, over its direction, its part wedged with w.
    apex_ends = mesh.incidence[1, 0] * core.apexes(frames)[..., None]          # [frames, ends] Point-[edges]
    lines = apex_ends[..., 0] & apex_ends[..., 1]                              # [frames] Bivector-[edges]
    distances = ((lines | core.mv.w).squared() / (lines ^ core.mv.w).squared()).square_root()   # [frames] Scalar+[edges]
    assert np.abs(distances.to_array() - 1).max() < 1e-10
    # Centred, the touching points balance about the centre of the sphere: at unit weight they average to a point
    # on the planes x, y and z through it.
    touches = core.touching(mesh, circles)                                     # Point-[edges]
    balance = (touches / (core.mv.w & touches)).sites.mean()                   # [] Point-
    assert np.abs((stack((core.mv.x, core.mv.y, core.mv.z)) & balance).to_array()).max() < 1e-4


if __name__ == "__main__":
    main()
