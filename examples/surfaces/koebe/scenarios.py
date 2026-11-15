"""Irregular sphere triangulation, centred Koebe polyhedron, and Möbius boost orbit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from numga import stack
from examples.surfaces.koebe import core
from examples.util import shapes

# Vertex count, angular jitter and seed on the Fibonacci spiral, and radial spread and seed for the convex hull.
n_points = 32
jitter = 0.03
direction_seed = 0
radius_spread, radius_seed = 0.06, 57
# Gauss-Newton iterations for the circle tangency solve and fixed-point steps for the balance centre.
n_newton_steps = 6
n_centre_iterations = 8
# Peak boost rapidity, tilt of the boost plane from x toward z, frame count per turn, and frame duration in ms.
rapidity = 0.6
tilt = 0.4
n_frames = 96
duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def sphere_triangulation() -> tuple[core.Topology, np.ndarray]:
    """Jittered Fibonacci-spiral directions (`[V, 3]`) scaled by random radii define an irregular convex-hull `Topology`."""
    turns = np.arange(n_points) + 0.5                                       # [n_points]
    polar = np.arccos(1 - 2 * turns / n_points)                             # [n_points]
    azimuth = np.pi * (1 + 5**0.5) * turns                                  # [n_points]
    across, along, up = shapes.planes(3)
    tilts = ((up ^ across) * (-polar / 2)).exp()                            # [n_points] Rotor+
    spiral = ((across ^ along) * (-azimuth / 2)).exp() >> (tilts >> up.dual())  # [n_points] Point-
    axes = stack((along ^ up, up ^ across, across ^ along))                 # [3] Bivector+
    weights = np.random.default_rng(direction_seed).normal(size=(n_points, 3))  # [n_points, 3]
    nudges = ((axes * weights).sum(axis=-1) * (jitter / 2)).exp()           # [n_points] Rotor+
    direction_coords = shapes.read(nudges >> spiral, shapes.planes(3))      # [n_points, 3]
    radii = np.exp(radius_spread * np.random.default_rng(radius_seed).normal(size=n_points))  # [n_points]
    topology, _ = shapes.convex_boundary(direction_coords * radii[:, None])
    return topology, direction_coords


def centred_polyhedron() -> tuple[core.Topology, core.Plane]:
    """Solves and centres the Koebe circle pattern (`Plane+[V]`) on `sphere_triangulation()`."""
    topology, direction_coords = sphere_triangulation()
    directions = (stack((core.mv.x, core.mv.y, core.mv.z)) * direction_coords).sum(axis=-1).dual().field()  # Point-[V]
    circles = core.koebe_circles(topology, core.cap_circles(topology, directions), n_newton_steps)  # Plane+[V]
    centre = core.balance_centre(core.tangency_points(topology, circles), n_centre_iterations)  # [] Point-
    centring = (1 - core.origin * centre).normalized()                      # [] Rotor+
    return topology, centring >> circles


def boost_bounds(circles: core.Plane) -> core.Point:
    """Tangent-cone apexes (`[12] Point-[V]`) sampled around the boost orbit fix a stationary camera viewport."""
    return core.cone_apexes(core.boost_rotors(rapidity, tilt, 12) >> circles)


def line_distances(incidence: core.Sparse, apexes: core.Point) -> core.Scalar:
    """Euclidean distances from `core.origin` to the lines joining incident endpoint pairs in `apexes` (`[...] Scalar+[E]`)."""
    places = incidence * apexes[..., None]                                  # [..., 2] Point-[E]
    lines = places[..., 0] & places[..., 1]                                 # [...] Antibivector-[E]
    return ((lines | core.mv.w).squared() / (lines ^ core.mv.w).squared()).square_root()


def save_states(states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments,
                name: str, duration_ms: int) -> list:
    """Streams `states` once through `animate(*arguments, ...)` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return kept


def main() -> None:
    from examples.util.animation import save_figure
    from examples.surfaces.koebe import render

    topology, circles = centred_polyhedron()
    save_figure(render.polyhedron_figure(topology, circles), "koebe")

    view = render.polyhedron_view(boost_bounds(circles), render.n_animation_pixels)
    recorded = save_states(
        core.boost_states(circles, rapidity, tilt, n_frames),
        render.boost_animation, topology, view,
        name="koebe_boost", duration_ms=duration_ms,
    )
    frames = stack(recorded)                                                # [n_frames] Plane+[V]

    # checks
    # Every vertex circle squares to one and touches its edge neighbours externally across the entire boost cycle.
    assert ((frames.squared() - 1).abs() < 1e-10).all()
    ends = topology.incidence[1, 0] * frames[..., None]                     # [n_frames, 2] Plane+[E]
    assert (((ends[..., 0] | ends[..., 1]) + 1).abs() < 1e-10).all()
    # Every face circle is orthogonal to its three vertex circles.
    duals = core.face_circles(topology, frames)                             # [n_frames] Patch-[F]
    corners = topology.incidence[2, 0] * frames[..., None]                  # [n_frames, 3] Plane+[F]
    assert (((duals[..., None] | corners).squared() / duals.squared()[..., None]).abs() < 1e-20).all()
    # Every primal and dual polyhedron edge lies at unit distance from the sphere centre.
    assert ((line_distances(topology.incidence[1, 0], core.cone_apexes(frames)) - 1).abs() < 1e-10).all()
    assert ((line_distances(topology.dual.incidence[1, 0], core.cone_apexes(duals)) - 1).abs() < 1e-10).all()
    # Centring places the unit-weight mean of the tangency points at the centre of the sphere.
    touches = core.tangency_points(topology, circles)                       # Point-[E]
    balance = (touches / (core.mv.w & touches)).sites.mean()                # [] Point-
    assert ((stack((core.mv.x, core.mv.y, core.mv.z)) & balance).abs() < 1e-4).all()


if __name__ == "__main__":
    main()
