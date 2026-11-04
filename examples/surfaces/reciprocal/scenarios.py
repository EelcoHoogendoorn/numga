"""A jittered grid lifted onto the paraboloid, and the same grid with heights breathing about it."""

from __future__ import annotations

import numpy as np

from examples.surfaces.reciprocal import core
from examples.util import shapes

# The grid: its vertices per side across the square from -1 to 1, how far its interior vertices are nudged,
# and the seed of the nudges.
N_SIDE = 5
JITTER = 0.08
GRID_SEED = 11
# How far each height breathes about the paraboloid, the seed of the phases, the frames of one breath and
# the duration of each.
AMPLITUDE = 0.02
PHASE_SEED = 3
N_FRAMES = 96
DURATION_MS = 40


# --- math -----------------------------------------------------------------------------
def grid() -> tuple[core.Topology, core.Point, core.Scalar]:
    """The grid's mesh and points come with their heights on the paraboloid, half their squared distance from the
    centre."""
    across = np.linspace(-1, 1, N_SIDE)
    coords = np.stack(np.meshgrid(across, across, indexing="ij"), axis=-1).reshape(-1, 2)   # [vertices, 2]
    inside = (np.abs(coords) < 1).all(axis=-1)
    coords[inside] += JITTER * np.random.default_rng(GRID_SEED).normal(size=(inside.sum(), 2))
    topology, coords = shapes.delaunay(coords)
    points = core.mv.antivector(np.concatenate([coords, np.zeros((len(coords), 1))], axis=-1)).field()   # Point-[vertices]
    return topology, points, points.scalar_norm_squared() / 2  # Scalar+[vertices]


def breathing() -> tuple[core.Topology, core.Point, core.Scalar]:
    """The grid's mesh and points come with their heights at every frame of one breath about the paraboloid, each
    with its own phase."""
    topology, points, heights = grid()
    phases = np.random.default_rng(PHASE_SEED).uniform(0, 2 * np.pi, topology.n_vertices)
    times = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)[:, None]
    swings = core.mv.scalar((AMPLITUDE * np.sin(times + phases))[..., None]).field()   # [frames] Scalar+[vertices]
    return topology, points, heights + swings                                          # [frames] Scalar+[vertices]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.reciprocal import render

    topology, points, heights = grid()
    save_figure(render.draw(topology, points, heights, core.stresses(topology, points, heights),
                            (core.slopes(topology, points, heights) ^ core.mv.z) * core.mv.z), "reciprocal")
    topology, points, breaths = breathing()
    frames = ((breath, core.stresses(topology, points, breath), (core.slopes(topology, points, breath) ^ core.mv.z) * core.mv.z) for breath in breaths)
    save_animation(render.animate(topology, points, frames), "reciprocal_breath", DURATION_MS)

    # checks
    inner = topology.reading(topology.interior[1], 1)                          # Scalar+[inner <- edges]
    stresses, folds = core.stresses(topology, points, breaths), core.folds(topology, points, breaths)  # [frames] Scalar+[inner], Scalar+[inner]
    # Every bar is in tension exactly where the terrain bends convex, and some bars go into compression.
    assert np.array_equal(stresses > 0, folds > 0) and (stresses < 0).any()
    # Every reciprocal edge is perpendicular to its bar, and every interior vertex is in equilibrium.
    duals = (core.slopes(topology, points, breaths) ^ core.mv.z) * core.mv.z                         # [frames] Plane+[faces]
    bars = (inner * (topology.d[0] * points)).batch()                          # [inner] Point-
    # The jump in slope, the faces' planes summed back onto their edges as each face runs along it, pairs to
    # nothing with its bar: both planes agree along it.
    assert np.abs(((inner * (~topology.d[1] * duals)).batch() & bars).to_array()).max() < 1e-10
    # The bars meeting at a vertex each pull it along themselves by their stress, summed back onto their ends
    # against the way each bar runs.
    pulls = (bars * core.stresses(topology, points, breaths)).field()          # [frames] Point-[inner]
    bar_ends = inner * topology.d[0]                                           # Scalar+[inner <- vertices]
    net = -(~bar_ends * pulls)                                                 # [frames] Point-[vertices]
    assert ((topology.reading(topology.interior[0], 0) * net).norm() < 1e-10).all()
    # On the paraboloid the stresses are the cotangent weights and the reciprocal vertices the circumcentres.
    edges = topology.boundary[2, 1] * (topology.d[0] * points)                 # [3] Point-[faces]
    tails, heads = topology.corners[2, 1]
    after, before = edges[tails], edges[heads]                                 # [3] Point-[faces], Point-[faces]
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()   # [3] Scalar+[faces]
    weights = (~topology.incidence[2, 1] * (cotangents / 2)).sum(axis=0)       # Scalar+[edges]
    assert np.abs(core.stresses(topology, points, heights).to_array() - (inner * weights).to_array()).max() < 1e-10
    centres = (core.slopes(topology, points, heights) ^ core.mv.z) * core.mv.z                       # Plane+[faces]
    corners = topology.incidence[2, 0] * points                                # [3] Point-[faces]
    # The slopes, read as points by the metric, lie as far from each of their face's corners.
    squared_distances = [(centres.dual() - corner).scalar_norm_squared().to_array() for corner in corners]
    assert np.abs(squared_distances[0] - squared_distances[1]).max() < 1e-10 and np.abs(squared_distances[0] - squared_distances[2]).max() < 1e-10


if __name__ == "__main__":
    main()
