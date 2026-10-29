"""A jittered grid lifted onto the paraboloid, and the same grid with heights breathing about it."""

from __future__ import annotations

import numpy as np

from examples.surfaces.reciprocal import core

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
def grid() -> tuple[core.Mesh, core.Point, np.ndarray]:
    """The grid's mesh and points come with their heights on the paraboloid, half their squared distance from the
    centre."""
    across = np.linspace(-1, 1, N_SIDE)
    coords = np.stack(np.meshgrid(across, across, indexing="ij"), axis=-1).reshape(-1, 2)   # [vertices, 2]
    inside = (np.abs(coords) < 1).all(axis=-1)
    coords[inside] += JITTER * np.random.default_rng(GRID_SEED).normal(size=(inside.sum(), 2))
    points = core.mv.antivector(np.concatenate([coords, np.zeros((len(coords), 1))], axis=-1)).field()   # Point-[vertices]
    return core.Mesh.triangulate(points), points, (coords**2).sum(axis=-1) / 2   # [vertices]


def breathing() -> tuple[core.Mesh, core.Point, np.ndarray]:
    """The grid's mesh and points come with their heights at every frame of one breath about the paraboloid, each
    with its own phase."""
    mesh, points, heights = grid()
    phases = np.random.default_rng(PHASE_SEED).uniform(0, 2 * np.pi, len(heights))
    times = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)[:, None]
    return mesh, points, heights + AMPLITUDE * np.sin(times + phases)                  # [frames, vertices]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.reciprocal import render

    mesh, points, heights = grid()
    save_figure(render.draw(mesh, points, heights, core.stresses(mesh, points, heights),
                            (core.slopes(mesh, points, heights) ^ core.mv.z) * core.mv.z), "reciprocal")
    mesh, points, breaths = breathing()
    frames = ((breath, core.stresses(mesh, points, breath), (core.slopes(mesh, points, breath) ^ core.mv.z) * core.mv.z) for breath in breaths)
    save_animation(render.animate(mesh, points, frames), "reciprocal_breath", DURATION_MS)

    # checks
    inner, (left, right) = mesh.interior_edges, mesh.edge_faces.T
    stresses, folds = core.stresses(mesh, points, breaths), core.folds(mesh, points, breaths)   # [frames] Scalar+[inner], Scalar+[inner]
    # Every bar is in tension exactly where the terrain bends convex, and some bars go into compression.
    assert np.array_equal(stresses > 0, folds > 0) and (stresses < 0).any()
    # Every reciprocal edge is perpendicular to its bar, and every interior vertex is in equilibrium.
    duals = (core.slopes(mesh, points, breaths) ^ core.mv.z) * core.mv.z                             # [frames, faces] Plane+
    bars = (mesh.d[0] * points).batch()[inner]                                 # [inner] Point-
    # The jump in slope pairs to nothing with its bar: both planes agree along it.
    assert np.abs(((duals[:, left] - duals[:, right]) & bars).to_array()).max() < 1e-10
    # The bars meeting at a vertex each pull it along themselves by their stress.
    pulls = (bars * core.stresses(mesh, points, breaths)).field()              # [frames] Point-[inner]
    tails, heads = (mesh.reading(mesh.edges[inner, end]) for end in range(2))  # Scalar+[inner <- vertices], Scalar+[inner <- vertices]
    net = (~tails * pulls - ~heads * pulls).batch()                            # [frames, vertices] Point-
    interior = np.setdiff1d(np.arange(mesh.n_vertices), mesh.edges[mesh.boundary_edges])
    assert (net[:, interior].norm() < 1e-10).all()
    # On the paraboloid the stresses are the cotangent weights and the reciprocal vertices the circumcentres.
    edges = mesh.boundary[2, 1] * (mesh.d[0] * points)                         # [3] Point-[faces]
    after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]                         # [3] Point-[faces], Point-[faces]
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()   # [3] Scalar+[faces]
    weights = (~mesh.incidence[2, 1] * (cotangents / 2)).sum(axis=0)           # Scalar+[edges]
    assert np.abs(core.stresses(mesh, points, heights).to_array() - weights.to_array()[inner]).max() < 1e-10
    centres = (core.slopes(mesh, points, heights) ^ core.mv.z) * core.mv.z                           # [faces] Plane+
    corners = (mesh.incidence[2, 0] * points).batch()                          # [3, faces] Point-
    # The slopes, read as points by the metric, lie as far from each of their face's corners.
    squared_distances = [(centres.dual() - corner).scalar_norm_squared().to_array() for corner in corners]
    assert np.abs(squared_distances[0] - squared_distances[1]).max() < 1e-10 and np.abs(squared_distances[0] - squared_distances[2]).max() < 1e-10


if __name__ == "__main__":
    main()
