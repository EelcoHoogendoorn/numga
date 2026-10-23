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
N_FRAMES = 48
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def grid() -> tuple[core.Mesh, np.ndarray]:
    """The grid's mesh, and its points' heights on the paraboloid, half their squared distance from the
    centre."""
    across = np.linspace(-1, 1, N_SIDE)
    coords = np.stack(np.meshgrid(across, across, indexing="ij"), axis=-1).reshape(-1, 2)   # [vertices, 2]
    inside = (np.abs(coords) < 1).all(axis=-1)
    coords[inside] += JITTER * np.random.default_rng(GRID_SEED).normal(size=(inside.sum(), 2))
    mesh = core.Mesh._triangulate(core.planar(coords).field())
    return mesh, (coords**2).sum(axis=-1) / 2                                  # [vertices]


def breathing() -> tuple[core.Mesh, np.ndarray]:
    """The grid's mesh, and its heights at every frame of one breath about the paraboloid, each with its own
    phase."""
    mesh, heights = grid()
    phases = np.random.default_rng(PHASE_SEED).uniform(0, 2 * np.pi, len(heights))
    times = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)[:, None]
    return mesh, heights + AMPLITUDE * np.sin(times + phases)                  # [frames, vertices]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.surfaces.reciprocal import render

    mesh, heights = grid()
    save_figure(render.draw(mesh, heights, core.stresses(mesh, heights), core.reciprocal(mesh, heights)), "reciprocal")
    mesh, breaths = breathing()
    frames = ((breath, core.stresses(mesh, breath), core.reciprocal(mesh, breath)) for breath in breaths)
    save_animation(render.animate(mesh, frames), "reciprocal_breath", DURATION_MS)

    # checks
    inner, left, right = core.beside(mesh)
    stresses, folds = core.stresses(mesh, breaths).to_array(), core.folds(mesh, breaths).to_array()
    # Every bar is in tension exactly where the terrain bends convex, and some bars go into compression.
    assert np.array_equal(stresses > 0, folds > 0) and (stresses < 0).any()
    # Every reciprocal edge is perpendicular to its bar, and every interior vertex is in equilibrium.
    duals = core.reciprocal(mesh, breaths)                                     # [frames, faces] Vector
    bars = mesh.edge_displacements.batch()[inner]                              # [inner] Point
    # The jump in slope pairs to nothing with its bar: both planes agree along it.
    assert np.abs(((duals[:, left] - duals[:, right]) & bars).to_array()).max() < 1e-10
    assert render.unbalanced(mesh, core.stresses(mesh, breaths)) < 1e-10
    # On the paraboloid the stresses are the cotangent weights and the reciprocal vertices the circumcentres.
    assert np.abs(core.stresses(mesh, heights).to_array() - mesh.edge_ratio.to_array()[inner]).max() < 1e-10
    centres = core.reciprocal(mesh, heights)                                   # [faces] Vector
    corners = mesh.corners                                                     # [3] Vector[faces]
    # The slopes, read as points by the metric, lie as far from each of their face's corners.
    reach = [(centres.dual() - corner.batch()).scalar_norm_squared().to_array() for corner in corners]
    assert np.abs(reach[0] - reach[1]).max() < 1e-10 and np.abs(reach[0] - reach[2]).max() < 1e-10


if __name__ == "__main__":
    main()
