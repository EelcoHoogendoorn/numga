"""Coverage and mathematical property tests for as-rigid-as-possible surface deformation."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.surfaces.arap import core, render, scenarios


def test_bending_deformation_and_rendering():
    """One coarse warm-started bending step preserves rest shape under identity targets, decreases energy, and renders."""
    topology, rest_vertices, handles, poses = scenarios.bending_setup(divisions=2, n_poses=1)
    (rest_result, _, _), = core.deformation_states(topology, rest_vertices, handles, [rest_vertices], scenarios.stiffness, 2)
    assert ((rest_result - rest_vertices).antinorm().batch() < 1e-9).all()

    states = list(core.deformation_states(topology, rest_vertices, handles, poses, scenarios.stiffness, 2))
    rigid_vertices, laplacian_vertices, energies = states[-1]
    assert ((energies[1:] - energies[:-1]) <= 1e-9 * energies[0]).all()

    frames = list(render.bar_animation(topology, rest_vertices, scenarios.tile_size, states))
    assert len(frames) == 1
    plt.close(render.bar_figure(topology, rest_vertices, [rigid_vertices, laplacian_vertices], scenarios.tile_size))
