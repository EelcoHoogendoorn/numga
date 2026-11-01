"""Round meshes of spheres relaxed under icosahedral symmetry: random seeds on the two-sphere relax into the mesh of the
chiral geodesic grid, its smallest angle near the ceiling its five-fold vertices set, and the 600-cell's edge
midpoints relax into tetrahedra without slivers."""

import numpy as np

from examples.surfaces.annealed_sphere import scenarios

# Steps enough for the relaxations to settle.
n_steps = 10


def test_random_seeds_relax_into_the_geodesic_grid():
    """Only the twelve points on the five-fold axes meet other than six faces, and the smallest angle comes within a
    few degrees of the pentagons' ceiling, about fifty-five degrees."""
    core = scenarios.two_sphere
    turns, anchors, seeds, _ = scenarios.two_sphere_scene()
    mesh, points, _ = list(core.relaxed(turns, anchors, seeds, n_steps, scenarios.pull))[-1]
    irregular = (np.round(core.valences(mesh).to_array()) != 6).sum()
    smallest = np.degrees(np.arccos(core.facet_cosines(mesh, points).to_array().max()))
    assert irregular == 12 and smallest > 50


def test_edge_midpoints_relax_without_slivers():
    """The 600-cell with its edges' midpoints, relaxed under the binary icosahedral group, has no tetrahedron with a
    dihedral angle below forty degrees."""
    core = scenarios.three_sphere
    group, cell_vertices = scenarios.three_sphere_scene()
    seeds = scenarios.cell_refinements(group, cell_vertices)["edge midpoints"]
    mesh, points, _ = list(core.relaxed(group, cell_vertices, seeds, n_steps, scenarios.pull))[-1]
    smallest = np.degrees(np.arccos(core.facet_cosines(mesh, points).to_array().max()))
    assert smallest > 40
