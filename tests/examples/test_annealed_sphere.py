"""Round meshes of spheres relaxed under icosahedral symmetry: random seeds on the two-sphere relax into a geodesic
grid, its smallest angle near the ceiling its five-fold vertices set, and the 600-cell's edge midpoints relax into
tetrahedra without slivers."""

import numpy as np

from numga import Algebra
from numga.algebras import VGA3D
from examples import instantiate

two_sphere = instantiate("examples.util.annealed_sphere", VGA3D)
three_sphere = instantiate("examples.util.annealed_sphere", Algebra("x+y+z+w+"))

# The free seeds on the two-sphere and their random start; how far each step turns a seed, and the steps enough for
# the relaxations to settle; the 600-cell's edge midpoints, each nudged off its symmetric place by about this much, so
# that no five points share a small sphere.
n_seeds = 3
seed = 7
pull = 0.5
n_steps = 10
nudge = 0.02


def two_sphere_scene():
    """The icosahedron's rotations, the twelve points on its five-fold axes, and the random seeds."""
    mv = two_sphere.mv
    planes = two_sphere.mirrors()                                               # [3] Plane+
    turns = two_sphere.icosahedral_turns()                                      # [rotations] Rotor+
    # The line the first and the second mirror share, a five-fold axis: as a point, its sense set by the handedness of
    # space, the join of the three axes' directions.
    handedness = mv.x.dual() & mv.y.dual() & mv.z.dual()                        # [] Scalar-
    axis = ((planes[0] ^ planes[1]) * handedness).normalized()                  # [] Point-
    anchors = two_sphere.unique_points(turns >> axis)                           # [axis points] Point-
    seeds = mv.vector(np.random.default_rng(seed).normal(size=(n_seeds, 3))).dual().normalized()   # [seeds] Point-
    return turns, anchors, seeds


def edge_midpoints():
    """The binary icosahedral group's turns, the 600-cell's vertices, their orbit of the origin, and one of each orbit
    of its edges' midpoints, moved out onto the sphere and nudged."""
    turns = three_sphere.binary_icosahedral_turns()                             # [rotations] Rotor+
    cell_vertices = turns >> three_sphere.mv.w.dual()                           # [vertices] Point-
    vertices = cell_vertices.field()                                            # Point-[vertices]
    mesh = three_sphere.hull(vertices)
    midpoints = (mesh.incidence[1, 0] * vertices).sum(axis=0).normalized()     # Point-[edges]
    seeds = three_sphere.orbit_representatives(turns, midpoints.batch())       # [orbits] Point-
    offsets = three_sphere.mv.vector(np.random.default_rng(seed).normal(size=(seeds.shape[0], 4))).dual().normalized()
    return turns, cell_vertices, (seeds + nudge * offsets).normalized()


def test_random_seeds_relax_into_a_geodesic_grid():
    """Only the twelve points on the five-fold axes meet other than six faces, and the smallest angle comes within a
    few degrees of the pentagons' ceiling, about fifty-five degrees."""
    turns, anchors, seeds = two_sphere_scene()
    mesh, points, _ = list(two_sphere.relaxed(turns, anchors, seeds, n_steps, pull))[-1]
    irregular = (np.round(two_sphere.valences(mesh).to_array()) != 6).sum()
    smallest = np.degrees(np.arccos(two_sphere.facet_cosines(mesh, points).to_array().max()))
    assert irregular == 12 and smallest > 50


def test_edge_midpoints_relax_without_slivers():
    """The 600-cell with its edges' midpoints, relaxed under the binary icosahedral group, has no tetrahedron with a
    dihedral angle below forty degrees."""
    turns, cell_vertices, seeds = edge_midpoints()
    mesh, points, _ = list(three_sphere.relaxed(turns, cell_vertices, seeds, n_steps, pull))[-1]
    smallest = np.degrees(np.arccos(three_sphere.facet_cosines(mesh, points).to_array().max()))
    assert smallest > 40
