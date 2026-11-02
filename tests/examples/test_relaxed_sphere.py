"""Round meshes of the two-sphere and the three-sphere relaxed under icosahedral symmetry: random seeds on the
two-sphere relax into a geodesic grid, its smallest angle near the ceiling its five-fold vertices set, and random
orbits on the three-sphere relax from slivers into round tetrahedra."""

import numpy as np

from numga import Algebra
from numga.algebras import VGA3D
from examples import instantiate

two_sphere = instantiate("examples.util.relaxed_sphere", VGA3D)
three_sphere = instantiate("examples.util.relaxed_sphere", Algebra("x+y+z+w+"))

# The free seeds on the two-sphere, and the orbits of free seeds on the three-sphere; their random start; how far each
# step moves a seed, and the steps enough for the relaxations to settle.
n_seeds = 3
n_orbits = 5
rng_seed = 7
pull = 0.5
n_steps = 20


def smallest_angle(core, mesh, points) -> float:
    """The smallest angle between two facets of any simplex, in degrees."""
    return np.degrees(np.arccos(core.facet_cosines(mesh, points).to_array().max()))


def test_random_seeds_relax_into_a_geodesic_grid():
    """Only the twelve points on the five-fold axes meet other than six faces, and the smallest angle comes within a
    few degrees of the pentagons' ceiling, about fifty-five degrees."""
    mv = two_sphere.mv
    planes = two_sphere.mirrors()                                               # [3] Plane+
    turns = two_sphere.icosahedral_turns()                                      # [rotations] Rotor+
    # The line the first and the second mirror share, a five-fold axis: as a point, its sense set by the handedness of
    # space, the join of the three axes' directions.
    handedness = mv.x.dual() & mv.y.dual() & mv.z.dual()                        # [] Scalar-
    axis = ((planes[0] ^ planes[1]) * handedness).normalized()                  # [] Point-
    anchors = two_sphere.unique_points(turns >> axis)                           # [axis points] Point-
    seeds = mv.vector(np.random.default_rng(rng_seed).normal(size=(n_seeds, 3))).dual().normalized()   # [seeds] Point-
    mesh, points = list(two_sphere.relaxed(turns, anchors, seeds, n_steps, pull))[-1]
    irregular = (np.round(two_sphere.valences(mesh).to_array()) != 6).sum()
    assert irregular == 12 and smallest_angle(two_sphere, mesh, points) > 50


def test_random_orbits_relax_from_slivers_into_round_tetrahedra():
    """The 600-cell's vertices and a few random orbits under the binary icosahedral group start with slivers, their
    smallest dihedral angle a few degrees, and relax to tetrahedra whose smallest dihedral angle exceeds thirty."""
    turns = three_sphere.binary_icosahedral_turns()                             # [rotations] Rotor+
    anchors = turns >> three_sphere.mv.w.dual()                                 # [rotations] Point-
    coordinates = np.random.default_rng(rng_seed).normal(size=(n_orbits, 4))
    seeds = three_sphere.mv.vector(coordinates).dual().normalized()             # [orbits] Point-
    steps = list(three_sphere.relaxed(turns, anchors, seeds, n_steps, pull))
    (start_mesh, start_points), (mesh, points) = steps[0], steps[-1]
    assert smallest_angle(three_sphere, start_mesh, start_points) < 5 and smallest_angle(three_sphere, mesh, points) > 30
