"""Round meshes of the two-sphere and the three-sphere relaxed under icosahedral symmetry: one seed on the two-sphere
and one orbit on the three-sphere relaxed for a step, and their meshes' facet angles and valences read."""

import numpy as np

from numga import Algebra
from numga.algebras import VGA3D
from examples import instantiate

two_sphere = instantiate("examples.util.relaxed_sphere", VGA3D)
three_sphere = instantiate("examples.util.relaxed_sphere", Algebra("x+y+z+w+"))

# The free seed's random start, and how far each step moves it.
rng_seed = 7
pull = 0.5


def test_relaxations_run():
    mv = two_sphere.mv
    planes = two_sphere.mirrors()                                               # [3] Plane+
    turns = two_sphere.icosahedral_turns()                                      # [rotations] Rotor+
    # The line the first and the second mirror share, a five-fold axis: as a point, its sense set by the handedness of
    # space, the join of the three axes' directions.
    handedness = mv.x.dual() & mv.y.dual() & mv.z.dual()                        # [] Scalar-
    axis = ((planes[0] ^ planes[1]) * handedness).normalized()                  # [] Point-
    anchors = two_sphere.unique_points(turns >> axis)                           # [axis points] Point-
    seeds = mv.vector(np.random.default_rng(rng_seed).normal(size=(1, 3))).dual().normalized()   # [seeds] Point-
    *_, (mesh, points) = two_sphere.relaxed(turns, anchors, seeds, 2, pull)
    two_sphere.facet_cosines(mesh, points)
    two_sphere.valences(mesh)

    turns = three_sphere.binary_icosahedral_turns()                             # [rotations] Rotor+
    anchors = turns >> three_sphere.mv.w.dual()                                 # [rotations] Point-
    seeds = three_sphere.mv.vector(np.random.default_rng(rng_seed).normal(size=(1, 4))).dual().normalized()   # [orbits] Point-
    *_, (mesh, points) = three_sphere.relaxed(turns, anchors, seeds, 2, pull)
    three_sphere.facet_cosines(mesh, points)
