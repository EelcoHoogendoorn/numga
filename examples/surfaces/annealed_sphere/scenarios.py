"""Round meshes of the two-sphere and the three-sphere relaxed from random seeds under the icosahedron's rotations and
the binary icosahedral group: on the two-sphere twelve points on the axes and a few seeds in sixty copies each, on
the three-sphere the 600-cell's vertices and a few seeds in a hundred and twenty copies each; and the 600-cell's own
refinements, by the centres of its cells and the midpoints of its edges."""

from __future__ import annotations

import numpy as np

from numga import Algebra, stack
from numga.algebras import VGA3D
from examples import instantiate

two_sphere = instantiate("examples.surfaces.annealed_sphere.core", VGA3D)
three_sphere = instantiate("examples.surfaces.annealed_sphere.core", Algebra("x+y+z+w+"))

# The free seeds on the two-sphere, their random start, the steps of the relaxation and how far each step turns a
# seed.
n_seeds = 3
seed = 7
n_steps = 60
pull = 0.5
# The chiral geodesic grid of the icosahedron, its faces' sides three steps and then two after a sixth of a turn on
# the triangular lattice: nine points inside each face, three orbits of sixty, with the twelve on the axes as many
# points as the seeds give.
geodesic_steps = (3, 2)
# The 600-cell's own refinements, its cells' centres and its edges' midpoints, each nudged off its symmetric place
# by about this much, so that no five points share a small sphere.
nudge = 0.02


def two_sphere_scene() -> tuple[two_sphere.Rotor, two_sphere.Point, two_sphere.Point, two_sphere.Point]:
    """The icosahedron's rotations, the twelve points on its five-fold axes, the random seeds, and the seeds of the
    geodesic grid."""
    mv = two_sphere.mv
    planes = two_sphere.mirrors()                                               # [3] Plane+
    turns = two_sphere.icosahedral_turns()                                      # [rotations] Rotor+
    # The line the first and the second mirror share, which meet at a fifth of a half turn, a five-fold axis: as a
    # point, its sense set by the handedness of space, the join of the three axes' directions.
    handedness = mv.x.dual() & mv.y.dual() & mv.z.dual()                        # [] Scalar-
    axis = ((planes[0] ^ planes[1]) * handedness).normalized()                  # [] Point-
    anchors = two_sphere.unique_points(turns >> axis)                           # [axis points] Point-
    seeds = mv.vector(np.random.default_rng(seed).normal(size=(n_seeds, 3))).dual().normalized()   # [seeds] Point-
    return turns, anchors, seeds, geodesic_seeds(axis, planes[1] * planes[2])


def geodesic_seeds(axis: two_sphere.Point, third: two_sphere.Rotor) -> two_sphere.Point:
    """One point of each of the geodesic grid's orbits off the axes, `[orbits] Point-`: the points of the triangular
    lattice inside the face of the icosahedron that a third of a turn carries around its centre, placed by their
    barycentric weights on its corners and moved out onto the sphere; of each three the turn carries into each
    other, the one nearest the face's first corner."""
    steps = np.array(geodesic_steps)
    # The face's sides from its first corner on the lattice: the steps, and the steps a sixth of a turn on.
    sides = np.stack([steps, [-steps[1], steps.sum()]], axis=-1)               # [2, 2]
    lattice = np.stack(np.meshgrid(np.arange(-steps[1], steps[0] + 1), np.arange(steps.sum() + 1)), axis=-1)
    weights = np.linalg.solve(sides, lattice.reshape(-1, 2).T).T               # [lattice, 2]
    barycentric = np.concatenate([1 - weights.sum(axis=-1, keepdims=True), weights], axis=-1)   # [lattice, 3]
    inside = barycentric[(barycentric > 1e-9).all(axis=-1)]                     # [inside, 3]
    nearest_first = inside[inside.argmax(axis=-1) == 0]                         # [orbits, 3]
    corners = stack([axis, third >> axis, third >> (third >> axis)])            # [3] Point-
    return (corners * nearest_first).sum(axis=-1).normalized()


def three_sphere_scene() -> tuple[three_sphere.Rotor, three_sphere.Point]:
    """The binary icosahedral group's turns and the 600-cell's vertices, their orbit of the origin."""
    turns = three_sphere.binary_icosahedral_turns()                             # [rotations] Rotor+
    return turns, turns >> three_sphere.mv.w.dual()


def three_sphere_seeds(n_orbits: int, start: int = seed) -> three_sphere.Point:
    """So many random seeds on the three-sphere, `[orbits] Point-`, from the random start."""
    coordinates = np.random.default_rng(start).normal(size=(n_orbits, 4))
    return three_sphere.mv.vector(coordinates).dual().normalized()


def cell_refinements(group: three_sphere.Rotor, cell_vertices: three_sphere.Point) -> dict[str, three_sphere.Point]:
    """The seeds of the 600-cell's refinements `{name: [orbits] Point-}`, one of each orbit of its cells' centres
    and of its edges' midpoints, moved out onto the sphere and nudged."""
    vertices = cell_vertices.field()                                           # Point-[vertices]
    mesh = three_sphere.hull(vertices)
    midpoints = (mesh.incidence[1, 0] * vertices).sum(axis=0).normalized()     # Point-[edges]
    centres = three_sphere.ramps(mesh, vertices).sum(axis=0).dual().normalized()   # Point-[cells]
    starts = {"cell centres": centres, "edge midpoints": midpoints}
    seeds = {name: three_sphere.orbit_representatives(group, points.batch()) for name, points in starts.items()}
    return {name: (points + nudge * three_sphere_seeds(points.shape[0])).normalized()
            for name, points in seeds.items()}


def main() -> None:
    turns, anchors, _, _ = two_sphere_scene()
    group, cell_vertices = three_sphere_scene()
    refined = cell_refinements(group, cell_vertices)

    # checks
    # The icosahedron has sixty rotations and twelve points on its five-fold axes, the binary icosahedral group a
    # hundred and twenty turns, which carry the origin to as many points, and the 600-cell's six hundred cells and
    # seven hundred and twenty edges fall into five and six orbits of them.
    assert turns.shape[0] == 60 and anchors.shape[0] == 12
    assert group.shape[0] == 120 and three_sphere.unique_points(cell_vertices).shape[0] == 120
    assert refined["cell centres"].shape[0] == 5 and refined["edge midpoints"].shape[0] == 6


if __name__ == "__main__":
    main()
