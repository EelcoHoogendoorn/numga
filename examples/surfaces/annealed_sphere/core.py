"""Round meshes of spheres of any dimension: points in orbits of a finite group of rotations, relaxed on the unit
sphere until the simplices between them are as round as the orbits allow.

A mesh of the sphere is the convex hull of points on it: seen from the centre its facets are the sphere's simplices,
the triangles of the two-sphere and the tetrahedra of the three-sphere, and the poles of their planes are its dual
vertices, the simplices' circumcentres. Each step turns every point toward the average of its simplices' poles, each
weighted by its share of the hull: the optimal Delaunay step, which rounds the simplices around the point.

The points come in orbits of the group: an orbit of anchors held still, and a few seeds in as many copies as the
group has rotations. Only the seeds move, their copies with them, so the mesh keeps the group's symmetry exactly. On
the two-sphere the group is the icosahedron's sixty rotations and the anchors the twelve points on their five-fold
axes; on the three-sphere it is the binary icosahedral group, a hundred and twenty turns alike in two square planes,
none of which keeps any point, and the anchors are the 600-cell's vertices, one orbit.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import reduce
from operator import and_

import numpy as np
from scipy.spatial import ConvexHull

from numga import Algebra, concatenate, stack
from numga.backend.context import Context
from examples import instantiate

# Supplied by examples.instantiate: the algebra of the space around the sphere, of any dimension, every generator
# squaring to one.
ga: Algebra
ctx: Context
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()                         # Scalar+
# Points and their directions from the centre are antivectors; planes through the centre, and the ramps across the
# simplices, vectors.
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Rotor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def mirrors() -> Plane:
    """The three mirrors of the icosahedron's symmetry, `[3] Plane+`, planes through the centre at a fifth of a half
    turn, a third of one and a quarter turn to each other: the rows of the Cholesky factor of their cosines, across
    the first three directions."""
    cosines = -np.cos(np.pi / np.array([[1, 5, 2], [5, 1, 3], [2, 3, 1]]))
    np.fill_diagonal(cosines, 1.0)
    return mv.vector(np.pad(np.linalg.cholesky(cosines), ((0, 0), (0, ga.n_dimensions - 3))))


def closure(generators: Rotor) -> Rotor:
    """The group of rotations the rotors generate, `[rotations] Rotor+`, each rotation once whatever the sign of its
    rotor, the identity first: every product of a generator with the rotations found so far, until none is new."""
    found = mv.rotor()[None]                                                    # [rotations] Rotor+
    while True:
        products = concatenate([found, (generators[:, None] * found[None]).reshape(-1)])
        kept = unique_up_to_sign(products)
        if len(kept) == found.shape[0]:
            return found
        found = products[kept]


def icosahedral_turns() -> Rotor:
    """The sixty rotations of the icosahedron, `[rotations] Rotor+`: the turns its mirrors make two at a time, a fifth
    and a third of a turn, and every product of them."""
    planes = mirrors()                                                          # [3] Plane+
    return closure(stack((planes[0] * planes[1], planes[1] * planes[2])))


def binary_icosahedral_turns() -> Rotor:
    """The hundred and twenty turns of the binary icosahedral group in four dimensions, `[rotations] Rotor+`: the
    icosahedron's turns of a fifth and a third, each carried alike in its own plane and the plane square to it, the
    half of the rotor the pseudoscalar keeps with one in the other half, turning by half its angle in both planes,
    and every product of them. None but the identity keeps any point of the three-sphere."""
    planes = mirrors()                                                          # [3] Plane+
    simple = stack((planes[0] * planes[1], planes[1] * planes[2]))              # [2] Rotor+
    pseudoscalar = mv.xyzw                                                      # [] Antiscalar+
    return closure((((1 + pseudoscalar) * simple + (1 - pseudoscalar)) / 2).cast(Rotor))


def orbit_points(turns: Rotor, anchors: Point, seeds: Point) -> Point:
    """The points of the orbits, `Point-[points]`: the anchors, then each seed's copies, the first under the
    identity."""
    copies = turns[None] >> seeds[:, None]                                      # [seeds, rotations] Point-
    return concatenate([anchors, copies.reshape(-1)]).field()


def ramps(mesh: Mesh, points: Point) -> Plane:
    """Each corner's ramp across its simplex, `[corners] Plane+[simplices]`: the plane through the centre and the
    corners across from it, their join, scaled to read one at the corner and so nothing at the others."""
    corners = mesh.incidence[-1, 0] * points                                    # [corners] Point-[simplices]
    facets = reduce(and_, corners[mesh.corners[-1, -2]])                        # [corners] Plane+[simplices]
    return facets / (facets & corners)


def optimal_delaunay_step(mesh: Mesh, points: Point, poles: Point, turns: Rotor, anchors: Point, seeds: Point,
                          pull: float) -> Point:
    """The seeds `[seeds] Point-` turned by `pull` of the way toward the average of their simplices' poles, each
    simplex weighted by its share of the hull, the points in `orbit_points`' order."""
    # Each simplex's share of the whole, the join of its corners over the sum of all of them.
    cones = reduce(and_, mesh.incidence[-1, 0] * points)                        # Scalar-[simplices]
    shares = cones / cones.sites.sum()                                          # Scalar+[simplices]
    pulled = (~mesh.incidence[-1, 0] * (shares * poles.normalized())).sum(axis=0).normalized()   # Point-[points]
    own = pulled.batch()[anchors.shape[0] + turns.shape[0] * np.arange(seeds.shape[0])]   # [seeds] Point-
    return (seeds + pull * (own - seeds)).normalized()


def relaxed(turns: Rotor, anchors: Point, seeds: Point, n_steps: int, pull: float
            ) -> Iterator[tuple[Mesh, Point, Point]]:
    """The mesh, its points and its simplices' poles at each step of the optimal Delaunay relaxation of the seeds."""
    for _ in range(n_steps):
        points = orbit_points(turns, anchors, seeds)                            # Point-[points]
        mesh = hull(points)
        # The pole of each simplex is the dual of the plane that reads one at every corner, the sum of their ramps.
        poles = ramps(mesh, points).sum(axis=0).dual()                          # Point-[simplices]
        yield mesh, points, poles
        seeds = optimal_delaunay_step(mesh, points, poles, turns, anchors, seeds, pull)


def facet_cosines(mesh: Mesh, points: Point) -> Scalar:
    """The cosine of the angle between each two facets of each simplex, `[pairs] Scalar+[simplices]`, on the
    simplices between the points' directions: a triangle's corner angles, a tetrahedron's dihedral angles. A
    corner's ramp, less its part across the simplex's own plane, rises square to the facet across from it."""
    corner_ramps = ramps(mesh, points.normalized())                             # [corners] Plane+[simplices]
    own = corner_ramps.sum(axis=0)                                              # Plane+[simplices]
    rises = ((corner_ramps ^ own) | own).normalized()                           # [corners] Plane+[simplices]
    # The facets across from two corners meet at the angle whose cosine is minus that of their rises.
    tails, heads = mesh.corners[-1, 1]
    return -(rises[tails] | rises[heads])


def valences(mesh: Mesh) -> Scalar:
    """How many simplices meet at each point, `Scalar+[points]`: one for each simplex, summed back to its corners."""
    simplices = mv.scalar(np.ones((mesh.n_elements[-1], 1))).field()            # Scalar+[simplices]
    return (~mesh.incidence[-1, 0] * simplices).sum(axis=0)


# --- plumbing -------------------------------------------------------------------------
def hull(points: Point) -> Mesh:
    """The convex hull of points around the centre, its facets wound outward, which scipy finds from the points'
    coordinates, one step outside the algebra."""
    # Each point read by pairing it with the planes across each direction.
    coordinates = (mv.vector(np.eye(ga.n_dimensions)) & points.batch()[..., None]).to_array()   # [points, dimensions]
    return Mesh.closed(ConvexHull(coordinates).simplices, coordinates)


def unique_up_to_sign(rotors: Rotor) -> np.ndarray:
    """The indices of the rotors `[n]` that differ, one of each, a rotor and its negative being one rotation, outside
    the algebra."""
    coefficients = np.asarray(rotors.kernel)
    signs = np.sign(coefficients[np.arange(len(coefficients)), np.argmax(np.abs(coefficients) > 1e-9, axis=1)])
    _, kept = np.unique(np.round(coefficients * signs[:, None], 8), axis=0, return_index=True)
    return np.sort(kept)


def orbit_representatives(turns: Rotor, candidates: Point) -> Point:
    """One point of each orbit the candidates `[n]` fall into, `[orbits] Point-`, read by pairing with the planes
    across each direction, outside the algebra: each candidate in turn, unless the orbit of one kept holds it."""
    axes = mv.vector(np.eye(ga.n_dimensions))                                   # [dimensions] Plane+
    coordinates = (axes & candidates[..., None]).to_array()                     # [n, dimensions]
    orbits = (axes & (turns[None] >> candidates[:, None])[..., None]).to_array()   # [n, rotations, dimensions]
    kept, covered = [], np.zeros(len(coordinates), dtype=bool)
    for index in range(len(coordinates)):
        if not covered[index]:
            kept.append(index)
            covered |= (np.abs(coordinates[:, None] - orbits[index][None]).max(axis=-1) < 1e-6).any(axis=-1)
    return candidates[np.array(kept)]


def unique_points(points: Point) -> Point:
    """The points `[n]` that differ, one of each, read by pairing with the planes across each direction, outside the
    algebra."""
    coordinates = (mv.vector(np.eye(ga.n_dimensions)) & points[..., None]).to_array()   # [n, dimensions]
    _, kept = np.unique(np.round(coordinates, 8), axis=0, return_index=True)
    return points[np.sort(kept)]
