"""Symmetry-orbit triangulations of the two-sphere and tetrahedralizations of the three-sphere
relax free seed orbits on the unit sphere by optimal Delaunay steps while preserving finite
rotational symmetry exactly.

Seen from the origin, the convex-hull facets of points on the unit sphere form spherical simplices
(triangles on the two-sphere and tetrahedra on the three-sphere) whose circumcentres, the dual
vertices, are the normalized duals of the facet planes. Each optimal Delaunay step moves every free
seed toward the cone-volume-weighted average of its incident simplices' circumcentres and projects
back onto the unit sphere, rounding the simplices as far as the symmetry orbits allow.

Vertices combine a fixed anchor orbit with full symmetry orbits of a few free seed points. Moving
only the seeds and regenerating their images keeps the group symmetry exact at every step. On the
two-sphere (`3` ambient dimensions), the group is the sixty-element icosahedral rotation group with
twelve five-fold axis anchors; on the three-sphere (`4` ambient dimensions), it is the
`120`-element binary icosahedral group of isoclinic rotations, which acts freely on the sphere and
takes the `600`-cell vertices as a single anchor orbit. Companion notes on simplicial fields and
plane-based orientation are in `docs/fields.md` and `docs/orientation.md`.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import reduce
from operator import and_

import numpy as np
from scipy.spatial import cKDTree

from numga import Algebra, concatenate, stack
from numga.backend.context import Context
from examples.util import shapes
from numga.topology import Topology

# Supplied by `examples.instantiate`: the 3D or 4D Euclidean ambient algebra around the unit sphere.
ga: Algebra
ctx: Context
mv = ctx.multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# Unit-sphere points and ray directions from the origin are intrinsic antivectors; planes through
# the origin and simplex gradients are extrinsic vectors.
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Rotor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def mirrors() -> Plane:
    """Cholesky factorization of the `(3, 5)` Coxeter cosine matrix embeds the three icosahedral
    symmetry mirrors (`[3] Plane+`) in the first three ambient coordinates."""
    cosines = -np.cos(np.pi / np.array([[1, 5, 2], [5, 1, 3], [2, 3, 1]]))
    np.fill_diagonal(cosines, 1.0)
    return mv.vector(np.pad(np.linalg.cholesky(cosines), ((0, 0), (0, ga.n_dimensions - 3))))


def closure(generators: Rotor) -> Rotor:
    """Multiplying `generators` repeatedly and deduplicating `r` against `-r` closes the finite
    rotation group (`[n_turns] Rotor+`) with the identity rotor at index `0`."""
    found = mv.rotor()[None]                                                    # [...] Rotor+
    while True:
        products = concatenate([found, (generators[:, None] * found[None]).reshape(-1)])
        kept = unique_up_to_sign(products)
        if len(kept) == found.shape[0]:
            return found
        found = products[kept]


def icosahedral_turns() -> Rotor:
    """Pairwise products of the three icosahedral `mirrors()` generate the `60` rotations (`[60]
    Rotor+`) of the icosahedral group."""
    planes = mirrors()                                                          # [3] Plane+
    return closure(stack((planes[0] * planes[1], planes[1] * planes[2])))


def binary_icosahedral_turns() -> Rotor:
    """Projecting the five-fold and three-fold icosahedral generators onto the self-dual chiral
    projector `(1 + I) / 2` generates the `120` isoclinic rotations (`[120] Rotor+`) of the binary
    icosahedral group in four dimensions, none of which (save the identity) fixes any point on the
    three-sphere."""
    planes = mirrors()                                                          # [3] Plane+
    simple = stack((planes[0] * planes[1], planes[1] * planes[2]))              # [2] Rotor+
    pseudoscalar = mv.xyzw                                                      # [] Antiscalar+
    return closure((((1 + pseudoscalar) * simple + (1 - pseudoscalar)) / 2).cast(Rotor))


def orbit_points(turns: Rotor, anchors: Point, seeds: Point) -> Point:
    """Concatenating `anchors` with `turns >> seeds` assembles the vertex field `Point-[N[0]]`,
    placing each seed itself first in its orbit block under the leading identity rotor."""
    copies = turns[None] >> seeds[:, None]                                      # [..., ...] Point-
    return concatenate([anchors, copies.reshape(-1)]).field()


def gradients(topology: Topology, points: Point) -> Plane:
    """Joining the opposite facet corners of each top simplex and dividing by their join with the
    remaining corner gives the simplex gradients (`[n_dim + 1] Plane+[N[-1]]`) that pair to `1` on
    their own corner and `0` on the opposite facet."""
    n_dim = topology.n_dim
    N = topology.n_elements
    corners = topology.incidence[-1, 0] * points                                # [n_dim + 1] Point-[N[-1]]
    facets = reduce(and_, corners[topology.corners[-1, -2]])                    # [n_dim + 1] Plane+[N[-1]]
    return facets / (facets & corners)


def optimal_delaunay_step(topology: Topology, points: Point, turns: Rotor, anchors: Point, seeds: Point,
                          pull: float) -> Point:
    """Cone-volume-weighted circumcentres (`poles`) of incident top simplices pull each seed in
    `seeds` (`[n_seeds] Point-`) by fraction `pull` before renormalizing back onto the unit
    sphere."""
    N = topology.n_elements
    n_anchors, n_turns, n_seeds = anchors.shape[0], turns.shape[0], seeds.shape[0]
    # Summing a simplex's corner gradients gives the plane that reads 1 on every corner; its normalized dual is the circumcentre.
    poles = gradients(topology, points).sum(axis=0).dual().normalized()         # Point-[N[-1]]
    # Weight each top simplex by its cone join from the origin divided by the total hull volume.
    cones = reduce(and_, topology.incidence[-1, 0] * points)                    # Scalar-[N[-1]]
    shares = cones / cones.sites.sum()                                          # Scalar+[N[-1]]
    averages = (topology.incidence[-1, 0].adjugate() * (shares * poles)).sum(axis=0).normalized()  # Point-[N[0]]
    # Read the averaged target at the leading identity-image index of each seed's orbit block.
    seed_sites = n_anchors + n_turns * np.arange(n_seeds)                       # [n_seeds]
    seed_averages = (topology.reading(seed_sites, 0) * averages).batch()        # [n_seeds] Point-
    return (seeds + pull * (seed_averages - seeds)).normalized()


def relaxed(turns: Rotor, anchors: Point, seeds: Point, n_steps: int, pull: float) -> Iterator[tuple[Topology, Point]]:
    """Yields `(topology, points)` across `n_steps` rounds of convex-hull construction and optimal
    Delaunay seed relaxation."""
    for _ in range(n_steps):
        points = orbit_points(turns, anchors, seeds)
        topology = hull(points)
        yield topology, points
        seeds = optimal_delaunay_step(topology, points, turns, anchors, seeds, pull)


def facet_cosines(topology: Topology, points: Point) -> Scalar:
    """Projecting each corner's simplex gradient orthogonal to its simplex plane gives inward facet
    normals whose negated pairwise inner products equal the interior corner or dihedral cosines
    (`[n_pairs] Scalar+[N[-1]]`)."""
    N = topology.n_elements
    corner_gradients = gradients(topology, points.normalized())                 # [...] Plane+[N[-1]]
    simplex_planes = corner_gradients.sum(axis=0)                               # Plane+[N[-1]]
    rises = ((corner_gradients ^ simplex_planes) | simplex_planes).normalized() # [...] Plane+[N[-1]]
    # Inward facet normals meet at the supplement of the interior angle between the facets.
    tails, heads = topology.corners[-1, 1]
    return -(rises[tails] | rises[heads])


def valences(topology: Topology) -> Scalar:
    """Adjugate vertex-simplex incidence sums unit top-simplex weights into the incident-simplex
    count `Scalar+[N[0]]` at each vertex."""
    N = topology.n_elements
    simplices = mv.scalar(np.ones((N[-1], 1))).field()                          # Scalar+[N[-1]]
    return (topology.incidence[-1, 0].adjugate() * simplices).sum(axis=0)


# --- plumbing -------------------------------------------------------------------------
# Coordinate/coefficient distance threshold below which points or signed rotors count as identical.
same = 1e-6


def coordinates_of(values) -> np.ndarray:
    """Regressive pairing against the ambient basis planes reads `values` (`[...]`) into a
    coordinate array of shape `[..., n_dimensions]`."""
    return (mv.vector(np.eye(ga.n_dimensions)) & values[..., None]).to_array()


def hull(points: Point) -> Topology:
    """Outward-wound convex-hull boundary `Topology` of `points` around the origin, computed via
    `shapes.convex_boundary` on their ambient coordinates."""
    coordinates = coordinates_of(points.batch())
    topology, _ = shapes.convex_boundary(coordinates)
    return topology


def unique_up_to_sign(rotors: Rotor) -> np.ndarray:
    """Canonicalizing each rotor's first non-negligible coefficient to positive sign and querying a
    k-d tree within `same` selects the first index of each distinct rotation in `rotors` (`[n]`)."""
    coefficients = np.asarray(rotors.kernel)
    signs = np.sign(coefficients[np.arange(len(coefficients)), np.argmax(np.abs(coefficients) > same, axis=1)])
    signed = coefficients * signs[:, None]
    # Drop any rotor that lies within `same` of an earlier canonicalized rotor.
    pairs = cKDTree(signed).query_pairs(same, output_type="ndarray")
    return np.setdiff1d(np.arange(len(signed)), pairs.max(axis=-1))


def orbit_representatives(turns: Rotor, candidates: Point) -> Point:
    """Greedy orbit screening retains the first point from each distinct `turns`-orbit present in
    `candidates` (`[n_candidates] Point-`)."""
    coordinates = coordinates_of(candidates)
    orbits = coordinates_of(turns[None] >> candidates[:, None])
    kept, covered = [], np.zeros(len(coordinates), dtype=bool)
    for index in range(len(coordinates)):
        if not covered[index]:
            kept.append(index)
            covered |= (np.abs(coordinates[:, None] - orbits[index][None]).max(axis=-1) < same).any(axis=-1)
    return candidates[np.array(kept)]


def unique_points(points: Point) -> Point:
    """Nearest-neighbour filtering within `same` retains the first occurrence of each distinct point
    in `points` (`[n_points] Point-`)."""
    coordinates = coordinates_of(points)
    pairs = cKDTree(coordinates).query_pairs(same, output_type="ndarray")
    return points[np.setdiff1d(np.arange(len(coordinates)), pairs.max(axis=-1))]
