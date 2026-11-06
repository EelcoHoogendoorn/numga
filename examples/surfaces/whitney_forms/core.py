"""Whitney forms interpolate vertex and edge data across triangle meshes.

The triangle's signed patch defines both corner gradients and edge forms. Their reconstruction maps let a material conductivity act on local fields, with the adjugate carrying its response back to the mesh.

One dimension up, in projective space, a tetrahedron's forms of every degree are one quotient: the join of the corners off a simplex over its join with the simplex's own corners. Pairing each edge form with each curl through the tetrahedron's volume links only opposite edges.
"""

from __future__ import annotations

from functools import reduce
from operator import and_

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D as ga, PGA3D as space
from numga.extensor import Extensor
from numga.sparse import SparseExtensor, spdiag
from examples.util.topology import Topology, simplex_parity

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-
Conductivity = ga.gatype((Point, Plane))
triangle = Topology(3, [[0, 1, 2]])

# The tetrahedron, in projective space: its corners are points of weight one, so that their joins are the planes, lines
# and points through them.
space_mv = NumpyContext(space).multivector
SpaceScalar = space.gatype.scalar()                                          # Scalar+
SpacePlane = space.gatype.vector()                                           # Vector+
SpaceBivector = space.gatype.bivector()                                      # Bivector+
SpacePoint = space.gatype.antivector()                                       # Antivector-


# --- plumbing -------------------------------------------------------------------------
def complements(topology: Topology, m: int) -> np.ndarray:
    """`[k, n - m]`: the corners each of an n-simplex's m-simplices leaves out, in the order of `topology.corners[n, m]`,
    run so that the m-simplex followed by them runs the n-simplex's own sense: facing each corner, its facet, outward;
    across each edge of a tet, the opposite edge."""
    n = topology.n_dim
    simplices = topology.corners[n, m].T                                       # [k, m + 1]
    top = topology.corners[n, 0][0]                                            # [corners]
    left_out = np.stack([np.setdiff1d(top, simplex) for simplex in simplices])   # [k, n - m]
    sense = simplex_parity(np.concatenate([simplices, left_out], axis=-1))   # [k]
    return np.where(sense[:, None] > 0, left_out, left_out[:, ::-1])


def part_middles(n_cuts: int) -> np.ndarray:
    """`[parts, 3]`: the barycentric weights of the middles of a triangle's equal parts, each side cut in `n_cuts`:
    the parts that point as the triangle does, then those that point the other way."""
    steps = np.arange(n_cuts)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    upright = first + second < n_cuts
    inverted = first + second < n_cuts - 1
    upright_corners = np.stack((first[upright], second[upright], n_cuts - 1 - first[upright] - second[upright]), axis=-1)
    inverted_corners = np.stack((first[inverted], second[inverted], n_cuts - 2 - first[inverted] - second[inverted]),
                                axis=-1)
    return np.concatenate((upright_corners + 1 / 3, inverted_corners + 2 / 3)) / n_cuts


# --- math -----------------------------------------------------------------------------
def barycentric_hats(corners: Point, points: Point) -> tuple[Scalar, Plane]:
    """The barycentric hat values at `points` and the constant corner gradients."""
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]  # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    # Dividing each opposite edge by twice the signed area patch gives the gradient
    # that falls by one from its corner to the opposite edge.
    gradients = sides / (2 * patch)                                          # [corners] Plane+
    hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+
    return hats, gradients


def edge_whitney_forms(corners: Point, points: Point) -> Plane:
    """The three edge Whitney 1-forms evaluated at `points`, circulating around each edge's opposite corner."""
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]  # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    # Along the two edges meeting at a corner, the radial displacement from that corner is parallel
    # to the edge, while along the opposite edge its join with the edge equals twice the triangle patch.
    return (points[None, :] - corners[:, None]) / (2 * patch)                # [corners, samples] Plane+


def tetrahedral_forms(topology: Topology, vertices: SpacePoint, m: int) -> Extensor:
    """The Whitney m-forms of each tetrahedron's m-simplices, `[k] (m + 1)-vector+[tets]`, each in the sense its
    m-simplex is stored: the join of the corners off the m-simplex over its join with the m-simplex's own corners. It
    reads one on its own m-simplex's join and nothing on another's, which repeats a corner off its own, and its value
    at a point `p` is `forms & p`: the ramp's height for a corner, for an edge the plane the edge across spans with
    the point, for a face the line from the corner it faces through the point."""
    # The corners off each m-simplex joined; the order they are joined in stands above and below the line alike, and
    # cancels.
    complements = reduce(and_, topology.off[3, m] * vertices)                # [k] (m + 1)-vector-[tets]
    # Each m-simplex's own corners joined in the order it is stored, the one sense the form keeps.
    spans = reduce(and_, topology.on[3, m] * vertices)                       # [k] (3 - m)-vector-[tets]
    return complements / (complements & spans)


def tetrahedral_edge_pages(topology: Topology, vertices: SpacePoint, n_pages: int) -> SpacePoint:
    """Level surfaces of the six edge Whitney 1-forms, `n_pages` of each at even steps of its unit circulation along
    its edge: the triangles the edge across spans with points of the edge, each the edge across's first corner, the
    point and its second, `[tet_edges, pages, 3] Point-[tets]`."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    tails, heads = topology.corners[3, 1]
    opposite_tails, opposite_heads = topology.complements[3, 1]
    # The form reads its own edge evenly, so pages through evenly spaced points of the edge lie equal shares of its
    # circulation apart.
    fractions = np.arange(1, n_pages + 1) / (n_pages + 1)                    # [pages]
    points = corners[tails, None] * (1 - fractions) + corners[heads, None] * fractions   # [tet_edges, pages] Point-[tets]
    # The form at a point is the plane the edge across spans with it, so the triangle that edge spans with the point
    # lies in the form's kernel throughout, a level surface of the form.
    opposite_tail = corners[opposite_tails, None].broadcast_to(points.shape)             # [tet_edges, pages] Point-[tets]
    opposite_head = corners[opposite_heads, None].broadcast_to(points.shape)             # [tet_edges, pages] Point-[tets]
    return stack((opposite_tail, points, opposite_head), axis=-1)


def tetrahedral_face_lines(topology: Topology, vertices: SpacePoint, n_cuts: int) -> SpacePoint:
    """Field lines of the four face Whitney 2-forms, through the middles of each face's equal parts, each side cut in
    `n_cuts`: each from the corner the face faces to where it crosses the face, `[corners, lines, 2] Point-[tets]`."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    # The corners of the face facing each corner, and the middles of its equal parts: the form's flux through its face
    # is even, so each line through them carries an equal share of the unit flux.
    facets = corners[topology.complements[3, 0].T]                           # [corners, 3] Point-[tets]
    crossings = (facets[:, None] * part_middles(n_cuts)).sum(axis=-1)        # [corners, lines] Point-[tets]
    # The form at a point is the line from the corner the face faces through the point, and reads no flux through any
    # patch along it: its field lines run straight out of that corner.
    return stack((corners[:, None].broadcast_to(crossings.shape), crossings), axis=-1)


def tetrahedral_helicity(topology: Topology, vertices: SpacePoint) -> SpaceScalar:
    """The volume pairing of each edge's Whitney 1-form with each edge's curl, made symmetric: it links only opposite
    edges, the same for every tetrahedron of the same sense, `[tet_edges, tet_edges] Scalar+[tets]`."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    forms = tetrahedral_forms(topology, vertices, 1)                         # [tet_edges] Bivector+[tets]
    # A form's curl reads two steps as twice the form does, constant; the form itself is linear across the
    # tetrahedron, so against a constant curl it integrates exactly at the centroid.
    curls = 2 * forms                                                        # [tet_edges] Bivector+[tets]
    centroid_forms = forms & corners.mean(axis=0)                            # [tet_edges] Plane+[tets]
    densities = ((centroid_forms[:, None] ^ curls[None, :]) + (centroid_forms[None, :] ^ curls[:, None])) / 2  # [tet_edges, tet_edges] Trivector+[tets]
    # The tetrahedron's volume, the join of its edges from its first corner in its own order, over six.
    volume = reduce(and_, corners[1:] - corners[0]) / 6                      # Vector-[tets]
    return densities & volume                                                # [tet_edges, tet_edges] Scalar+[tets]


class Operators:
    """Topological exterior derivatives and Whitney reconstruction maps on an oriented triangle mesh."""

    def __init__(self, topology: Topology, vertices: Point):
        self.topology = topology
        self.d0 = topology.d[0] * Scalar                                     # (Scalar+ <- Scalar+)[edges <- vertices]
        self.d1 = topology.d[1] * Scalar                                     # (Scalar+ <- Scalar+)[faces <- edges]
        edges = topology.d[0] * vertices                                     # Point-[edges]
        self.corners = topology.incidence[2, 0] * vertices                   # [corners] Point-[faces]
        self.sides = topology.boundary[2, 1] * edges                         # [corners] Point-[faces]
        self.patches = (self.sides[0] & self.sides[1]) / 2                   # Patch-[faces]
        self.areas = self.patches.norm()                                     # Scalar+[faces]
        self.centres = self.corners.sum(axis=0) / 3                          # Point-[faces]

        # Each corner hat's gradient is constant across its face; assembling by corner maps
        # vertex potentials to piecewise-constant face gradients without any metric.
        self.corner_gradients = self.sides / (2 * self.patches)              # [corners] Plane+[faces]
        self.gradient = (spdiag(self.corner_gradients) * topology.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- vertices]

        # At each face centroid the three barycentric hats equal 1/3, so the edge 1-form
        # reduces to one third of the difference of its endpoints' corner gradients.
        tails, heads = topology.corners[2, 1]
        centroid_forms = (self.corner_gradients[heads] - self.corner_gradients[tails]) / 3  # [corners] Plane+[faces]
        self.whitney1 = (spdiag(centroid_forms) * topology.oriented[2, 1]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- edges]

        # The exterior derivative of each edge's Whitney 1-form is twice the wedge of its endpoints'
        # gradients, matching the constant Whitney 2-form of unit integral over the face patch.
        curls = 2 * (self.corner_gradients[tails] ^ self.corner_gradients[heads])  # [corners] Bivector+[faces]
        self.curl = (spdiag(curls) * topology.oriented[2, 1]).sum(axis=0) * Scalar  # (Bivector+ <- Scalar+)[faces <- edges]
        self.whitney2 = mv.xy / (mv.xy & self.patches) * Scalar              # (Bivector+ <- Scalar+)[faces]

        # Evaluating the linear Whitney 1-forms at the three edge midpoints integrates
        # quadratic products over every triangle exactly.
        midpoints = topology.incidence[2, 1] * (topology.incidence[1, 0].mean(axis=0) * vertices)  # [midpoints] Point-[faces]
        midpoint_forms = (midpoints[:, None] - self.corners[None, :]) / (2 * self.patches)  # [midpoints, corners] Plane+[faces]
        self.midpoint_whitney1 = (spdiag(midpoint_forms) * topology.boundary[2, 1]).sum(axis=1) * Scalar  # [midpoints] (Plane+ <- Scalar+)[faces <- edges]


def edge_conductance(operators: Operators, conductivity: Conductivity) -> SparseExtensor:
    """The 1-form conductance operator on edges, integrated exactly over each triangle from its three edge midpoints."""
    # Each midpoint carries one third of the triangle's area; pulling the material map back
    # through the midpoint evaluation maps and summing over the three midpoints is exact for quadratics.
    weights = ((operators.areas / 3) * conductivity)[..., None]              # [..., midpoints] (Point- <- Plane+)[faces]
    return operators.midpoint_whitney1.adjugate()(weights(operators.midpoint_whitney1)).sum(axis=-1)  # [...] (Amount- <- Scalar+)[edges <- edges]


def conduction(operators: Operators, conductivity: Conductivity,
               electrodes: np.ndarray, prescribed: Scalar
               ) -> tuple[Scalar, Point, Scalar]:
    """Steady potential, face current, and streamfunction for prescribed electrode potentials."""
    topology = operators.topology
    # Pulling the area-weighted conductivity back through the face gradient gives the vertex conductance.
    medium = operators.areas * conductivity                                  # [cases] (Point- <- Plane+)[faces]
    conductance = operators.gradient.adjugate()(medium(operators.gradient))  # [cases] (Amount- <- Scalar+)[vertices <- vertices]
    free_vertices = np.setdiff1d(np.arange(topology.n_vertices), electrodes)
    free = ~topology.reading(free_vertices, 0) * Scalar                      # (Scalar+ <- Scalar+)[vertices <- free]
    rhs = -free.adjugate()(conductance(prescribed))                          # [cases] Amount-[free]
    potential = prescribed + free(free.adjugate()(conductance(free)).solve(rhs))  # [cases] Scalar+[vertices]
    current = -conductivity(operators.gradient(potential))                   # [cases] Point-[faces]

    # Turning the face current by a quarter turn gives, through the unit isotropic conductivity `Plane.dual()`,
    # the target gradient of the streamfunction whose level curves follow the current paths.
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Rotor+
    isotropic_medium = operators.areas * Plane.dual()                        # (Point- <- Plane+)[faces]
    laplacian = operators.gradient.adjugate()(isotropic_medium(operators.gradient))  # (Amount- <- Scalar+)[vertices <- vertices]
    unpinned = ~topology.reading(np.arange(1, topology.n_vertices), 0) * Scalar # (Scalar+ <- Scalar+)[vertices <- unpinned]
    stream_rhs = unpinned.adjugate()(operators.gradient.adjugate()(operators.areas * (quarter_turn >> current)))  # [cases] Amount-[unpinned]
    streamfunction = unpinned(unpinned.adjugate()(laplacian(unpinned)).solve(stream_rhs))  # [cases] Scalar+[vertices]
    return potential, current, streamfunction
