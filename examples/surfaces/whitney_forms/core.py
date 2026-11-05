"""Whitney forms interpolate vertex and edge data across triangle meshes.

The triangle's signed patch defines both corner gradients and edge forms. Their reconstruction maps let a material conductivity act on local fields, with the adjugate carrying its response back to the mesh.

One dimension up, in the algebra of space, a tetrahedron's signed volume does the same for its corner gradients, its edge forms and its face forms; pairing each edge form with each curl through that volume links only opposite edges.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D as ga, VGA3D as space
from numga.sparse import SparseExtensor, spdiag
from examples.util.topology import Topology, simplex_parity

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-
Conductivity = ga.gatype((Point, Plane))
triangle = Topology(3, [[0, 1, 2]])

# The tetrahedron, in the algebra of space.
space_mv = NumpyContext(space).multivector
SpaceScalar = space.gatype.scalar()                                          # Scalar+
SpacePlane = space.gatype.vector()                                           # Vector+
SpaceBivector = space.gatype.bivector()                                      # Bivector+
SpacePoint = space.gatype.antivector()                                       # Antivector-
SpacePatch = space.gatype.antibivector()                                     # Antibivector-
Volume = space.gatype.antitrivector()                                        # Antitrivector-


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


def tetrahedral_hats(topology: Topology, vertices: SpacePoint, points: SpacePoint
                     ) -> tuple[SpaceScalar, SpacePlane, SpacePatch, Volume]:
    """The barycentric hat values at `points` `[samples] Point-[tets]`, the constant corner gradients, the outward face
    patches, and the signed volume of each tetrahedron."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    # Each tetrahedron's edges in its own vertex order; the first three run from its first corner.
    edges = topology.oriented[3, 1] * (topology.d[0] * vertices)             # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    # Each face's patch, half the join of two of its sides, read as the face facing each corner, outward.
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [face_corners] Point-[faces]
    face_patches = topology.boundary[3, 2] * ((sides[0] & sides[1]) / 2)     # [corners] Patch-[tets]
    # The outward patch reads a step from a corner across to its face as three times the volume, and the hat falls
    # by one along that step: its gradient is the patch turned inward, over three times the volume.
    gradients = -face_patches / (3 * volume)                                 # [corners] Plane+[tets]
    hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+[tets]
    return hats, gradients, face_patches, volume


def tetrahedral_edge_forms(topology: Topology, vertices: SpacePoint, points: SpacePoint
                           ) -> tuple[SpacePlane, SpaceBivector]:
    """The six edge Whitney 1-forms at `points` `[samples] Point-[tets]`, circulating around each edge's opposite
    edge, and their constant curls."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    edges = topology.oriented[3, 1] * (topology.d[0] * vertices)             # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    # The edge across each edge, run so that the two together run the tetrahedron's own sense.
    opposite_tails, opposite_heads = complements(topology, 1).T
    opposite_edges = corners[opposite_heads] - corners[opposite_tails]       # [tet_edges] Point-[tets]
    # The patch the opposite edge sweeps to a point vanishes along that edge and, over six times the volume, reads
    # one along the edge itself; its curl is the opposite edge over three times the volume, a plane of space.
    forms = (opposite_edges[:, None] & (points[None, :] - corners[opposite_tails, None])) / (6 * volume)  # [tet_edges, samples] Plane+[tets]
    curls = opposite_edges / (3 * volume)                                    # [tet_edges] Bivector+[tets]
    return forms, curls


def tetrahedral_face_forms(topology: Topology, vertices: SpacePoint, points: SpacePoint) -> SpaceBivector:
    """The four face Whitney 2-forms at `points` `[samples] Point-[tets]`, radiating from each face's opposite
    corner."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    edges = topology.oriented[3, 1] * (topology.d[0] * vertices)             # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    # Along the three faces at a corner the displacement from that corner lies in the face, while across the face
    # facing it, it joins the face's outward patch to three times the volume.
    return (points[None, :] - corners[:, None]) / (3 * volume)               # [corners, samples] Bivector+[tets]


def tetrahedral_edge_pages(topology: Topology, vertices: SpacePoint, n_pages: int) -> SpacePoint:
    """Level surfaces of the six edge Whitney 1-forms, `n_pages` of each at even steps of its unit circulation along
    its edge: the triangles the opposite edge spans with points of the edge, each the opposite edge's tail, the point and
    the opposite edge's head, `[tet_edges, pages, 3] Point-[tets]`."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    tails, heads = topology.corners[3, 1]
    opposite_tails, opposite_heads = complements(topology, 1).T
    # The form reads its own edge evenly, so pages through evenly spaced points of the edge lie equal shares of its
    # circulation apart.
    fractions = np.arange(1, n_pages + 1) / (n_pages + 1)                    # [pages]
    points = corners[tails, None] * (1 - fractions) + corners[heads, None] * fractions   # [tet_edges, pages] Point-[tets]
    # The form at a point is the patch the opposite edge sweeps to it: it reads nothing along the opposite edge, nor
    # along any step within the triangle that edge spans with the point, so the triangle lies in the form's kernel
    # throughout, a level surface of the form.
    opposite_tail = corners[opposite_tails, None].broadcast_to(points.shape)             # [tet_edges, pages] Point-[tets]
    opposite_head = corners[opposite_heads, None].broadcast_to(points.shape)             # [tet_edges, pages] Point-[tets]
    return stack((opposite_tail, points, opposite_head), axis=-1)


def tetrahedral_face_lines(topology: Topology, vertices: SpacePoint, n_cuts: int) -> SpacePoint:
    """Field lines of the four face Whitney 2-forms, through the middles of each face's equal parts, each side cut in
    `n_cuts`: each from the corner the face faces to where it crosses the face, `[corners, lines, 2] Point-[tets]`."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    # The corners of the face facing each corner, and the middles of its equal parts: the form's flux through its face
    # is even, so each line through them carries an equal share of the unit flux.
    facets = corners[complements(topology, 0)]                               # [corners, 3] Point-[tets]
    crossings = (facets[:, None] * part_middles(n_cuts)).sum(axis=-1)        # [corners, lines] Point-[tets]
    # The form at a point is the step to it from the corner the face faces, over three times the volume, and reads no
    # flux through any patch along that step: its kernel runs along the step, so its field lines run straight out of
    # that corner.
    return stack((corners[:, None].broadcast_to(crossings.shape), crossings), axis=-1)


def tetrahedral_helicity(topology: Topology, vertices: SpacePoint) -> SpaceScalar:
    """The signed-volume pairing of each edge's Whitney 1-form with each edge's curl, made symmetric: it links only
    opposite edges, the same for every tetrahedron of the same sense, `[tet_edges, tet_edges] Scalar+[tets]`."""
    corners = topology.incidence[3, 0] * vertices                            # [corners] Point-[tets]
    _, gradients, _, volume = tetrahedral_hats(topology, vertices, corners)  # [corners, corners] Scalar+[tets], [corners] Plane+[tets], [corners] Patch-[tets], Volume-[tets]
    _, curls = tetrahedral_edge_forms(topology, vertices, corners)           # [tet_edges, corners] Plane+[tets], [tet_edges] Bivector+[tets]
    # At the centroid every hat is a quarter, so each edge form is a quarter of the difference of its ends'
    # gradients; a linear form against a constant curl integrates exactly there.
    tails, heads = topology.corners[3, 1]
    centroid_forms = (gradients[heads] - gradients[tails]) / 4               # [tet_edges] Plane+[tets]
    densities = ((centroid_forms[:, None] ^ curls[None, :]) + (centroid_forms[None, :] ^ curls[:, None])) / 2  # [tet_edges, tet_edges] Trivector+[tets]
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
        self.whitney2 = spdiag(mv.xy / (mv.xy & self.patches) * Scalar)      # (Bivector+ <- Scalar+)[faces <- faces]

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
