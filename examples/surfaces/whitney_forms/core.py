"""Whitney forms interpolate vertex and edge data across triangle meshes.

The triangle's signed patch defines both corner gradients and edge forms. Their reconstruction maps let a material conductivity act on local fields, with the adjugate carrying its response back to the mesh.

One dimension up, in the algebra of space, a tetrahedron's signed volume does the same for its corner gradients, its edge forms and its face forms; pairing each edge form with each curl through that volume links only opposite edges.
"""

from __future__ import annotations

import numpy as np

from numga.algebras import VGA2D as ga, VGA3D as space
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-
Conductivity = ga.gatype((Point, Plane))
triangle = Mesh(3, [[0, 1, 2]])

# The tetrahedron, in the algebra of space.
space_mesh_module = instantiate("examples.util.mesh", space)
SpaceMesh, space_mv, simplex_parity = space_mesh_module.Mesh, space_mesh_module.mv, space_mesh_module.simplex_parity
SpaceScalar = space.gatype.scalar()                                          # Scalar+
SpacePlane = space.gatype.vector()                                           # Vector+
SpaceBivector = space.gatype.bivector()                                      # Bivector+
SpacePoint = space.gatype.antivector()                                       # Antivector-
SpacePatch = space.gatype.antibivector()                                     # Antibivector-
Volume = space.gatype.antitrivector()                                        # Antitrivector-


# --- plumbing -------------------------------------------------------------------------
def complements(mesh: Mesh | SpaceMesh, m: int) -> np.ndarray:
    """`[k, n - m]`: the corners each of an n-simplex's m-simplices leaves out, in the order of `mesh.corners[n, m]`,
    run so that the m-simplex followed by them runs the n-simplex's own sense: facing each corner, its facet, outward;
    across each edge of a tet, the opposite edge."""
    n = mesh.n_dim
    simplices = mesh.corners[n, m].T                                           # [k, m + 1]
    top = mesh.corners[n, 0][0]                                                # [corners]
    left_out = np.stack([np.setdiff1d(top, simplex) for simplex in simplices])   # [k, n - m]
    sense = simplex_parity(np.concatenate([simplices, left_out], axis=-1))   # [k]
    return np.where(sense[:, None] > 0, left_out, left_out[:, ::-1])


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


def tetrahedral_hats(mesh: SpaceMesh, vertices: SpacePoint, points: SpacePoint
                     ) -> tuple[SpaceScalar, SpacePlane, SpacePatch, Volume]:
    """The barycentric hat values at `points` `[samples] Point-[tets]`, the constant corner gradients, the outward face
    patches, and the signed volume of each tetrahedron."""
    corners = mesh.incidence[3, 0] * vertices                                # [corners] Point-[tets]
    # Each tetrahedron's edges in its own vertex order; the first three run from its first corner.
    edges = mesh.oriented[3, 1] * (mesh.d[0] * vertices)                     # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    # Each face's patch, half the join of two of its sides, read as the face facing each corner, outward.
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                     # [face_corners] Point-[faces]
    face_patches = mesh.boundary[3, 2] * ((sides[0] & sides[1]) / 2)         # [corners] Patch-[tets]
    # The outward patch reads a step from a corner across to its face as three times the volume, and the hat falls
    # by one along that step: its gradient is the patch turned inward, over three times the volume.
    gradients = -face_patches / (3 * volume)                                 # [corners] Plane+[tets]
    hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+[tets]
    return hats, gradients, face_patches, volume


def tetrahedral_edge_forms(mesh: SpaceMesh, vertices: SpacePoint, points: SpacePoint
                           ) -> tuple[SpacePlane, SpaceBivector]:
    """The six edge Whitney 1-forms at `points` `[samples] Point-[tets]`, circulating around each edge's opposite
    edge, and their constant curls."""
    corners = mesh.incidence[3, 0] * vertices                                # [corners] Point-[tets]
    edges = mesh.oriented[3, 1] * (mesh.d[0] * vertices)                     # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    # The edge across each edge, run so that the two together run the tetrahedron's own sense.
    opposite_tails, opposite_heads = complements(mesh, 1).T
    opposite_edges = corners[opposite_heads] - corners[opposite_tails]       # [tet_edges] Point-[tets]
    # The patch the opposite edge sweeps to a point vanishes along that edge and, over six times the volume, reads
    # one along the edge itself; its curl is the opposite edge over three times the volume, a plane of space.
    forms = (opposite_edges[:, None] & (points[None, :] - corners[opposite_tails, None])) / (6 * volume)  # [tet_edges, samples] Plane+[tets]
    curls = opposite_edges / (3 * volume)                                    # [tet_edges] Bivector+[tets]
    return forms, curls


def tetrahedral_face_forms(mesh: SpaceMesh, vertices: SpacePoint, points: SpacePoint) -> SpaceBivector:
    """The four face Whitney 2-forms at `points` `[samples] Point-[tets]`, radiating from each face's opposite
    corner."""
    corners = mesh.incidence[3, 0] * vertices                                # [corners] Point-[tets]
    edges = mesh.oriented[3, 1] * (mesh.d[0] * vertices)                     # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    # Along the three faces at a corner the displacement from that corner lies in the face, while across the face
    # facing it, it joins the face's outward patch to three times the volume.
    return (points[None, :] - corners[:, None]) / (3 * volume)               # [corners, samples] Bivector+[tets]


def tetrahedral_helicity(mesh: SpaceMesh, vertices: SpacePoint) -> SpaceScalar:
    """The signed-volume pairing of each edge's Whitney 1-form with each edge's curl, made symmetric: it links only
    opposite edges, the same for every tetrahedron of the same sense, `[tet_edges, tet_edges] Scalar+[tets]`."""
    corners = mesh.incidence[3, 0] * vertices                                # [corners] Point-[tets]
    _, gradients, _, volume = tetrahedral_hats(mesh, vertices, corners)      # [corners, corners] Scalar+[tets], [corners] Plane+[tets], [corners] Patch-[tets], Volume-[tets]
    _, curls = tetrahedral_edge_forms(mesh, vertices, corners)               # [tet_edges, corners] Plane+[tets], [tet_edges] Bivector+[tets]
    # At the centroid every hat is a quarter, so each edge form is a quarter of the difference of its ends'
    # gradients; a linear form against a constant curl integrates exactly there.
    tails, heads = mesh.corners[3, 1]
    centroid_forms = (gradients[heads] - gradients[tails]) / 4               # [tet_edges] Plane+[tets]
    densities = ((centroid_forms[:, None] ^ curls[None, :]) + (centroid_forms[None, :] ^ curls[:, None])) / 2  # [tet_edges, tet_edges] Trivector+[tets]
    return densities & volume                                                # [tet_edges, tet_edges] Scalar+[tets]


class Operators:
    """Topological exterior derivatives and Whitney reconstruction maps on an oriented triangle mesh."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh = mesh
        self.d0 = mesh.d[0] * Scalar                                         # (Scalar+ <- Scalar+)[edges <- vertices]
        self.d1 = mesh.d[1] * Scalar                                         # (Scalar+ <- Scalar+)[faces <- edges]
        edges = mesh.d[0] * vertices                                         # Point-[edges]
        self.corners = mesh.incidence[2, 0] * vertices                       # [corners] Point-[faces]
        self.sides = mesh.boundary[2, 1] * edges                             # [corners] Point-[faces]
        self.patches = (self.sides[0] & self.sides[1]) / 2                   # Patch-[faces]
        self.areas = self.patches.norm()                                     # Scalar+[faces]
        self.centres = self.corners.sum(axis=0) / 3                          # Point-[faces]

        # Each corner hat's gradient is constant across its face; assembling by corner maps
        # vertex potentials to piecewise-constant face gradients without any metric.
        self.corner_gradients = self.sides / (2 * self.patches)              # [corners] Plane+[faces]
        self.gradient = (spdiag(self.corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- vertices]

        # At each face centroid the three barycentric hats equal 1/3, so the edge 1-form
        # reduces to one third of the difference of its endpoints' corner gradients.
        tails, heads = mesh.corners[2, 1]
        centroid_forms = (self.corner_gradients[heads] - self.corner_gradients[tails]) / 3  # [corners] Plane+[faces]
        self.whitney1 = (spdiag(centroid_forms) * mesh.oriented[2, 1]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- edges]

        # The exterior derivative of each edge's Whitney 1-form is twice the wedge of its endpoints'
        # gradients, matching the constant Whitney 2-form of unit integral over the face patch.
        curls = 2 * (self.corner_gradients[tails] ^ self.corner_gradients[heads])  # [corners] Bivector+[faces]
        self.curl = (spdiag(curls) * mesh.oriented[2, 1]).sum(axis=0) * Scalar  # (Bivector+ <- Scalar+)[faces <- edges]
        self.whitney2 = spdiag(mv.xy / (mv.xy & self.patches) * Scalar)      # (Bivector+ <- Scalar+)[faces <- faces]

        # Evaluating the linear Whitney 1-forms at the three edge midpoints integrates
        # quadratic products over every triangle exactly.
        midpoints = mesh.incidence[2, 1] * (mesh.incidence[1, 0].mean(axis=0) * vertices)  # [midpoints] Point-[faces]
        midpoint_forms = (midpoints[:, None] - self.corners[None, :]) / (2 * self.patches)  # [midpoints, corners] Plane+[faces]
        self.midpoint_whitney1 = (spdiag(midpoint_forms) * mesh.boundary[2, 1]).sum(axis=1) * Scalar  # [midpoints] (Plane+ <- Scalar+)[faces <- edges]


def edge_conductance(operators: Operators, conductivity: Conductivity) -> SparseExtensor:
    """The 1-form conductance operator on edges, integrated exactly over each triangle from its three edge midpoints."""
    # Each midpoint carries one third of the triangle's area; pulling the material map back
    # through the midpoint evaluation maps and summing over the three midpoints is exact for quadratics.
    weights = spdiag(((operators.areas / 3) * conductivity)[..., None])      # [..., midpoints] (Point- <- Plane+)[faces <- faces]
    return operators.midpoint_whitney1.adjugate()(weights(operators.midpoint_whitney1)).sum(axis=-1)  # [...] (Amount- <- Scalar+)[edges <- edges]


def conduction(operators: Operators, conductivity: Conductivity,
               electrodes: np.ndarray, prescribed: Scalar
               ) -> tuple[Scalar, Point, Scalar]:
    """Steady potential, face current, and streamfunction for prescribed electrode potentials."""
    mesh = operators.mesh
    # Pulling the area-weighted conductivity back through the face gradient gives the vertex conductance.
    conductance = operators.gradient.adjugate()(spdiag(operators.areas * conductivity)(operators.gradient))  # [cases] (Amount- <- Scalar+)[vertices <- vertices]
    free_vertices = np.setdiff1d(np.arange(mesh.n_vertices), electrodes)
    free = ~mesh.reading(free_vertices) * Scalar                             # (Scalar+ <- Scalar+)[vertices <- free]
    rhs = -free.adjugate()(conductance(prescribed))                          # [cases] Amount-[free]
    potential = prescribed + free(free.adjugate()(conductance(free)).solve(rhs))  # [cases] Scalar+[vertices]
    current = -conductivity(operators.gradient(potential))                   # [cases] Point-[faces]

    # Turning the face current by a quarter turn gives, through the unit isotropic conductivity `Plane.dual()`,
    # the target gradient of the streamfunction whose level curves follow the current paths.
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Rotor+
    laplacian = operators.gradient.adjugate()(spdiag(operators.areas * Plane.dual())(operators.gradient))  # (Amount- <- Scalar+)[vertices <- vertices]
    unpinned = ~mesh.reading(np.arange(1, mesh.n_vertices)) * Scalar         # (Scalar+ <- Scalar+)[vertices <- unpinned]
    stream_rhs = unpinned.adjugate()(operators.gradient.adjugate()(operators.areas * (quarter_turn >> current)))  # [cases] Amount-[unpinned]
    streamfunction = unpinned(unpinned.adjugate()(laplacian(unpinned)).solve(stream_rhs))  # [cases] Scalar+[vertices]
    return potential, current, streamfunction
