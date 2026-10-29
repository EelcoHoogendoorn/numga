"""Beltrami fields carry magnetic linkage with stationary energy at fixed helicity.

Magnetostatics in spacetime, `x+y+z+t-`: the torus rests in one instant, its points are directions of space, and the magnetic field is a plane of space that reads its flux through an oriented patch as `field & patch`. A hat is constant in time, so its gradient is the volume its opposite face sweeps through time over the prism the tetrahedron sweeps, `(face_patches & time) / (3 * prism)`. Whitney edge forms reconstruct the potential, and their curls give the field. Energy pairs the field with its dual; helicity pairs the potential and its curl with signed volume. Their adjugates carry both readings back to edge amplitudes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga.algebras import STA as ga
from numga.sparse import spdiag
from examples.electromagnetism.beltrami_fields.plumbing import Mesh, as_scalar, complements, mv, time

Scalar = ga.gatype.scalar()          # Scalar+
Plane = ga.gatype.vector()           # Vector+
# The magnetic field, a plane of space.
Field = ga.gatype.bivector().derive.within("x y z")  # Bivector+
Trivector = ga.gatype.trivector()     # Trivector+
Point = ga.gatype.antivector()        # Antivector-
Patch = ga.gatype.antibivector()      # Antibivector-
Volume = ga.gatype.antitrivector()    # Antitrivector-
Amount = ga.gatype.antiscalar()       # Antiscalar-
Energy = ga.gatype((Amount, Scalar))
Curl = ga.gatype((Field, Scalar))


# --- math -----------------------------------------------------------------------------
def tetrahedral_hats(mesh: Mesh, vertices: Point, points: Point) -> tuple[Scalar, Plane, Patch, Volume]:
    """The barycentric hat values at `points` `[samples] Point-[tets]`, the constant corner gradients, the outward face
    patches, and the signed tetrahedral volume."""
    corners = mesh.incidence[3, 0] * vertices                                # [corners] Point-[tets]
    # Each tetrahedron's edges in its own vertex order; the first three run from its first corner.
    edges = mesh.oriented[3, 1] * (mesh.d[0] * vertices)                     # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    # The prism the tetrahedron sweeps in a unit of time; its size is the tetrahedron's volume.
    prism = volume & time                                                    # Scalar-[tets]
    # Each face's patch, half the join of two of its sides, read as the face facing each corner, outward.
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                     # [3] Point-[faces]
    face_patches = mesh.boundary[3, 2] * ((sides[0] & sides[1]) / 2)         # [corners] Patch-[tets]
    # Each hat is constant in time: the volume each outward opposite face sweeps through time, over three
    # times the prism, is the plane that reads a rise from zero across that face to one at its corner.
    gradients = (face_patches & time) / (3 * prism)                          # [corners] Plane+[tets]
    hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+[tets]
    return hats, gradients, face_patches, volume


def edge_whitney_forms(mesh: Mesh, vertices: Point, points: Point) -> tuple[Plane, Field]:
    """The six edge Whitney 1-forms evaluated at `points` `[samples] Point-[tets]` and their constant exterior
    derivatives across the tetrahedron."""
    corners = mesh.incidence[3, 0] * vertices                                # [corners] Point-[tets]
    edges = mesh.oriented[3, 1] * (mesh.d[0] * vertices)                     # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    prism = volume & time                                                    # Scalar-[tets]
    # The edge across each edge, run so that the two complete the tetrahedron's own sense.
    opposite_tails, opposite_heads = complements(mesh, 1).T
    opposite_edges = corners[opposite_heads] - corners[opposite_tails]       # [tet_edges] Point-[tets]
    # The area patch swept from the opposite edge to the evaluation point circulates around that opposite edge,
    # vanishing along it; swept through time, over six times the prism, it reads one along the edge itself.
    swept_patches = (points[None, :] - corners[opposite_tails, None]) & opposite_edges[:, None]  # [tet_edges, samples] Patch-[tets]
    forms = (swept_patches & time) / (6 * prism)                             # [tet_edges, samples] Plane+[tets]
    curls = (opposite_edges & time) / (3 * prism)                            # [tet_edges] Field+[tets]
    return forms, curls


def face_whitney_forms(mesh: Mesh, vertices: Point, points: Point) -> Field:
    """The four face Whitney 2-forms evaluated at `points` `[samples] Point-[tets]`, radiating from each face's
    opposite corner."""
    corners = mesh.incidence[3, 0] * vertices                                # [corners] Point-[tets]
    edges = mesh.oriented[3, 1] * (mesh.d[0] * vertices)                     # [tet_edges] Point-[tets]
    volume = (edges[0] & edges[1] & edges[2]) / 6                            # Volume-[tets]
    prism = volume & time                                                    # Scalar-[tets]
    # Along the three faces meeting at a corner, the radial displacement from that corner lies inside the face,
    # while across the opposite face its join with the face patch equals three times the signed volume;
    # swept through time, over three times the prism, it reads one across that face.
    return ((points[None, :] - corners[:, None]) & time) / (3 * prism)       # [tet_faces, samples] Field+[tets]


def tetrahedral_helicity(mesh: Mesh, vertices: Point) -> tuple[Scalar, Trivector]:
    """Opposite edges carry a geometry-independent helicity pairing through the tetrahedron's signed volume."""
    corners = mesh.incidence[3, 0] * vertices                                # [corners] Point-[tets]
    _, gradients, _, volume = tetrahedral_hats(mesh, vertices, corners)      # [corners, corners] Scalar+[tets], [corners] Plane+[tets], [corners] Patch-[tets], Volume-[tets]
    # Each edge's local tail and head among the tetrahedron's corners.
    tails, heads = mesh.corners(3, 1).T
    centroid_forms = (gradients[heads] - gradients[tails]) / 4               # [tet_edges] Plane+[tets]
    curls = 2 * (gradients[tails] ^ gradients[heads])                        # [tet_edges] Field+[tets]
    # Wedging each edge's centroid 1-form with another edge's constant 2-form curl gives the helicity 3-form;
    # pairing with the signed volume cancels the geometry on every opposite-edge pair.
    densities = ((centroid_forms[:, None] ^ curls[None, :]) + (centroid_forms[None, :] ^ curls[:, None])) / 2  # [tet_edges, tet_edges] Trivector+[tets]
    return densities & volume, densities                                     # [tet_edges, tet_edges] Scalar+[tets], [tet_edges, tet_edges] Trivector+[tets]


class Operators:
    """The Whitney reconstructions of potentials, edge amplitudes and fluxes, the curl, and the magnetic energy and
    helicity on a tetrahedral mesh."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh, self.vertices = mesh, vertices
        self.corners = mesh.incidence[3, 0] * vertices                       # [corners] Point-[tets]
        self.centres = mesh.incidence[3, 0].mean(axis=0) * vertices           # Point-[tets]
        # Each tetrahedron's edges in its own vertex order; the first three run from its first corner.
        edges = mesh.oriented[3, 1] * (mesh.d[0] * vertices)                  # [tet_edges] Point-[tets]
        self.volumes = (edges[0] & edges[1] & edges[2]) / 6                  # Volume-[tets]
        # The prism each tetrahedron sweeps in a unit of time; its size is the tetrahedron's volume.
        self.prisms = self.volumes & time                                    # Scalar-[tets]
        # Each face's patch, half the join of two of its sides, read as the face facing each corner, outward.
        sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                 # [3] Point-[faces]
        self.face_patches = mesh.boundary[3, 2] * ((sides[0] & sides[1]) / 2)  # [corners] Patch-[tets]

        # Each corner hat's gradient is constant across its tetrahedron; assembled by corner, the gradients map
        # vertex potentials to piecewise-constant tetrahedral gradients without any metric.
        self.corner_gradients = (self.face_patches & time) / (3 * self.prisms)  # [corners] Plane+[tets]
        self.gradient = (spdiag(self.corner_gradients * Scalar) * mesh.incidence[3, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[tets <- vertices]

        # At each tetrahedron centroid the four barycentric hats equal 1/4, so the edge 1-form
        # reduces to one quarter of the difference of its endpoints' corner gradients.
        tails, heads = mesh.corners(3, 1).T
        centroid_1forms = (self.corner_gradients[heads] - self.corner_gradients[tails]) / 4  # [tet_edges] Plane+[tets]
        # A tet's edges run from its lower local corner to its higher, the mesh's edges from the lower vertex to the
        # higher: `mesh.oriented[3, 1]` reads each mesh edge with the sign of their agreement.
        self.whitney1 = (spdiag(centroid_1forms * Scalar) * mesh.oriented[3, 1]).sum(axis=0)  # (Plane+ <- Scalar+)[tets <- edges]

        # The exterior derivative of each edge's Whitney 1-form is twice the wedge of its endpoints' gradients,
        # matching the face Whitney 2-form evaluated at the centroid on the edges' circulations around each face.
        curls = 2 * (self.corner_gradients[tails] ^ self.corner_gradients[heads])  # [tet_edges] Field+[tets]
        self.curl = (spdiag(curls * Scalar) * mesh.oriented[3, 1]).sum(axis=0)  # (Field+ <- Scalar+)[tets <- edges]
        centroid_2forms = ((self.centres - self.corners) & time) / (3 * self.prisms)  # [tet_faces] Field+[tets]
        self.whitney2 = (spdiag(centroid_2forms * Scalar) * mesh.boundary[3, 2]).sum(axis=0)  # (Field+ <- Scalar+)[tets <- faces]

        # Magnetic energy pulls the vacuum, which takes each tetrahedron's field to the excitation it causes
        # weighted by its volume, back through the curl; magnetic helicity pulls the metric-free volume
        # pairing back through the centroid 1-form and the curl.
        vacuum = spdiag(self.prisms.norm() * Field.dual())                   # (Patch- <- Field+)[tets <- tets]
        self.stiffness = self.curl.adjugate()(vacuum(self.curl))             # (Amount- <- Scalar+)[edges <- edges]
        volume_weights = spdiag(Plane ^ self.volumes)                        # (Patch- <- Plane+)[tets <- tets]
        raw_helicity = self.curl.adjugate()(volume_weights(self.whitney1))   # (Amount- <- Scalar+)[edges <- edges]
        self.helicity = (raw_helicity + raw_helicity.adjugate()) / 2         # (Amount- <- Scalar+)[edges <- edges]


def selective_decay(initial: Scalar, eigenvalues: Scalar, times: np.ndarray) -> Iterator[Scalar]:
    """Yield selectively damped modal amplitudes rescaled to preserve their initial helicity."""
    initial_helicity = (initial.squared() / eigenvalues).sum(axis=0)          # [] Scalar+
    for elapsed in times:
        # Finer modes decay faster; the shared scale fixes the linkage of the whole field.
        damped = initial * (-eigenvalues.squared() * elapsed).exp()           # [selected] Scalar+
        helicity = (damped.squared() / eigenvalues).sum(axis=0)               # [] Scalar+
        yield damped * (initial_helicity / helicity).square_root()           # [selected] Scalar+


def reconstruct(states: Iterable[Scalar], modes: Scalar, curl: Curl) -> Iterator[Field]:
    """Yield the reconstructed magnetic flux field for each modal state."""
    for amplitudes in states:
        cochain = (amplitudes * modes).sum(axis=0)                            # Scalar+[edges]
        yield curl(cochain)                                                  # Field+[tets]
