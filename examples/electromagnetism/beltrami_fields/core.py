"""Beltrami fields carry magnetic linkage with stationary energy at fixed helicity.

Magnetostatics in spacetime, `x+y+z+t-`: the torus rests in one instant, its points are directions of space, and the magnetic field is a plane of space that reads its flux through an oriented patch as `field & patch`. A hat is constant in time, so its gradient is the volume its opposite face sweeps through time over the prism the tetrahedron sweeps, `(face_patches & time) / (3 * prism)`. Whitney edge forms reconstruct the potential, and their curls give the field. Energy pairs the field with its dual; helicity pairs the potential and its curl with signed volume. Their adjugates carry both readings back to edge amplitudes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga.algebras import STA as ga
from numga.sparse import spdiag
from examples.electromagnetism.beltrami_fields.plumbing import (
    Mesh, as_scalar, mv, time, edge_tails, edge_heads,
    opposite_tails, opposite_heads, face_first, face_second, face_third,
)

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
def tetrahedral_hats(corners: Point, points: Point) -> tuple[Scalar, Plane, Patch, Volume]:
    """The barycentric hat values at `points`, the constant corner gradients, the outward face patches, and the signed tetrahedral volume."""
    volume = ((corners[1] - corners[0]) & (corners[2] - corners[0]) & (corners[3] - corners[0])) / 6  # [] Volume-
    # The prism the tetrahedron sweeps in a unit of time; its size is the tetrahedron's volume.
    prism = volume & time                                                    # [] Scalar-
    face_patches = ((corners[face_second] - corners[face_first]) & (corners[face_third] - corners[face_first])) / 2  # [corners] Patch-
    # Each hat is constant in time: the volume each outward opposite face sweeps through time, over three
    # times the prism, is the plane that reads a rise from zero across that face to one at its corner.
    gradients = (face_patches & time) / (3 * prism)                          # [corners] Plane+
    hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+
    return hats, gradients, face_patches, volume


def edge_whitney_forms(corners: Point, points: Point) -> tuple[Plane, Field]:
    """The six edge Whitney 1-forms evaluated at `points` and their constant exterior derivatives across the tetrahedron."""
    volume = ((corners[1] - corners[0]) & (corners[2] - corners[0]) & (corners[3] - corners[0])) / 6  # [] Volume-
    prism = volume & time                                                    # [] Scalar-
    opposite_edges = corners[opposite_heads] - corners[opposite_tails]       # [tet_edges] Point-
    # The area patch swept from the opposite edge to the evaluation point circulates around that opposite edge,
    # vanishing along it; swept through time, over six times the prism, it reads one along the edge itself.
    swept_patches = (points[None, :] - corners[opposite_tails, None]) & opposite_edges[:, None]  # [tet_edges, samples] Patch-
    forms = (swept_patches & time) / (6 * prism)                             # [tet_edges, samples] Plane+
    curls = (opposite_edges & time) / (3 * prism)                            # [tet_edges] Field+
    return forms, curls


def face_whitney_forms(corners: Point, points: Point) -> Field:
    """The four face Whitney 2-forms evaluated at `points`, radiating from each face's opposite corner."""
    volume = ((corners[1] - corners[0]) & (corners[2] - corners[0]) & (corners[3] - corners[0])) / 6  # [] Volume-
    prism = volume & time                                                    # [] Scalar-
    # Along the three faces meeting at a corner, the radial displacement from that corner lies inside the face,
    # while across the opposite face its join with the face patch equals three times the signed volume;
    # swept through time, over three times the prism, it reads one across that face.
    return ((points[None, :] - corners[:, None]) & time) / (3 * prism)       # [tet_faces, samples] Field+


def tetrahedral_helicity(corners: Point) -> tuple[Scalar, Trivector]:
    """Opposite edges carry a geometry-independent helicity pairing through the tetrahedron's signed volume."""
    _, gradients, _, volume = tetrahedral_hats(corners, corners)             # [corners, corners] Scalar+, [corners] Plane+, [corners] Patch-, [] Volume-
    centroid_forms = (gradients[edge_heads] - gradients[edge_tails]) / 4     # [tet_edges] Plane+
    curls = 2 * (gradients[edge_tails] ^ gradients[edge_heads])              # [tet_edges] Field+
    # Wedging each edge's centroid 1-form with another edge's constant 2-form curl gives the helicity 3-form;
    # pairing with the signed volume cancels the geometry on every opposite-edge pair.
    densities = ((centroid_forms[:, None] ^ curls[None, :]) + (centroid_forms[None, :] ^ curls[:, None])) / 2  # [tet_edges, tet_edges] Trivector+
    return densities & volume, densities                                     # [tet_edges, tet_edges] Scalar+, [tet_edges, tet_edges] Trivector+


class Operators:
    """The Whitney reconstructions of potentials, edge amplitudes and fluxes, the curl, and the magnetic energy and
    helicity on a tetrahedral mesh."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh, self.vertices = mesh, vertices
        self.corners = mesh.incidence[3, 0] * vertices                       # [corners] Point-[tets]
        self.centres = self.corners.sum(axis=0) / 4                          # Point-[tets]
        self.volumes = ((self.corners[1] - self.corners[0]) & (self.corners[2] - self.corners[0]) & (self.corners[3] - self.corners[0])) / 6  # Volume-[tets]
        # The prism each tetrahedron sweeps in a unit of time; its size is the tetrahedron's volume.
        self.prisms = self.volumes & time                                    # Scalar-[tets]
        self.face_patches = ((self.corners[face_second] - self.corners[face_first]) & (self.corners[face_third] - self.corners[face_first])) / 2  # [corners] Patch-[tets]

        # Each corner hat's gradient is constant across its tetrahedron; assembled by corner, the gradients map
        # vertex potentials to piecewise-constant tetrahedral gradients without any metric.
        self.corner_gradients = (self.face_patches & time) / (3 * self.prisms)  # [corners] Plane+[tets]
        self.gradient = (spdiag(self.corner_gradients * Scalar) * mesh.incidence[3, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[tets <- vertices]

        # At each tetrahedron centroid the four barycentric hats equal 1/4, so the edge 1-form
        # reduces to one quarter of the difference of its endpoints' corner gradients.
        centroid_1forms = (self.corner_gradients[edge_heads] - self.corner_gradients[edge_tails]) / 4  # [tet_edges] Plane+[tets]
        # A tet's edges run from its lower local corner to its higher, the mesh's edges from the lower vertex to the
        # higher: `mesh.oriented[3, 1]` reads each mesh edge with the sign of their agreement.
        self.whitney1 = (spdiag(centroid_1forms * Scalar) * mesh.oriented[3, 1]).sum(axis=0)  # (Plane+ <- Scalar+)[tets <- edges]

        # The exterior derivative of each edge's Whitney 1-form is twice the wedge of its endpoints' gradients,
        # matching the face Whitney 2-form evaluated at the centroid on the edges' circulations around each face.
        curls = 2 * (self.corner_gradients[edge_tails] ^ self.corner_gradients[edge_heads])  # [tet_edges] Field+[tets]
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
