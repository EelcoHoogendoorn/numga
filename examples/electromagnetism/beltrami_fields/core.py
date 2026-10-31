"""Beltrami fields carry magnetic linkage with stationary energy at fixed helicity.

Magnetostatics in spacetime, `x+y+z+t-`: the torus rests in one instant, its points are directions of space, and the magnetic field is a plane of space that reads its flux through an oriented patch as `field & patch`. A hat is constant in time, so its gradient is the volume its opposite face sweeps through time over the prism the tetrahedron sweeps, `(face_patches & time) / (3 * prism)`. Whitney edge forms reconstruct the potential, and their curls give the field. Energy pairs the field with its dual; helicity pairs the potential and its curl with signed volume. Their adjugates carry both readings back to edge amplitudes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga.algebras import STA as ga
from numga.sparse import spdiag
from examples.electromagnetism.beltrami_fields.plumbing import Mesh, as_scalar, mv, time

Scalar = ga.gatype.scalar()          # Scalar+
Plane = ga.gatype.vector()           # Vector+
# The magnetic field, a plane of space.
Field = ga.gatype.bivector().derive.within("x y z")  # Bivector+
Point = ga.gatype.antivector()        # Antivector-
Patch = ga.gatype.antibivector()      # Antibivector-
Volume = ga.gatype.antitrivector()    # Antitrivector-
Amount = ga.gatype.antiscalar()       # Antiscalar-
Energy = ga.gatype((Amount, Scalar))
Curl = ga.gatype((Field, Scalar))


# --- math -----------------------------------------------------------------------------
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
        tails, heads = mesh.corners[3, 1]
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
