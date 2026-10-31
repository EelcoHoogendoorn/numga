"""Lie advection carries magnetic flux across a fixed mesh, in the spacetime of the plane, `x+y+t-`.

An edge's circulation reading changes at the rate the field reads on the world sheet the edge sweeps through spacetime, `edge_motions & sides`, each motion `time + velocity`. The magnetic field is the plane `xy` of space and reads only the sheet's spatial part; an electric field is a plane through time. Ideal Ohm's law is the field vanishing on the world sheets of edges that move with the conductor, so a fixed edge's reading changes by the magnetic flux through the patch between its own world sheet and the comoving one. Endpoint contributions cancel around a closed boundary. A prescribed flow gives one evolution map on edge readings; resistivity dissipates the field through the curl's adjugate.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.algebras import STA2D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
# The magnetic field, a plane of space.
Magnetic = Bivector.derive.within("x y")                                     # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-
Amount = ga.gatype.antiscalar()                                              # Antiscalar-
Evolution = ga.gatype((Scalar, Scalar))
triangle = Mesh(3, [[0, 1, 2]])

# The direction in time of an observer at rest with the mesh, which pairs to one with `mv.t`.
time = -mv.t.dual()                                                          # [] Point-


# --- math -----------------------------------------------------------------------------
def cartan_decomposition(corners: Point, corner_forms: Plane, corner_velocities: Point
                         ) -> tuple[Scalar, Scalar, Patch, Bivector]:
    """The endpoint potential differences, world-sheet magnetic fluxes, edge world sheets,
    and exterior-derivative 2-form for a linear 1-form on a moving triangle."""
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]  # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    # Each hat is constant in time: its gradient is the plane through the opposite side and time,
    # over twice the prism the triangle sweeps in a unit of time.
    gradients = (time & sides) / (2 * (time & patch))                        # [corners] Plane+
    # Wedging each corner hat gradient with its corner 1-form sums to the constant 2-form curl on the face.
    face_bivector = (gradients ^ corner_forms).sum(axis=0)                   # [] Bivector+
    # Each corner advances through spacetime along its world line, `time + velocity` per unit time.
    corner_motions = time + corner_velocities                                # [corners] Point-
    interior_readings = corner_forms & corner_motions                        # [corners] Scalar+
    endpoint_differences = (triangle.boundary[2, 1] * (triangle.d[0] * interior_readings.field())).batch()[:, 0]  # [corners] Scalar+
    edge_motions = (triangle.incidence[2, 1] * (triangle.incidence[1, 0].mean(axis=0) * corner_motions.field())).batch()[:, 0]  # [corners] Point-
    world_sheets = edge_motions & sides                                      # [corners] Patch-
    # Pairing the 2-form curl with each edge's world sheet gives the magnetic flux crossing the moving edge.
    swept_fluxes = face_bivector & world_sheets                              # [corners] Scalar+
    return endpoint_differences, swept_fluxes, world_sheets, face_bivector


class Operators:
    """Topological exterior derivatives, Whitney reconstruction maps, and curl-curl operators on a triangle mesh."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh, self.vertices = mesh, vertices
        self.d1 = mesh.d[1] * Scalar                                         # (Scalar+ <- Scalar+)[faces <- edges]
        self.edges = mesh.d[0] * vertices                                    # Point-[edges]
        self.edge_midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices   # Point-[edges]
        self.sides = mesh.boundary[2, 1] * self.edges                        # [corners] Point-[faces]
        self.patches = (self.sides[0] & self.sides[1]) / 2                   # Patch-[faces]
        # The prism each face sweeps in a unit of time; its size is the face's area.
        self.prisms = time & self.patches                                    # Scalar-[faces]
        self.areas = self.prisms.norm()                                      # Scalar+[faces]
        self.flux_density = spdiag(1 / self.areas) * self.d1                 # (Scalar+ <- Scalar+)[faces <- edges]
        self.normals = self.patches / self.areas                             # Patch-[faces]

        # Each corner hat's gradient is constant across its face; assembling by corner maps
        # vertex streamfunctions to piecewise-constant face gradients without any metric.
        self.corner_gradients = (time & self.sides) / (2 * self.prisms)      # [corners] Plane+[faces]
        self.gradient = (spdiag(self.corner_gradients * Scalar) * mesh.incidence[2, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[faces <- vertices]

        # The exterior derivative of each edge's Whitney 1-form is twice the wedge of its endpoints'
        # gradients, matching the constant Whitney 2-form of unit integral over the face patch.
        tails, heads = mesh.corners[2, 1]
        curls = 2 * (self.corner_gradients[tails] ^ self.corner_gradients[heads])  # [corners] Magnetic+[faces]
        self.curl = (spdiag(curls * Scalar) * mesh.oriented[2, 1]).sum(axis=0)  # (Magnetic+ <- Scalar+)[faces <- edges]

        # Averaging adjacent face fields onto each edge preserves exactness of uniform 2-forms and
        # makes the ideal Lie advection rate skew-symmetric in the 2-form magnetic energy.
        face_to_edge = ~mesh.incidence[2, 1].sum(axis=0)                     # Scalar+[edges <- faces]
        edge_counts = face_to_edge * as_scalar(np.ones(len(mesh.faces))).field()  # Scalar+[edges]
        to_edges = spdiag(1 / edge_counts) * face_to_edge                    # Scalar+[edges <- faces]
        self.edge_velocity_map = to_edges * Point                            # (Point- <- Point-)[edges <- faces]
        self.edge_curl = (to_edges * Magnetic)(self.curl)                    # (Magnetic+ <- Scalar+)[edges <- edges]

        # The vacuum takes each face's field to the excitation it causes, weighted by the face's area; pulled
        # back through the curl, it gives the curl stiffness, from edge readings to the edge currents.
        vacuum = spdiag(self.areas * Magnetic.dual())                        # (Patch- <- Magnetic+)[faces <- faces]
        self.curl_stiffness = self.curl.adjugate()(vacuum(self.curl))        # (Amount- <- Scalar+)[edges <- edges]
        # A unit resistance on each interior edge turns its current into the resistive electric field, a plane
        # through time, read on the edge's world sheet: a rate of its reading. Restricting the response to
        # interior edges keeps boundary circulation fixed under resistivity.
        interior = ~SparseExtensor.selection(mv.context, mesh.interior_edges, len(mesh.edges)) * Scalar  # (Scalar+ <- Scalar+)[edges <- interior]
        unit_resistance = spdiag(as_scalar(np.ones(len(mesh.interior_edges))).field() * Amount.dual_inverse())  # (Scalar+ <- Amount-)[interior <- interior]
        self.curl_laplacian = interior(unit_resistance(interior.adjugate()(self.curl_stiffness)))  # (Scalar+ <- Scalar+)[edges <- edges]


def vortex_streamfunction(vertices: Point, swirl_rate: float, ellipticity: float) -> Scalar:
    """Streamfunction of an elliptical shearing vortex that vanishes smoothly on the unit circle."""
    abscissa, ordinate = mv.x & vertices, mv.y & vertices                    # Scalar+[vertices], Scalar+[vertices]
    radial_envelope = (1 - vertices.squared()).squared()                     # Scalar+[vertices]
    quadrupole = 1 + ellipticity * (abscissa.squared() - ordinate.squared())  # Scalar+[vertices]
    return swirl_rate * radial_envelope * quadrupole                         # Scalar+[vertices]


def initial_cochain(operators: Operators, lobe_centres: Point,
                    lobe_strengths: np.ndarray, lobe_radius: float) -> Scalar:
    """Initial edge 1-form cochain from localized circulating Gaussian potentials around `lobe_centres`."""
    displacements = operators.edge_midpoints.batch()[:, None] - lobe_centres[None, :]  # [edges, lobes] Point-
    # Pairing the unit bivector with the displacement from each lobe centre gives a circulating 1-form.
    envelopes = (-displacements.squared() / (2 * lobe_radius**2)).exp()      # [edges, lobes] Scalar+
    potential = ((envelopes * lobe_strengths) * (mv.xy & displacements)).sum(axis=-1).field()  # Plane+[edges]
    return potential & operators.edges                                       # Scalar+[edges]


def edge_world_sheets(operators: Operators, stream: Scalar) -> tuple[Point, Patch, Patch]:
    """The divergence-free face velocity, and the world sheets each mesh edge sweeps per unit time,
    held fixed and carried by the flow."""
    face_velocity = operators.gradient(stream) ^ operators.normals           # Point-[faces]
    edge_velocity = operators.edge_velocity_map(face_velocity)               # Point-[edges]
    fixed_sheets = time & operators.edges                                    # Patch-[edges]
    comoving_sheets = (time + edge_velocity) & operators.edges               # Patch-[edges]
    return face_velocity, fixed_sheets, comoving_sheets                      # Point-[faces], Patch-[edges], Patch-[edges]


def advect(initial: Scalar, rate: Evolution, dt: float, n_steps: int) -> Iterator[Scalar]:
    """Successive edge 1-form cochains along fourth-order Runge-Kutta steps."""
    cochain = initial                                                        # [cases] Scalar+[edges]
    yield cochain
    for _ in range(n_steps):
        first = rate(cochain)                                                # [cases] Scalar+[edges]
        second = rate(cochain + (dt / 2) * first)                            # [cases] Scalar+[edges]
        third = rate(cochain + (dt / 2) * second)                            # [cases] Scalar+[edges]
        fourth = rate(cochain + dt * third)                                  # [cases] Scalar+[edges]
        cochain = cochain + (dt / 6) * (first + 2 * second + 2 * third + fourth)  # [cases] Scalar+[edges]
        yield cochain
