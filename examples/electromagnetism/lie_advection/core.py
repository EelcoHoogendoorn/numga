"""Magnetic circulation cochains evolve across a fixed planar triangulation in 2+1D spacetime.

Each moving edge sweeps a world sheet `(time + velocity) & edge` through spacetime, and pairing the electromagnetic bivector field with that sheet gives the rate of change of the edge's circulation reading. Under ideal Ohm's law the field vanishes on comoving world sheets, so a fixed edge's reading changes by the magnetic flux across the spatial patch between its fixed and comoving sheets, while interior edge resistance dissipates magnetic energy through the curl's adjugate.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.algebras import STA2D as ga
from numga.sparse import spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Upright = Plane.derive.within(Algebra("x+y+"))                               # Vector+
Field = ga.gatype.bivector()                                                 # Bivector+
# The magnetic field, a spatial bivector plane.
Magnetic = Field.derive.within(Algebra("x+y+"))                              # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-
Antiscalar = ga.gatype.antiscalar()                                          # Antiscalar-
ScalarMap = ga.gatype((Scalar, Scalar))                                      # Scalar+ <- Scalar+
CurlMap = ga.gatype((Magnetic, Scalar))                                      # Magnetic+ <- Scalar+
StiffnessMap = ga.gatype((Antiscalar, Scalar))                               # Antiscalar- <- Scalar+
triangle = Topology(3, [[0, 1, 2]])

# The unit future direction of an observer at rest in the spatial frame, pairing to 1 with `mv.t`.
time = -mv.t.dual()                                                          # [] Point-


# --- math -----------------------------------------------------------------------------
def cartan_decomposition(corners: Point, corner_potentials: Plane,
                         corner_velocities: Point) -> tuple[Scalar, Scalar, Patch, Field]:
    """A moving triangle's edge readings change by endpoint potential differences (`[3] Scalar+[1]`) plus world-sheet fluxes (`[3] Scalar+[1]`) from the world sheets (`[3] Patch-[1]`) and face field (`Field+[1]`)."""
    sides = triangle.boundary[2, 1] * (triangle.d[0] * corners)              # [3] Point-[1]
    patch = (sides[0] & sides[1]) / 2                                        # Patch-[1]
    # Each vertex function is constant in time, so its simplex gradient is the plane through the
    # opposite side and time divided by twice the spacetime prism swept per unit time.
    gradients = (time & sides) / (2 * (time & patch))                        # [3] Plane+[1]
    face_potentials = triangle.incidence[2, 0] * corner_potentials           # [3] Plane+[1]
    face_field = (gradients ^ face_potentials).sum(axis=0)                   # Field+[1]
    # Each corner advances through spacetime along its world line `time + velocity` per unit time.
    corner_motions = time + corner_velocities                                # Point-[3]
    interior_readings = corner_potentials & corner_motions                   # Scalar+[3]
    endpoint_differences = triangle.boundary[2, 1] * (triangle.d[0] * interior_readings)  # [3] Scalar+[1]
    edge_motions = triangle.incidence[2, 1] * (triangle.incidence[1, 0].mean(axis=0) * corner_motions)  # [3] Point-[1]
    world_sheets = edge_motions & sides                                      # [3] Patch-[1]
    # Pairing the face field with each edge's world sheet gives the magnetic flux crossing the moving edge.
    swept_fluxes = face_field & world_sheets                                 # [3] Scalar+[1]
    return endpoint_differences, swept_fluxes, world_sheets, face_field


def vortex_streamfunction(vertices: Point, swirl_rate: float, ellipticity: float) -> Scalar:
    """An elliptical shearing vortex has streamfunction (`Scalar+[V]`) vanishing smoothly in value and slope on the unit circle."""
    abscissa, ordinate = mv.x & vertices, mv.y & vertices                    # Scalar+[V], Scalar+[V]
    radial_envelope = (1 - vertices.squared()).squared()                     # Scalar+[V]
    quadrupole = 1 + ellipticity * (abscissa.squared() - ordinate.squared()) # Scalar+[V]
    return swirl_rate * radial_envelope * quadrupole                         # Scalar+[V]


def initial_cochain(topology: Topology, vertices: Point, lobe_centres: Point,
                    lobe_strengths: np.ndarray, lobe_radius: float) -> Scalar:
    """Localized Gaussian magnetic potentials around `lobe_centres` pair with oriented edges to form the initial circulation cochain (`Scalar+[E]`)."""
    edges = topology.d[0] * vertices                                         # Point-[E]
    edge_midpoints = topology.incidence[1, 0].mean(axis=0) * vertices        # Point-[E]
    displacements = edge_midpoints - lobe_centres                            # [...] Point-[E]
    # Pairing the spatial bivector with the displacement from each lobe centre forms a circulating potential plane.
    envelopes = (-displacements.squared() / (2 * lobe_radius**2)).exp()      # [...] Scalar+[E]
    potential = ((envelopes * lobe_strengths) * (mv.xy & displacements)).sum(axis=0)  # Plane+[E]
    return potential & edges                                                 # Scalar+[E]


def whitney_curl(topology: Topology, vertices: Point) -> CurlMap:
    """Corner simplex gradients wedge in pairs across oriented edges to form the piecewise-constant Whitney magnetic 2-form curl (`(Magnetic+ <- Scalar+)[F <- E]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    prisms = time & ((sides[0] & sides[1]) / 2)                              # Scalar-[F]
    corner_gradients = (time & sides) / (2 * prisms)                         # [3] Plane+[F]
    # Each edge's Whitney 1-form has exterior derivative equal to twice the wedge of its endpoint simplex gradients.
    tails, heads = topology.corners[2, 1]
    curls = 2 * (corner_gradients[tails] ^ corner_gradients[heads])          # [3] Magnetic+[F]
    return (spdiag(curls * Scalar) * topology.oriented[2, 1]).sum(axis=0)    # (Magnetic+ <- Scalar+)[F <- E]


def swept_edge_patches(topology: Topology, vertices: Point, stream: Scalar) -> Patch:
    """Fixed and comoving edge world sheets under `stream` differ by the swept spatial patches (`Patch-[E]`)."""
    edges = topology.d[0] * vertices                                         # Point-[E]
    sides = topology.boundary[2, 1] * edges                                  # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    prisms = time & patches                                                  # Scalar-[F]
    areas = prisms.norm()                                                    # Scalar+[F]
    normals = patches / areas                                                # Patch-[F]
    corner_gradients = (time & sides) / (2 * prisms)                         # [3] Plane+[F]
    gradient = (spdiag(corner_gradients * Scalar) * topology.incidence[2, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[F <- V]
    face_velocity = gradient(stream) ^ normals                               # Point-[F]
    face_to_edge = topology.incidence[2, 1].sum(axis=0).adjugate()           # Scalar+[E <- F]
    edge_counts = face_to_edge * (areas / areas)                             # Scalar+[E]
    to_edges = spdiag(1 / edge_counts) * face_to_edge                        # (Scalar+ <- Scalar+)[E <- F]
    edge_velocity = (to_edges * Point)(face_velocity)                        # Point-[E]
    fixed_sheets = time & edges                                              # Patch-[E]
    comoving_sheets = (time + edge_velocity) & edges                         # Patch-[E]
    return fixed_sheets - comoving_sheets                                    # Patch-[E]


def advection_map(topology: Topology, vertices: Point, curl: CurlMap,
                  swept_patches: Patch) -> ScalarMap:
    """Adjacent face magnetic fields average onto edges and pair with `swept_patches` to form the Lie advection map (`(Scalar+ <- Scalar+)[E <- E]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    areas = (time & ((sides[0] & sides[1]) / 2)).norm()                      # Scalar+[F]
    # Averaging adjacent face fields onto each edge preserves uniform fields and skew-symmetry in magnetic energy.
    face_to_edge = topology.incidence[2, 1].sum(axis=0).adjugate()           # Scalar+[E <- F]
    edge_counts = face_to_edge * (areas / areas)                             # Scalar+[E]
    to_edges = spdiag(1 / edge_counts) * face_to_edge                        # (Scalar+ <- Scalar+)[E <- F]
    edge_curl = (to_edges * Magnetic)(curl)                                  # (Magnetic+ <- Scalar+)[E <- E]
    return (Magnetic & swept_patches)(edge_curl)                             # (Scalar+ <- Scalar+)[E <- E]


def curl_stiffness_map(topology: Topology, vertices: Point, curl: CurlMap) -> StiffnessMap:
    """Pulling the area-weighted vacuum duality map back through `curl` forms the magnetic stiffness (`(Antiscalar- <- Scalar+)[E <- E]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    areas = (time & ((sides[0] & sides[1]) / 2)).norm()                      # Scalar+[F]
    vacuum = areas * Magnetic.dual()                                         # (Patch- <- Magnetic+)[F]
    return curl.adjugate()(vacuum(curl))                                     # (Antiscalar- <- Scalar+)[E <- E]


def curl_laplacian_map(topology: Topology, curl_stiffness: StiffnessMap) -> ScalarMap:
    """Restricting unit-resistance magnetic diffusion to interior edges keeps boundary circulation fixed (`(Scalar+ <- Scalar+)[E <- E]`)."""
    interior = topology.reading(topology.interior[1], 1).adjugate() * Scalar # (Scalar+ <- Scalar+)[E <- E_interior]
    unit_resistance = Antiscalar.dual_inverse()                              # [] Scalar+ <- Antiscalar-
    return interior(unit_resistance(interior.adjugate()(curl_stiffness)))    # (Scalar+ <- Scalar+)[E <- E]


def flux_density_map(topology: Topology, vertices: Point) -> ScalarMap:
    """Dividing face boundary sums by triangle areas maps edge cochains to face magnetic flux densities (`(Scalar+ <- Scalar+)[F <- E]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    areas = (time & ((sides[0] & sides[1]) / 2)).norm()                      # Scalar+[F]
    return spdiag(1 / areas) * (topology.d[1] * Scalar)                      # (Scalar+ <- Scalar+)[F <- E]


def cochain_trajectory(initial: Scalar, rate: ScalarMap, dt: float, n_steps: int) -> Iterator[Scalar]:
    """Yields edge circulation cochains (`[...] Scalar+[E]`) across `n_steps` fourth-order Runge-Kutta steps."""
    cochain = initial                                                        # [...] Scalar+[E]
    yield cochain
    for _ in range(n_steps):
        first_rate = rate(cochain)                                           # [...] Scalar+[E]
        second_rate = rate(cochain + (dt / 2) * first_rate)                  # [...] Scalar+[E]
        third_rate = rate(cochain + (dt / 2) * second_rate)                  # [...] Scalar+[E]
        fourth_rate = rate(cochain + dt * third_rate)                        # [...] Scalar+[E]
        cochain = cochain + (dt / 6) * (first_rate + 2 * second_rate + 2 * third_rate + fourth_rate)  # [...] Scalar+[E]
        yield cochain
