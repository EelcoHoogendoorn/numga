"""Finite conductivity moves conserved charge through an electric potential field.

Electric displacement determines the charge present, while conduction determines its rate
of loss. Both responses pair with the same spacetime field. Insulating gaps isolate each
body's initial charge as the conducting interior relaxes towards a constant potential.
"""
from __future__ import annotations

from collections.abc import Iterator
import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import Sparse, spdiag
from numga.topology import Topology

ga = Algebra("x+y+t-w0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                     # Scalar+
Upright = ga.gatype.from_blades(("x", "y"))                     # Vector+
Point = ga.gatype.antivector()                                  # Antivector-
Field = ga.gatype.bivector()                                    # Bivector+
Excitation = ga.gatype.antibivector()                           # Antibivector-
origin = mv.w.dual()                                            # Point-
time = -mv.t.dual()                                             # Point-
axes = stack((mv.x.dual(), mv.y.dual()))                        # [2] Point-
sweeps = axes & time                                            # [2] Excitation-
electric_response = (sweeps * (Field & sweeps)).sum(axis=0)     # [] Excitation- <- Field+


def triangle_geometry(topology: Topology, vertices: Point) -> tuple[Scalar, Scalar, Field]:
    """Triangle areas (`Scalar+[F]`), vertex dual areas (`Scalar+[V]`), and simplex-gradient electric fields (`[3] Field+[F]`) come from the vertex positions."""
    corners = topology.incidence[2, 0]                          # [3] Scalar+[F <- V]
    corner_mean = corners.mean(axis=0)                          # Scalar+[F <- V]
    corner_vertices = corners * vertices                        # [3] Point-[F]
    double_areas = (corner_vertices[0] & corner_vertices[1] & corner_vertices[2] & time).norm()  # Scalar+[F]
    areas = double_areas / 2                                    # Scalar+[F]
    dual_areas = corner_mean.adjugate() * areas                 # Scalar+[V]
    facing = topology.off[2, 0] * vertices                      # [2, 3] Point-[F]
    sides = facing[0] & facing[1]                               # [3] Excitation-[F]
    gradients = ((sides & time) / (sides & time & corner_vertices)).cast(Upright)  # [3] Upright+[F]
    corner_fields = -(gradients ^ mv.t)                         # [3] Field+[F]
    return areas, dual_areas, corner_fields


def surface_operators(topology: Topology, areas: Scalar, corner_fields: Field, fractions: Scalar,
                      permittivity: float, conductivities: np.ndarray) -> tuple[Sparse, Sparse, Sparse, Scalar]:
    """Permittivity and conductivity weight the same electric-field pairing to form the vertex charge and outflow maps, with interior Poisson operator `(Scalar+ <- Scalar+)[V_interior <- V_interior]`."""
    corners = topology.incidence[2, 0]                          # [3] Scalar+[F <- V]
    corner_mean = corners.mean(axis=0)                          # Scalar+[F <- V]
    # Displacement stores electric energy while conduction dissipates it through the same field pairing.
    local_energy = areas * (corner_fields[:, None] & electric_response(corner_fields[None, :]))  # [3, 3] Scalar+[F]
    conductivity = (conductivities * (corner_mean * fractions)).sum(axis=0)  # Scalar+[F]
    charge_map = (corners[:, None].adjugate() * spdiag(permittivity * local_energy)
                  * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[V <- V]
    outflow_map = (corners[:, None].adjugate() * spdiag(conductivity * local_energy)
                   * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[V <- V]
    extension = topology.reading(topology.interior[0], 0).adjugate()  # Scalar+[V <- V_interior]
    poisson = (extension.adjugate() * charge_map * extension) * Scalar  # (Scalar+ <- Scalar+)[V_interior <- V_interior]
    return charge_map, outflow_map, extension, poisson


def relaxation_states(topology: Topology, vertices: Point, fractions: Scalar, boundary: Scalar,
                      permittivity: float, conductivities: np.ndarray, body_charges: np.ndarray,
                      frame_step: float, n_substeps: int) -> Iterator[tuple[Scalar, Scalar, Scalar, Point]]:
    """Yields the potential (`Scalar+[V]`), integrated vertex charge (`Scalar+[V]`), charge density (`Scalar+[V]`), and rest-charge force (`Point-[F]`) across the relaxation."""
    corners = topology.incidence[2, 0]                          # [3] Scalar+[F <- V]
    areas, dual_areas, corner_fields = triangle_geometry(topology, vertices)
    charge_map, outflow_map, extension, poisson = surface_operators(
        topology, areas, corner_fields, fractions, permittivity, conductivities)

    # Only the outer boundary voltage is held; every interior vertex voltage solves Gauss's law.
    parcels = dual_areas * fractions                            # [2] Scalar+[V]
    charge = (body_charges * parcels / parcels.sites.sum()).sum(axis=0)  # Scalar+[V]
    boundary_voltage = boundary[0]                              # Scalar+[V]
    potential = boundary_voltage + extension * poisson.solve(
        extension.adjugate() * (charge - charge_map * boundary_voltage))  # Scalar+[V]
    field = (corner_fields * (corners * potential)).sum(axis=0)  # Field+[F]
    yield potential, charge, charge / dual_areas, field.commutator(time)

    substep_dt = frame_step / n_substeps
    for target_voltage in boundary[1:]:
        previous_voltage = boundary_voltage
        for substep in range(n_substeps):
            # Equal and opposite conduction transfers conserve each isolated body's initial charge.
            charge = charge - substep_dt * (outflow_map * potential)  # Scalar+[V]
            boundary_voltage = previous_voltage + ((substep + 1) / n_substeps) * (target_voltage - previous_voltage)  # Scalar+[V]
            potential = boundary_voltage + extension * poisson.solve(
                extension.adjugate() * (charge - charge_map * boundary_voltage))  # Scalar+[V]
        field = (corner_fields * (corners * potential)).sum(axis=0)  # Field+[F]
        yield potential, charge, charge / dual_areas, field.commutator(time)
