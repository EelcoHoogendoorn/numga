"""Finite conductivity moves conserved charge through an electric potential field.

Electric displacement determines charge, while conduction determines its rate of loss.
Both responses pair with the same spacetime field. Insulating gaps isolate each body's
initial charge; the conducting interior relaxes towards a constant potential.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import reduce
from operator import and_

from numga import Algebra, NumpyContext, stack
from numga.sparse import spdiag
from numga.topology import Topology

ga = Algebra("x+y+t-w0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Field = ga.gatype.bivector()                        # Bivector+
origin = mv.w.dual()                               # [] Point-
time = -mv.t.dual()                                # [] Point-
axes = stack((mv.x.dual(), mv.y.dual()))            # [axes] Point-
sweeps = axes & time                              # [axes] Excitation-
electric_response = (sweeps * (Field & sweeps)).sum(axis=0)  # [] Excitation- <- Field+


def simplices(topology: Topology, vertices: Point) -> tuple[Scalar, Scalar, Field]:
    """Each face's area, each vertex's cell, and the field of each corner's unit potential in its face."""
    corners = topology.incidence[2, 0]
    mean = corners.mean(axis=0)
    spans = reduce(and_, corners * vertices, time)
    areas = spans.norm() / 2                       # Scalar+[faces]
    cells = mean.adjugate() * areas                # Scalar+[vertices]
    facing = topology.off[2, 0] * vertices
    complements = facing[0] & facing[1] & time
    hats = complements / (complements & (corners * vertices))
    gradients = hats - (hats & origin) * mv.w      # [corners] Plane+[faces]
    hat_fields = -(gradients ^ mv.t)              # [corners] Field+[faces]
    return areas, cells, hat_fields


def relaxation(topology: Topology, vertices: Point, fractions: Scalar, boundary: Scalar,
               permittivity: float, conductivities, body_charges, frame_step: float,
               n_substeps: int) -> Iterator[tuple[Scalar, Scalar]]:
    """The evolving potential and integrated charge occupy the same mesh vertices."""
    corners = topology.incidence[2, 0]
    mean = corners.mean(axis=0)
    areas, cells, hat_fields = simplices(topology, vertices)

    # Displacement stores electric energy; conduction dissipates it through the same pairing.
    local = areas * (hat_fields[:, None] & electric_response(hat_fields[None, :]))
    conductivity = (conductivities * (mean * fractions)).sum(axis=0)
    charge_from_voltage = (corners[:, None].adjugate() * spdiag(permittivity * local)
                           * corners[None, :]).sum(axis=0).sum(axis=0)
    outflow_from_voltage = (corners[:, None].adjugate() * spdiag(conductivity * local)
                            * corners[None, :]).sum(axis=0).sum(axis=0)

    # Only the remote boundary voltage is held; every interior voltage remains independent.
    extension = topology.reading(topology.interior[0], 0).adjugate()
    poisson = (extension.adjugate() * charge_from_voltage * extension) * Scalar
    parcels = cells * fractions
    charge = (body_charges * parcels / parcels.sites.sum()).sum(axis=0)
    held = boundary[0]
    potential = held + extension * poisson.solve(extension.adjugate() * (charge - charge_from_voltage * held))
    yield potential, charge

    step = frame_step / n_substeps
    for target in boundary[1:]:
        previous = held
        for substep in range(n_substeps):
            # Equal and opposite transfers conserve each isolated body's initial charge.
            charge = charge - step * (outflow_from_voltage * potential)
            held = previous + ((substep + 1) / n_substeps) * (target - previous)
            potential = held + extension * poisson.solve(extension.adjugate() * (charge - charge_from_voltage * held))
        yield potential, charge


def densities_and_forces(topology: Topology, vertices: Point,
                         states: Iterable[tuple[Scalar, Scalar]]) -> Iterator[tuple[Scalar, Scalar, Point]]:
    """Each state's potential, with its charge density at the vertices and the force on a unit charge at rest in each face."""
    corners = topology.incidence[2, 0]
    areas, cells, hat_fields = simplices(topology, vertices)
    for potential, charge in states:
        field = (hat_fields * (corners * potential)).sum(axis=0)  # Field+[faces]
        yield potential, charge / cells, field.commutator(time)
