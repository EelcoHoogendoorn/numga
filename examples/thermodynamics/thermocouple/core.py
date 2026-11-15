"""Coupled charge and entropy transport drives the Seebeck and Peltier effects across a triangulated two-metal thermocouple.

Conduction electrons carry both electric charge and entropy, whereas lattice vibrations carry entropy alone, and the electrons in the two metals carry different entropies per unit charge. When the junction is heated with open ends, each leg builds up the electric field needed to hold its electrons still against the temperature fall, leaving the difference between the two legs as a voltage across the ends. When an electric current passes through the junction instead, electrons arrive from one metal carrying one entropy and depart through the other carrying a different entropy, cooling or warming the junction according to the current sense.

Planar projective spacetime is joined with a degenerate direction for entropy. Unit charge is `-t.dual()` along time and unit entropy is `s.dual()`, combining into each carrier's extensive parcel. The electrochemical potential along time and the temperature along entropy form one `Potential` 1-form whose spatial fall `-gradients ^ Potential` is a driving bivector. Joining a carrier velocity with its parcel defines the advection map `Velocity.regressive(parcels)` from velocities to fluxes, and its adjugate maps a drive to the mechanical force on the carrier so `drive & advection(velocity) == force & velocity`. Composing advection with a symmetric mobility yields a self-adjugate material response `advection(mobility(advection.adjugate()))` with non-negative dissipation `drive & response(drive)`.

Across the triangulated hairpin, triangle simplex gradients pull the local responses back to the symmetric vertex conductance `gradient.adjugate()((areas * responses)(gradient))`. Because both vertex conductance and lumped dual-cell entropy storage equal their own adjugates, heating the junction raises the open-circuit foot voltage per unit heating by the same amount that drawing current lowers the junction temperature per unit current at every instant. Each dual cell stores entropy proportional to temperature rise without storing net charge, so temperature diffuses over time while the electrochemical potential adjusts instantaneously within linear response around equilibrium.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

# Planar projective spacetime joined with a metric-free direction for entropy.
ga = Algebra("x+y+t-w0") * Algebra("s0")
space = Algebra("x+y+")
quantities = Algebra("t-s0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
Parcel = Plane.derive.within(quantities).derive.dual                          # Antivector-
Velocity = Plane.derive.within(space).derive.dual                             # Antivector-
Force = Plane.derive.within(space)                                            # Vector+
Potential = Plane.derive.within(quantities)                                   # Vector+
Drive = ga.gatype.bivector().derive.touching(space, quantities)               # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
Response = ga.gatype((Flux, Drive))                                           # Antibivector- <- Bivector+
origin = mv.w.dual()                                                          # [] Point-
time = -mv.t.dual()                                                           # [] Parcel-
entropy = mv.s.dual()                                                         # [] Parcel-


# --- math -----------------------------------------------------------------------------
def simplex_gradients(topology: Topology, vertices: Point) -> tuple[Scalar, Force]:
    """Opposite-edge joins on `topology` determine the triangle areas (`Scalar+[F]`) and corner simplex gradients (`[3] Force+[F]`) from unit-weight `vertices` (`Point-[V]`)."""
    corners = topology.incidence[2, 0] * vertices                             # [3] Point-[F]
    facing = topology.off[2, 0] * vertices                                    # [2, 3] Point-[F]
    sides = facing[0] & facing[1]                                             # [3] Antibivector-[F]
    # Joining each opposite side with time and entropy forms a hyperplane whose quotient by the corner join equals one
    # at that corner and zero on the opposite side; casting to Force drops the origin offset along w.
    gradients = ((sides & time & entropy) / (sides & time & entropy & corners)).cast(Force)   # [3] Force+[F]
    areas = (corners[0] & corners[1] & corners[2] & time & entropy).norm() / 2                # Scalar+[F]
    return areas, gradients


def response(charges: np.ndarray, entropies: np.ndarray, mobilities: np.ndarray) -> Response:
    """Carrier charges, entropies, and scalar mobilities sum over the trailing carrier axis into the symmetric material response (`[...] Flux- <- Drive+`)."""
    parcels = charges * time + entropies * entropy                            # [...] Parcel-
    # Joining a carrier velocity with its parcel forms its flux, while the adjugate maps a drive to the carrier force.
    advection = Velocity.regressive(parcels)                                  # [...] Flux- <- Velocity-
    mobility = mv.scalar(mobilities[..., None]) * Force.dual()                # [...] Velocity- <- Force+
    return advection(mobility(advection.adjugate())).sum(axis=-1)


def conductance(topology: Topology, areas: Scalar, gradients: Force, responses: Response) -> SparseExtensor:
    """Triangle simplex gradients pull the area-weighted material responses (`(Flux- <- Drive+)[F]`) back to the symmetric vertex conductance (`(Parcel- <- Potential+)[V <- V]`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    # Wedging each corner's simplex gradient with the vertex potential forms the negated drive on every triangle.
    gradient = (spdiag(gradients ^ Potential) * corners).sum(axis=0)          # (Drive+ <- Potential+)[F <- V]
    return gradient.adjugate()((areas * responses)(gradient))


def storage(topology: Topology, areas: Scalar, capacities: np.ndarray) -> SparseExtensor:
    """Lumped dual-cell areas and scalar `capacities` (`[V]`) form the diagonal vertex entropy-storage operator (`(Parcel- <- Potential+)[V <- V]`)."""
    dual_areas = (topology.incidence[2, 0].adjugate() * (areas / 3)).sum(axis=0)   # Scalar+[V]
    # Pairing a potential with unit entropy reads the temperature rise and stores proportional entropy without charge.
    unit_capacity = entropy * (Potential & entropy)                           # [] Parcel- <- Potential+
    return spdiag(dual_areas * mv.scalar(capacities[:, None]).field() * unit_capacity)


def settling_potentials(
    topology: Topology,
    loss_map: SparseExtensor,
    storage_map: SparseExtensor,
    held_vertices: np.ndarray,
    sources: Parcel,
    time_step: float,
    n_frames: int,
) -> Iterator[Potential]:
    """Yields the vertex potentials (`Potential+[V]`) across `n_frames` backward-Euler steps of `time_step` from equilibrium with `held_vertices` pinned to zero."""
    V, E, F = topology.n_elements
    free = topology.reading(np.setdiff1d(np.arange(V), held_vertices), 0)     # Scalar+[V_free <- V]
    step_map = free * (loss_map + storage_map / time_step) * free.adjugate()   # (Parcel- <- Potential+)[V_free <- V_free]
    # Widen sources and initial potentials to their full charge-and-entropy layouts before stepping.
    zero_field = 0 * ((mv.t + mv.s) & sources)                                # Scalar+[V]
    sources = sources + zero_field * (time + entropy)                         # Parcel-[V]
    potentials = zero_field * (mv.t + mv.s)                                   # Potential+[V]
    for _ in range(n_frames):
        yield potentials
        potentials = free.adjugate() * step_map.solve(free * (sources + storage_map(potentials) / time_step))
