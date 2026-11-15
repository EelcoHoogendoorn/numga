"""Two-carrier galvanomagnetic and thermomagnetic transport couples charge and heat across triangulated conductors in planar projective spacetime joined with a null entropy direction.

Electrons and holes carry opposite charges alongside positive entropy per carrier, so an electrochemical gradient pushes the two species in opposite directions while a temperature gradient pushes both toward the cooler contact. A perpendicular magnetic field deflects each drifting species through its Hall angle, turning opposite charges opposite ways. Along an insulated Hall bar, a single carrier species builds up a transverse Hall voltage that cancels the magnetic deflection and leaves the longitudinal resistance unchanged, whereas under open-circuit heating its Seebeck voltage holds the carriers at rest. At carrier compensation, as in a semimetal such as bismuth, the opposite transverse charge currents cancel so no Hall voltage forms, magnetoresistance rises with the field, and both species carry entropy to the same edge in the Ettingshausen effect; under a thermal gradient the field parts electrons and holes to opposite edges to produce a peak transverse Nernst voltage. Around a Corbino ring, closed rim contacts prevent transverse charge accumulation so carriers spiral at every balance and radial heating drives a circulating azimuthal current. In a magnetized conductor with zero external magnetic field, spin-dependent skew scattering deflects spin-up and spin-down electrons through opposite angles in the spin plane, producing an anomalous Hall voltage when magnetized and a pure transverse spin Hall current when unmagnetized.

In planar projective spacetime with degenerate point weight `w` joined with a null entropy direction `s`, each carrier species is modelled as an intrinsic `Parcel` `valence * (time + entropy_per_charge * entropy)` while the electrochemical potential and temperature form one `Potential` 1-form whose negative wedge with the spatial gradient gives the thermodynamic `Drive`. Commuting the magnetic bivector and the drive with a moving carrier, `(magnetic + drive).commutator(parcel + valence * velocity)`, combines the electrostatic push, the thermal push, and the Lorentz turning while spending zero magnetic power, so Drude's steady drift inverts the scaled spatial rotor `1 - valence * mobility * magnetic - skew * spin_plane`. Joining each species' velocity with its parcel via `velocity.regressive(parcel)` forms a unified charge-and-entropy `Flux`; reversing the magnetic field or magnetization reverses the spatial rotors and maps the linear transport `Response` to its `.adjugate()`, expressing Onsager-Casimir reciprocity while keeping the Lorentz-invariant dissipation `drive & flux` positive. Wedging each triangle corner's simplex gradient with the driven flux assembles the dual-cell outflow operator for isothermal current-driven and open-circuit heat-driven steady states.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

# Planar projective spacetime with degenerate weight w, joined with a null direction s for entropy.
ga = Algebra("x+y+t-w0") * Algebra("s0")
space = Algebra("x+y+")
quantities = Algebra("t-s0")
mv = NumpyContext(ga).multivector

Scalar = ga.gatype.scalar()                                                   # Scalar+
Antiscalar = ga.gatype.antiscalar()                                           # Antiscalar-
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
Parcel = Plane.derive.within(quantities).derive.dual                          # Antivector-
Spatial = Plane.derive.within(space).derive.dual                              # Antivector-
Upright = Plane.derive.within(space)                                          # Vector+
Potential = Plane.derive.within(quantities)                                   # Vector+
Drive = ga.gatype.bivector().derive.touching(space, quantities)               # Bivector+
Magnetic = ga.gatype.bivector().derive.within(space)                          # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
DriftMap = ga.gatype((Spatial, Drive))                                        # Spatial- <- Drive+
Response = ga.gatype((Flux, Drive))                                           # Flux- <- Drive+

x_direction = mv.x.dual()                                                     # [] Spatial-
y_direction = mv.y.dual()                                                     # [] Spatial-
time = -mv.t.dual()                                                           # [] Parcel-
entropy = mv.s.dual()                                                         # [] Parcel-
origin = mv.w.dual()                                                          # [] Point-
spin_plane = mv.xy                                                            # [] Magnetic+


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Upright]:
    """Joining opposite corners of each triangle gives its area (`Scalar+[F]`) and its three corner simplex gradients (`[3] Upright+[F]`)."""
    corners = topology.incidence[2, 0] * vertices                             # [3] Point-[F]
    facing = topology.off[2, 0] * vertices                                    # [2, 3] Point-[F]
    # Joining the opposite edge with time and entropy forms the hyperplane of constant off-plane potentials, normalized to read one at the corner.
    sides = facing[0] & facing[1]                                             # [3] Antibivector-[F]
    unread = time & entropy                                                   # [] Antibivector-
    gradients = ((sides & unread) / (sides & unread & corners)).cast(Upright)  # [3] Upright+[F]
    double_areas = corners[0] & corners[1] & corners[2] & unread              # Antiscalar-[F]
    return double_areas.norm() / 2, gradients


def drifts(magnetic: Magnetic, valences: np.ndarray, entropies_per_charge: np.ndarray, mobilities: np.ndarray,
           skews: np.ndarray) -> DriftMap:
    """Drude steady balance between collision drag, Lorentz turning, and spin skew scattering maps a drive to each carrier species' drift velocity (`[..., 2] Spatial- <- Drive+`)."""
    parcels = valences * (time + entropies_per_charge * entropy)              # [2] Parcel-
    # Commuting the drive and magnetic field with a moving parcel separates into the rest push on the parcel plus magnetic turning of its spatial drift.
    rest_push = Drive.commutator(parcels)                                     # [2] Spatial- <- Drive+
    # Magnetic turning and spin skew scattering combine with collision drag into one scaled spatial rotor per carrier species.
    hall_rotors = 1 - valences * mobilities * magnetic[..., None] - skews * spin_plane   # [..., 2] Rotor+
    return hall_rotors.inverse() * (mobilities * rest_push)


def response(magnetic: Magnetic, valences: np.ndarray, entropies_per_charge: np.ndarray, mobilities: np.ndarray,
             skews: np.ndarray, densities: np.ndarray, conductivity: float) -> Response:
    """Carrier drift fluxes weighted by `densities` (`[..., 2]`) combine with lattice thermal conduction into the linear transport map (`[...] Flux- <- Drive+`)."""
    parcels = valences * (time + entropies_per_charge * entropy)              # [2] Parcel-
    carrier_drifts = drifts(magnetic, valences, entropies_per_charge, mobilities, skews)   # [..., 2] Spatial- <- Drive+
    carrier_response = (densities * carrier_drifts.regressive(parcels)).sum(axis=-1)       # [...] Flux- <- Drive+
    lattice_response = conductivity * Drive.commutator(entropy).regressive(entropy)        # [] Flux- <- Drive+
    return carrier_response + lattice_response


def outflow(topology: Topology, areas: Scalar, gradients: Upright, response: Response) -> SparseExtensor:
    """Wedging triangle simplex gradients with the driven flux assembles the dual-cell charge and entropy outflow operator (`(Parcel- <- Potential+)[V <- V]`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    corner_drives = -(gradients ^ Potential)                                  # [3] (Drive+ <- Potential+)[F]
    triangle_outflow = areas * (gradients[:, None] ^ response(corner_drives)[None, :])   # [3, 3] (Parcel- <- Potential+)[F]
    return (corners[:, None].adjugate() * spdiag(triangle_outflow) * corners[None, :]).sum(axis=0).sum(axis=0)


def steady_potentials(topology: Topology, areas: Scalar, gradients: Upright, response: Response,
                      high_contacts: np.ndarray, low_contacts: np.ndarray, current: float,
                      temperature_difference: float) -> tuple[Potential, Potential]:
    """Zero net charge and entropy outflow on free dual cells determines the isothermal current-driven and open-circuit heat-driven vertex potentials (`Potential+[V]`)."""
    V, E, F = topology.n_elements
    contact_signs = np.zeros(V)
    contact_signs[high_contacts], contact_signs[low_contacts] = 1, -1
    # Solving unit electrochemical and thermal drops between the contacts gives two linear basis states whose combination enforces fixed current or open circuit.
    contact_potentials = mv.scalar(contact_signs[:, None]).field() * stack((mv.t, mv.s)) / 2   # [2] Potential+[V]
    free_reading = topology.reading(np.flatnonzero(contact_signs == 0), 0)    # Scalar+[V_free <- V]
    vertex_outflow = outflow(topology, areas, gradients, response)            # (Parcel- <- Potential+)[V <- V]
    free_outflow = free_reading * vertex_outflow * free_reading.adjugate()    # (Parcel- <- Potential+)[V_free <- V_free]
    basis_potentials = contact_potentials + free_reading.adjugate() * free_outflow.solve(
        -(free_reading * vertex_outflow(contact_potentials)))                 # [2] Potential+[V]
    # Charge leaving the high-contact dual cells gives the net current entering the conductor under each basis drop.
    contact_currents = mv.t & (topology.reading(high_contacts, 0) * vertex_outflow(basis_potentials)).sites.sum()   # [2] Scalar+
    unit_electric, unit_thermal = basis_potentials                            # Potential+[V], Potential+[V]
    electric_current, thermal_current = contact_currents                      # [] Scalar+, [] Scalar+
    return (unit_electric * (current / electric_current),
            (unit_thermal - unit_electric * (thermal_current / electric_current)) * temperature_difference)


def steady_states(topology: Topology, areas: Scalar, gradients: Upright, high_contacts: np.ndarray,
                  low_contacts: np.ndarray, current: float, temperature_difference: float, magnetic: Magnetic,
                  valences: np.ndarray, entropies_per_charge: np.ndarray, mobilities: np.ndarray, skews: np.ndarray,
                  densities: np.ndarray, conductivity: float
                  ) -> Iterator[tuple[np.ndarray, Potential, Spatial, Potential, Spatial]]:
    """Yields per-frame carrier densities (`[2]`), vertex potentials (`Potential+[V]`), and triangle drift velocities (`[2] Spatial-[F]`) under current and heat drives."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    for frame, frame_densities in enumerate(densities):
        carrier_drifts = drifts(magnetic[frame], valences, entropies_per_charge, mobilities, skews)   # [2] Spatial- <- Drive+
        frame_response = response(magnetic[frame], valences, entropies_per_charge, mobilities, skews,
                                  frame_densities, conductivity)              # [] Flux- <- Drive+
        electric_potentials, thermal_potentials = steady_potentials(
            topology, areas, gradients, frame_response, high_contacts, low_contacts, current, temperature_difference)
        potentials = stack((electric_potentials, thermal_potentials))         # [2] Potential+[V]
        triangle_drives = -(gradients ^ (corners * potentials[:, None])).sum(axis=-1)   # [2] Drive+[F]
        yield (frame_densities, electric_potentials, carrier_drifts(triangle_drives[0]),
               thermal_potentials, carrier_drifts(triangle_drives[1]))
