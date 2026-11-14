"""Spacetime London electrodynamics for Meissner expulsion in a rod and fluxoid quantization in a ring.

Below the critical temperature, conduction electrons in a superconductor are modelled as a resistive normal fluid
alongside a dissipationless superfluid whose fraction grows as the metal cools. Because the canonical momentum of the
superfluid is the spacetime gradient of one condensate phase, its exterior derivative vanishes. The space-time part of
that closure accelerates supercurrent under an electric field, as in an idealized zero-resistance conductor, while the
spatial part locks supercurrent directly to the electromagnetic potential and expels magnetic flux from the interior.

Over a planar cross-section where fields and currents do not vary along the rod axis, an axial potential produces a
magnetic field bivector in planes through the rod axis whose level contours in the cross-section trace magnetic field
lines. Balancing the Ampère stiffness of that field against normal Ohmic conduction and the London supercurrent reduces
each cooling step to one linear system on the interior vertices.

Around a superconducting ring, single-valuedness of the condensate phase quantizes the fluxoid, the sum of magnetic flux
through the hole and supercurrent circulation around its inner edge, to integer multiples of the flux quantum. Axial
magnetic field screens into the ring over the London penetration depth while standing uniform across the hole, and
minimizing the sum of magnetic departure energy and supercurrent kinetic energy selects the integer fluxoid retained
after field-cooled switching.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import STA as ga
from numga.sparse import spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Plane = ga.gatype.vector().derive.within(Algebra("x+y+z+"))                   # Vector+
Point = Plane.derive.dual                                                     # Antivector-
Excitation = ga.gatype.antibivector()                                         # Antibivector-
Field = Excitation.derive.dual                                                # Bivector+
Current = ga.gatype.antivector()                                              # Antivector-

# Symmetry directions along the rod and rest-frame clock, vacuum constitutive duality, and rest-frame superfluid inertia.
rod_direction = mv.z.dual()                                                   # [] Point-
time = -mv.t.dual()                                                           # [] Current-
vacuum = Field.dual()                                                         # [] Excitation- <- Field+
axial_excitation = vacuum(mv.xy)                                              # [] Excitation-
inertia = mv.t | (Current & time).dual()                                      # [] Plane+ <- Current-


def triangle_areas(topology: Topology, vertices: Point) -> Scalar:
    """Signed side joins swept along the rod and through time give each triangle's area (`Scalar+[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [3] Point-[F]
    double_areas = ((sides[0] & sides[1]) & rod_direction) & time             # Scalar-[F]
    return double_areas.norm() / 2


def simplex_gradients(topology: Topology, vertices: Point) -> Plane:
    """Each corner's simplex gradient (`[3] Plane+[F]`) pairs to one from the facing side to that corner."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [3] Point-[F]
    double_areas = ((sides[0] & sides[1]) & rod_direction) & time             # Scalar-[F]
    return (rod_direction & time & sides) / double_areas


def cooling_states(
    topology: Topology,
    areas: Scalar,
    gradients: Plane,
    rod_mask: Scalar,
    heights: Scalar,
    reference_potential: Scalar,
    penetration_depth: float,
    superfluid_shares: np.ndarray,
    applied_fields: np.ndarray,
    dt: float,
    n_substeps: int,
) -> Iterator[tuple[Scalar, Scalar]]:
    """Yields nodal potential and supercurrent (`[...] Scalar+[V]`) every `n_substeps` cooling steps of `dt`."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    dual_areas = (corners.adjugate() * (areas / 3)).sum(axis=0)               # Scalar+[V]
    free = topology.reading(topology.interior[0], 0)                          # Scalar+[V_free <- V]
    # Ampère stiffness pairs each corner's axial-potential field bivector with the vacuum excitation of another.
    corner_fields = gradients ^ mv.z                                          # [3] Field+[F]
    local_stiffness = (corner_fields[:, None] & vacuum(corner_fields[None, :])) * areas   # [3, 3] Scalar+[F]
    stiffness = (corners[:, None].adjugate() * spdiag(local_stiffness) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[V <- V]
    conduction = spdiag(dual_areas * rod_mask / dt)                           # Scalar+[V <- V]
    potential = applied_fields[0] * heights + 0 * reference_potential         # [...] Scalar+[V]
    for step_index, (share, field_strength) in enumerate(zip(superfluid_shares, applied_fields)):
        london_weight = spdiag(dual_areas * rod_mask * share / penetration_depth ** 2)   # Scalar+[V <- V]
        boundary_potential = field_strength * heights                         # Scalar+[V]
        step_operator = conduction + stiffness + london_weight                # Scalar+[V <- V]
        interior_system = (free * step_operator * free.adjugate()) * Scalar   # (Scalar+ <- Scalar+)[V_free <- V_free]
        rhs = conduction * potential + london_weight * reference_potential - step_operator * boundary_potential
        potential = boundary_potential + free.adjugate() * interior_system.solve(free * rhs)   # [...] Scalar+[V]
        if step_index % n_substeps == 0:
            supercurrent = -(rod_mask * share / penetration_depth ** 2) * (potential - reference_potential)
            yield potential, supercurrent


def magnetic_field(topology: Topology, gradients: Plane, potential: Scalar) -> Field:
    """Corner simplex gradients wedged with nodal axial potentials sum to the magnetic field (`[...] Field+[F]`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    return (gradients ^ ((corners * potential[..., None]) * mv.z)).sum(axis=-1)


def radial_profile(topology: Topology, vertices: Point, field: Field, radial_width: float) -> tuple[Scalar, Scalar]:
    """Triangles along the upper radius give surface depth and magnetic field strength (`[...] Scalar+`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    centroids = corners.mean(axis=0) * vertices                               # Point-[F]
    radii = centroids.antinorm()                                              # Scalar+[F]
    ray_mask = ((mv.x & centroids).abs() < radial_width) & ((mv.y & centroids) > 0) & (radii < 1)
    return (1 - radii).batch()[ray_mask], field.norm().batch()[ray_mask]


def ring_fields(
    topology: Topology,
    areas: Scalar,
    gradients: Plane,
    in_hole: np.ndarray,
    penetration_depth: float,
    quanta: np.ndarray,
    applied_fields: np.ndarray,
) -> Scalar:
    """London screening on a ring determines the axial field (`[...] Scalar+[V]`) from fluxoid quanta and rim fields."""
    V, E, F = topology.n_elements
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    hole_faces = in_hole[topology.elements[2]].all(axis=-1)                   # [F]
    ring_indicator = mv.scalar((~hole_faces)[:, None]).field()                # Scalar+[F]
    # Ampère supercurrents of unit corner axial fields pair with rest-frame inertia to assemble London screening.
    corner_currents = gradients ^ axial_excitation                            # [3] Current-[F]
    local_kinetic = (inertia(corner_currents[:, None]) & corner_currents[None, :]) * (ring_indicator * areas)   # [3, 3] Scalar+[F]
    kinetic_form = (corners[:, None].adjugate() * spdiag(local_kinetic) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[V <- V]
    screening = penetration_depth ** 2 * kinetic_form + (corners.adjugate() * (ring_indicator * areas / 3)).sum(axis=0)   # Scalar+[V <- V]
    rim_mask = np.isin(np.arange(V), topology.exterior[0])                    # [V]
    free = topology.reading(np.flatnonzero(~(in_hole | rim_mask)), 0)         # Scalar+[V_free <- V]
    hole_unit = mv.scalar(in_hole[:, None]).field()                           # Scalar+[V]
    rim_unit = mv.scalar(rim_mask[:, None]).field()                           # Scalar+[V]
    boundary_units = stack([hole_unit, rim_unit])                             # [2] Scalar+[V]
    ring_system = (free * screening * free.adjugate()) * Scalar               # (Scalar+ <- Scalar+)[V_free <- V_free]
    basis_fields = boundary_units + free.adjugate() * ring_system.solve(-(free * (screening * boundary_units)))   # [2] Scalar+[V]
    # Each basis mode's fluxoid combines hole magnetic flux with inner-edge screening reaction.
    hole_area = (mv.scalar(hole_faces[:, None]).field() * areas).sites.sum()  # [] Scalar+
    basis_fluxoids = (hole_unit * (screening * basis_fields)).sites.sum() + hole_area * np.array([1, 0])   # [2] Scalar+
    hole_strength = (quanta - applied_fields * basis_fluxoids[1]) / basis_fluxoids[0]   # [...] Scalar+
    return hole_strength * basis_fields[0] + applied_fields * basis_fields[1]


def equilibrium_quanta(
    topology: Topology,
    areas: Scalar,
    gradients: Plane,
    in_hole: np.ndarray,
    penetration_depth: float,
    candidate_quanta: np.ndarray,
    cooling_fields: np.ndarray,
) -> np.ndarray:
    """Energy minimization across candidate fluxoids picks the retained quantum number for each cooling field (`[...]`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    dual_areas = (corners.adjugate() * (areas / 3)).sum(axis=0)               # Scalar+[V]
    hole_faces = in_hole[topology.elements[2]].all(axis=-1)                   # [F]
    ring_areas = mv.scalar((~hole_faces)[:, None]).field() * areas            # Scalar+[F]
    fields = ring_fields(
        topology, areas, gradients, in_hole, penetration_depth,
        candidate_quanta[:, None], cooling_fields[None, :],
    )                                                                         # [...] Scalar+[V]
    slopes = (gradients * (corners * fields[..., None])).sum(axis=-1)         # [...] Plane+[F]
    currents = slopes ^ axial_excitation                                      # [...] Current-[F]
    departures = fields - cooling_fields[None, :]                             # [...] Scalar+[V]
    kinetic_energy = penetration_depth ** 2 * (ring_areas * (inertia(currents) & currents)).sites.sum()   # [...] Scalar+
    total_cost = (dual_areas * departures * departures).sites.sum() + kinetic_energy   # [...] Scalar+
    return candidate_quanta[total_cost.argmin(axis=0)]
