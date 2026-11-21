"""Electromechanical equilibrium of a laminated piezoelectric strip in planar spacetime.

Displacements are intrinsic spatial directions, electric fields are extrinsic planes through
time, and polarization and electric displacement are intrinsic excitations swept through time.
A material's elastic, piezoelectric and dielectric maps define its electric enthalpy on each
triangle through the corner simplex gradients, and pulling their work pairings back to the
vertices gives mechanical equilibrium and charge balance for grounded, driven or floating
electrodes.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.algebras import STA2D as ga
from numga.sparse import Sparse, SparseExtensor, spdiag
from numga.topology import Topology
from examples.electromagnetism.piezoelectric.numerical import solve_coupled

# Planar spacetime x+y+t-: two positive spatial axes and one negative time axis.
ctx = NumpyContext(ga, units=True)
mv = ctx.multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector().derive.within(Algebra("x+y+"))
Point = Plane.derive.dual
Field = ga.gatype.bivector().derive.touching(Algebra("t-"))
Excitation = Field.derive.dual
Strain = ga.gatype((Plane, Point))
Elastic = ga.gatype((Scalar, Point, Plane, Point, Plane))
Polarization = ga.gatype((Excitation, Point, Plane))
Permittivity = ga.gatype((Excitation, Field))

time = -mv.t.dual()                                                               # Antivector-
axis = mv.x.dual()                                                                # Point-
pole = mv.y.dual()                                                                # Point-


def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane]:
    """Opposite edge vectors joined against time give each triangle's area (`Scalar+[F]`, `length^2`) and corner simplex gradients (`[3] Plane+[F]`, `1/length`)."""
    # Each corner's opposite edge vector, oriented along the triangle's boundary cycle.
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)                  # [3] Point-[F], length
    double_areas = (sides[0] & sides[1]) & time                                   # Antiscalar-[F], length^2
    # Sweeping each opposite side through time and dividing by twice the signed area gives the corner's simplex gradient.
    gradients = (time & sides) / double_areas                                     # [3] Plane+[F], 1/length
    return double_areas.norm() / 2, gradients


def strain(topology: Topology, displacement: Point, gradients: Plane) -> Strain:
    """Corner displacements paired with the simplex gradients form the symmetric strain map (`[F] Plane+ <- Point-`) on each triangle."""
    # Evaluate on the batch of triangles so the adjugate acts on each triangle's local map.
    corner_displacements = (topology.incidence[2, 0] * displacement).batch()      # [..., 3, F] Point-, length
    gradient = (corner_displacements.dual_inverse() * (gradients.batch() & Point)).sum(axis=-2)  # [..., F] Plane+ <- Point-
    return (gradient + gradient.adjugate()) / 2


def constitutive(directions: Plane, stretch_stiffnesses: Scalar, cross_stiffness: Scalar,
                 shear_stiffness: Scalar, stretch_polarizations: Scalar,
                 shear_polarization: Scalar, permittivities: Scalar
                 ) -> tuple[Elastic, Polarization, Permittivity]:
    """Two orthogonal material planes (`[...] Plane+`) determine the elastic, piezoelectric and permittivity maps."""
    transverse_plane, poling_plane = directions[..., 0], directions[..., 1]       # [...] Plane+, [...] Plane+
    sheets = directions.dual() & time                                             # [..., 2] Excitation-
    transverse_sheet, poling_sheet = sheets[..., 0], sheets[..., 1]               # [...] Excitation-, [...] Excitation-
    # Each material plane reads extension along its own normal direction; their symmetric cross reading is shear.
    stretches = (directions & Point) * (Plane & directions.dual())                # [..., 2] Scalar+ <- (Point-, Plane+)
    shear = (
        (transverse_plane & Point) * (Plane & poling_plane.dual())
        + (poling_plane & Point) * (Plane & transverse_plane.dual())
    )                                                                             # [...] Scalar+ <- (Point-, Plane+)
    elastic = (
        (stretches * stretches * stretch_stiffnesses).sum(axis=-1)
        + cross_stiffness * (stretches[..., 0] * stretches[..., 1] + stretches[..., 1] * stretches[..., 0])
        + shear_stiffness * shear * shear
    )
    # Extension excites polarization along the poling world-sheet; shear excites it across the transverse world-sheet.
    polarization = (
        poling_sheet * (stretches * stretch_polarizations).sum(axis=-1)
        + transverse_sheet * shear_polarization * shear
    )
    permittivity = (sheets * (Field & sheets) * permittivities).sum(axis=-1)
    return elastic, polarization, permittivity


def enthalpy(local_strain: Strain, electric_field: Field, elastic: Elastic, polarization: Polarization,
             permittivity: Permittivity, poling: Scalar) -> Scalar:
    """Local strain (`[...] Plane+ <- Point-`) and electric field (`[...] Field+`, `voltage/length`) contract with the material maps into the electric enthalpy density (`[...] Scalar+`, `energy/length^3`)."""
    deformation = local_strain.dual()                                             # [...] Point- <- Point-
    # Each trace contracts one displacement-gradient pair of the constitutive map with the local deformation.
    elastic_density = elastic(deformation, Plane, deformation, Plane).trace(1, 2).trace(1, 2) / 2  # [...] Scalar+, energy/length^3
    piezo_excitation = polarization(deformation, Plane).trace(1, 2) * poling      # [...] Excitation-, charge/length^2
    dielectric_excitation = permittivity(electric_field)                          # [...] Excitation-, charge/length^2
    return elastic_density - (electric_field & piezo_excitation) - (electric_field & dielectric_excitation) / 2


def assemble(topology: Topology, areas: Scalar, gradients: Plane, elastic: Elastic, polarization: Polarization,
             permittivity: Permittivity, poling: Scalar, width: Scalar
             ) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor]:
    """Pullback through the corner incidences sums the triangle enthalpy blocks into vertex maps (`[V, V]`)."""
    # Bind each corner's simplex gradient to form triangle-level work and excitation maps (test corner first, trial corner second).
    corner_gradients = gradients.batch()                                          # [3, F] Plane+, 1/length
    work = elastic(Point, corner_gradients[None], Point, corner_gradients[:, None])  # [3, 3, F] Scalar+ <- (Point-, Point-)
    piezo_map = polarization(Point, corner_gradients) * poling.batch()            # [3, F] Excitation- <- Point-
    field_map = corner_gradients ^ (-mv.t * Scalar)                               # [3, F] Field+ <- Scalar+, 1/length

    # Convert the bilinear work form into a linear map from trial displacement to test force.
    elastic_blocks = (Plane & Point).solve(work)                                  # [3, 3, F] Plane+ <- Point-
    # Pulling excitation back across the electric field pairs trial voltage with test force;
    # its adjugate supplies induced charge from displacement without a second coupling law.
    mixed_blocks = -piezo_map[:, None].adjugate()(field_map[None])                # [3, 3, F] Plane+ <- Scalar+
    electric_blocks = -field_map[:, None].adjugate()(permittivity(field_map[None]))  # [3, 3, F] Antiscalar- <- Scalar+

    volumes = width * areas.batch()                                               # [F] Scalar+, length^3
    corners = topology.incidence[2, 0]                                            # [3] Scalar+[F <- V]
    return tuple(
        (corners[:, None].adjugate() * spdiag((local_blocks * volumes).field()) * corners[None, :]).sum(axis=0).sum(axis=0)
        for local_blocks in (elastic_blocks, mixed_blocks, electric_blocks)
    )


def equilibrium(mechanical: SparseExtensor, coupling: SparseExtensor, dielectric: SparseExtensor,
                free_motion: Sparse, free_voltage: Sparse,
                electrode_potential: Scalar, mechanical_load: Plane) -> tuple[Point, Scalar]:
    """Mechanical equilibrium and charge balance determine the vertex displacements (`Point-[V]`, `length`) and potentials (`Scalar+[V]`, `voltage`)."""
    # Restrict the enthalpy blocks to unconstrained displacements and independent potential groups.
    elastic = free_motion.adjugate() * mechanical * free_motion
    mixed = free_motion.adjugate() * coupling * free_voltage
    capacitance = free_voltage.adjugate() * dielectric * free_voltage
    load = free_motion.adjugate() * (mechanical_load - coupling(electrode_potential))
    charge = -(free_voltage.adjugate() * dielectric(electrode_potential))
    displacement, potential = solve_coupled(elastic, mixed, capacitance, load, charge)
    return free_motion * displacement, electrode_potential + free_voltage * potential


def load_cycle(displacement: Point, potential: Scalar, axial_strain: Scalar,
               n_frames: int) -> Iterator[tuple[Point, Scalar, Scalar]]:
    """Yields sinusoidal equilibrium states (`Point-[V], Scalar+[V], Scalar+[F]`) across one drive period."""
    for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        amplitude = float(np.sin(phase))
        yield displacement * amplitude, potential * amplitude, axial_strain * amplitude
