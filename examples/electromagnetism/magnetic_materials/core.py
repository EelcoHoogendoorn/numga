"""A permanent magnet, soft iron and a conductor share one solved magnetic field.

Prescribed magnetization drives bound current on the magnet's flanks, permeability scales the
iron's reversible magnetic excitation, and conductivity induces opposing currents in the copper.
"""
from __future__ import annotations

from collections.abc import Iterator
import numpy as np
from numga import Algebra, Extensor, NumpyContext, stack
from numga.sparse import Sparse, SparseExtensor, spdiag
from numga.topology import Topology

ga = Algebra("x+y+z+t-w0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                   # Scalar+
Upright = ga.gatype.vector().derive.within(Algebra("x+y+"))   # Vector+
Point = ga.gatype.antivector()                                # Antivector-
Current = ga.gatype.antivector()                              # Antivector-
Field = ga.gatype.bivector()                                  # Bivector+
Excitation = ga.gatype.antibivector()                         # Antibivector-
origin = mv.w.dual()                                          # Point-
axis = mv.z.dual()                                            # Current-
time = -mv.t.dual()                                           # Current-
sweep = axis & time                                           # Excitation-
magnetic_planes = stack((mv.xz.dual(), mv.yz.dual()))         # [2] Excitation-
magnetic_response = (magnetic_planes * (Field & magnetic_planes)).sum(axis=0)  # Excitation- <- Field+
conduction = axis * (Field & sweep)                           # Current- <- Field+


def triangle_areas(topology: Topology, vertices: Point) -> Extensor:
    """Signed corner joins with `sweep` measure each triangle's cross-sectional area (`Scalar+[F]`)."""
    triangle_vertices = topology.incidence[2, 0] * vertices       # [3] Point-[F]
    double_areas = triangle_vertices[0] & triangle_vertices[1] & triangle_vertices[2] & sweep  # Antiscalar-[F]
    return double_areas.norm() / 2                                # Scalar+[F]


def simplex_gradients(topology: Topology, vertices: Point) -> Extensor:
    """Opposite edge joins determine the three corner simplex gradients on each triangle (`[3] Upright+[F]`)."""
    triangle_vertices = topology.incidence[2, 0] * vertices       # [3] Point-[F]
    facing = topology.off[2, 0] * vertices                        # [3, 2] Point-[F]
    sides = facing[0] & facing[1]                                 # [3] Excitation-[F]
    slopes = (sides & sweep) / (sides & sweep & triangle_vertices)  # [3] Vector+[F]
    return slopes.cast(Upright)                                   # [3] Upright+[F]


def magnetic_stiffness(topology: Topology, areas: Scalar, gradients: Upright,
                       permeability: Scalar) -> SparseExtensor:
    """Permeability weights the magnetic storage coupling across triangle corners (`Scalar+[V <- V]`)."""
    corners = topology.incidence[2, 0]                            # [3] Scalar+[F <- V]
    # An axial potential `potential * z` produces the cross-sectional magnetic field `gradients ^ z`.
    field_gradients = gradients ^ mv.z                            # [3] Field+[F]
    reluctance = magnetic_response / (corners.mean(axis=0) * permeability)  # (Excitation- <- Field+)[F]
    local_stiffness = areas * (field_gradients[:, None] & reluctance(field_gradients[None, :]))  # [3, 3] Scalar+[F]
    return (corners[:, None].adjugate() * spdiag(local_stiffness) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[V <- V]


def remanent_source(topology: Topology, areas: Scalar, gradients: Upright,
                    remanence: Excitation) -> Extensor:
    """Permanent magnetization supplies an integrated bound-current load at each vertex (`Scalar+[V]`)."""
    corners = topology.incidence[2, 0]                            # [3] Scalar+[F <- V]
    field_gradients = gradients ^ mv.z                            # [3] Field+[F]
    triangle_remanence = corners.mean(axis=0) * remanence         # Excitation-[F]
    return (corners.adjugate() * (areas * (field_gradients & triangle_remanence))).sum(axis=0)  # Scalar+[V]


def evolution(stiffness: SparseExtensor, source: Scalar, dual_areas: Scalar,
              conductivity: Scalar, extension: Sparse, coil_load: Scalar,
              drive: np.ndarray, step: float) -> Iterator[tuple[Extensor, Extensor]]:
    """Yields `(potential, current)` fields (`Scalar+[V]`, `Current-[V]`) across a coil pulse."""
    conductance_mass = dual_areas * conductivity                  # Scalar+[V]
    bound_current = axis * (source / dual_areas)                  # Current-[V]
    coil_current = axis * (coil_load / dual_areas)                # Current-[V]
    initial = (extension.adjugate() * stiffness * extension) * Scalar  # Scalar+[V_interior <- V_interior]
    potential = extension * initial.solve(extension.adjugate() * (source + coil_load * drive[0]))  # Scalar+[V]
    yield potential, bound_current + drive[0] * coil_current
    storage = spdiag(conductance_mass / step)                     # Scalar+[V <- V]
    implicit = (extension.adjugate() * (storage + stiffness / 2) * extension) * Scalar  # Scalar+[V_interior <- V_interior]
    explicit = storage - stiffness / 2                            # Scalar+[V <- V]
    # An isolated conductor redistributes charge until its net axial current vanishes.
    uniform = extension * implicit.solve(extension.adjugate() * (conductance_mass / step))  # Scalar+[V]
    total_conductance = conductance_mass.sites.sum()              # Scalar+
    normalization = 1 - (conductance_mass * uniform).sites.sum() / total_conductance  # Scalar+
    for previous_drive, following_drive in zip(drive[:-1], drive[1:]):
        midpoint_load = source + coil_load * ((previous_drive + following_drive) / 2)  # Scalar+[V]
        mean_potential = (conductance_mass * potential).sites.sum() / total_conductance  # Scalar+
        particular = extension * implicit.solve(
            extension.adjugate() * (explicit * potential + midpoint_load - conductance_mass * mean_potential / step)
        )                                                         # Scalar+[V]
        mean_following = (conductance_mass * particular).sites.sum() / total_conductance / normalization  # Scalar+
        following = particular + uniform * mean_following         # Scalar+[V]
        rate = (following - potential) / step                     # Scalar+[V]
        mean_rate = (conductance_mass * rate).sites.sum() / total_conductance  # Scalar+
        electric = ((rate - mean_rate) * mv.t) ^ mv.z             # Field+[V]
        induced_current = conductivity * conduction(electric)     # Current-[V]
        potential = following                                     # Scalar+[V]
        yield potential, bound_current + following_drive * coil_current + induced_current


def joule_heating(potentials: Scalar, dual_areas: Scalar, conductivity: Scalar,
                  step: float) -> Extensor:
    """Resistive power integrates `electric & (induced_currents & time)` across the conductor (`[...] Scalar+`)."""
    conductance_mass = dual_areas * conductivity                  # Scalar+[V]
    rates = (potentials[1:] - potentials[:-1]) / step             # [...] Scalar+[V]
    mean_rates = (conductance_mass * rates).sites.sum() / conductance_mass.sites.sum()  # [...] Scalar+
    electric = ((rates - mean_rates) * mv.t) ^ mv.z               # [...] Field+[V]
    induced_currents = conductivity * conduction(electric)        # [...] Current-[V]
    return (dual_areas * (electric & (induced_currents & time))).sites.sum()  # [...] Scalar+
