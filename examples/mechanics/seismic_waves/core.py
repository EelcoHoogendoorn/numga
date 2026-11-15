"""Elastic waves through a planet with a liquid core and a free surface.

Simplex gradients give a constant strain on each triangle. The material reads that strain,
and its response pulls back to nodal forces; a lumped mass map turns them into motion.
Zero shear stiffness gives a liquid; its shear viscosity, the shear part of the same work form applied to the
velocity, damps the shear motion it cannot store. Liquid and solid share the vertices where they meet.
The divergence and the rotation of the velocity, the simplex gradients paired with and wedged against the corner
velocities, separate compressional from shear motion.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D as ga
from numga.sparse import spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                      # Scalar+
Point = ga.gatype.antivector()                                   # Antivector-
Plane = ga.gatype.vector()                                       # Vector+
Force = ga.gatype.vector()                                       # Vector+
Bivector = ga.gatype.bivector()                                  # Bivector+
Acceleration = ga.gatype((Point, Point))                         # Point- <- Point-
Stiffness = ga.gatype((Force, Point))                            # Force+ <- Point-
Divergence = ga.gatype((Scalar, Point))                          # Scalar+ <- Point-
Rotation = ga.gatype((Bivector, Point))                          # Bivector+ <- Point-
Elastic = ga.gatype((Scalar, Plane, Plane, Point, Point))        # Scalar+ <- (Plane+, Plane+, Point-, Point-)

antiscalar = mv.xy.dual()                                        # Antiscalar-
dilation_traction = Plane * (Plane & Point)                      # [] Plane+ <- (Plane+, Plane+, Point-)
# Subtracting the exterior-product term removes infinitesimal rotation, leaving symmetric shear.
direct_shear = (Plane | Plane) * Point.dual_inverse()            # [] Plane+ <- (Plane+, Plane+, Point-)
shear_traction = 2 * direct_shear - (Plane | (Plane ^ Point.dual_inverse()))  # [] Plane+ <- (Plane+, Plane+, Point-)


def triangle_areas(topology: Topology, vertices: Point) -> Scalar:
    """Oriented corner displacements join into each triangle's area (`Scalar+[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)          # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                   # Patch-[F]
    return patches.antinorm()                                             # Scalar+[F]


def simplex_gradients(topology: Topology, vertices: Point) -> Plane:
    """Each triangle's opposite edge divided by twice its oriented patch gives its three simplex gradients (`[3] Plane+[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)          # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                   # Patch-[F]
    return sides / (2 * patches)                                          # [3] Plane+[F]


def constitutive_form(lame_lambda: Scalar, lame_mu: Scalar) -> Elastic:
    """Plane-strain isotropic Lamé parameters define a symmetric elastic work form on pairs of displacement gradients."""
    traction = lame_lambda * dilation_traction + lame_mu * shear_traction  # [...] Plane+ <- (Plane+, Plane+, Point-)
    return traction & Point                                               # [...] Scalar+ <- (Plane+, Plane+, Point-, Point-)


def viscous_form(viscosity: Scalar) -> Elastic:
    """Deviatoric plane-strain shear viscosity defines a dissipation form on velocity gradients without resisting swelling."""
    return (viscosity * (shear_traction - 2 * dilation_traction / 3)) & Point  # [...] Scalar+ <- (Plane+, Plane+, Point-, Point-)


def assembled_stiffness(topology: Topology, gradients: Plane, areas: Scalar, material: Elastic) -> Stiffness:
    """Pullback of a triangle work form along corner incidences assembles the nodal operator (`(Force+ <- Point-)[V <- V]`)."""
    work = material(gradients[:, None], gradients[None, :], Point, Point)
    local = (Force & Point).solve(work) * areas                           # [3, 3] (Force+ <- Point-)[F]
    corners = topology.incidence[2, 0]                                    # [3] Scalar+[F <- V]
    return (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)


def vertex_masses(topology: Topology, areas: Scalar, density: Scalar) -> Scalar:
    """Each triangle shares one third of its mass with each of its three corners (`Scalar+[V]`)."""
    corner_mean = topology.incidence[2, 0].mean(axis=0)                   # Scalar+[F <- V]
    return corner_mean.adjugate() * (density * areas)                     # Scalar+[V]


def divergence_operator(topology: Topology, gradients: Plane) -> Divergence:
    """Pairing each corner's simplex gradient with its nodal vector sums to the triangle divergence (`(Scalar+ <- Point-)[F <- V]`)."""
    corners = topology.incidence[2, 0]                                    # [3] Scalar+[F <- V]
    return (spdiag(gradients & Point) * corners).sum(axis=0)              # (Scalar+ <- Point-)[F <- V]


def rotation_operator(topology: Topology, gradients: Plane) -> Rotation:
    """Wedging each corner's simplex gradient with the dual inverse of its nodal vector gives the triangle rotation (`(Bivector+ <- Point-)[F <- V]`)."""
    corners = topology.incidence[2, 0]                                    # [3] Scalar+[F <- V]
    return (spdiag(gradients ^ Point.dual_inverse()) * corners).sum(axis=0)  # (Bivector+ <- Point-)[F <- V]


def field_energy(masses: Scalar, stiffness: Stiffness, displacement: Point, velocity: Point) -> Scalar:
    """Kinetic and elastic energy sum across all vertices (`Scalar+`)."""
    kinetic = (masses * velocity.scalar_norm_squared()).sites.sum() / 2   # [] Scalar+
    elastic = (stiffness(displacement) & displacement).sites.sum() / 2    # [] Scalar+
    return kinetic + elastic                                              # [] Scalar+


def wave_evolution(acceleration_map: Acceleration, drag_map: Acceleration, displacement: Point, velocity: Point,
                   damping_time: float, dt: float, n_frames: int, n_substeps: int) -> Iterator[tuple[Point, Point]]:
    """Yields nodal `(displacement, velocity)` fields (`Point-[V]`) advanced by explicit viscoelastic velocity Verlet."""
    acceleration = acceleration_map(displacement + damping_time * velocity) + drag_map(velocity)  # Point-[V]
    yield displacement, velocity
    for _ in range(n_frames):
        for _ in range(n_substeps):
            velocity = velocity + (dt / 2) * acceleration
            displacement = displacement + dt * velocity
            # Reading strain rate from the half-step velocity keeps the time step explicit.
            acceleration = acceleration_map(displacement + damping_time * velocity) + drag_map(velocity)
            velocity = velocity + (dt / 2) * acceleration
        yield displacement, velocity


def wave_readings(divergence: Divergence, rotation: Rotation,
                  states: Iterable[tuple[Point, Point]]) -> Iterator[tuple[Point, Scalar, Scalar]]:
    """Yields `(displacement, dilation_rate, vorticity)` fields on each frame of `states`."""
    for displacement, velocity in states:
        dilation_rate = divergence(velocity)                              # Scalar+[F]
        vorticity = rotation(velocity) & antiscalar                       # Scalar+[F]
        yield displacement, dilation_rate, vorticity


def normal_modes(stiffness: Stiffness, masses: Scalar, n_total_modes: int) -> tuple[Scalar, Point]:
    """Generalized eigenmodes of `stiffness` against `masses` give squared frequencies (`[n_total_modes] Scalar+`) and shapes (`[n_total_modes] Point-[V]`)."""
    to_points = Force.dual()                                              # [] Point- <- Force+
    return to_points(stiffness).eigh(spdiag(masses * Point), n_total_modes)


def travelling_waves(divergence: Divergence, rotation: Rotation, first_modes: Point, second_modes: Point,
                     peak: float, n_cycle_frames: int) -> Iterator[tuple[Point, Scalar, Scalar]]:
    """Yields `(displacement, dilation, vorticity)` fields as each degenerate mode pair completes one quarter-phase travelling cycle."""
    scales = peak / first_modes.antinorm().batch().max(axis=-1)           # [...] Scalar+
    for pair in range(len(first_modes)):
        for phase in np.linspace(0, 2 * np.pi, n_cycle_frames, endpoint=False):
            displacement = scales[pair] * (np.cos(phase) * first_modes[pair] + np.sin(phase) * second_modes[pair])  # Point-[V]
            yield displacement, divergence(displacement), rotation(displacement) & antiscalar
