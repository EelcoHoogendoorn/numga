"""Damped harmonic steps relax a triangulated surface onto a minimal soap film, from which boundary side sums recover its harmonic conjugate and Bonnet's isometric family.

A soap film settles where no small interior displacement can reduce its area, so its mean curvature vanishes across the interior. Turning each triangle's tangent displacements by a quarter turn in its own plane produces its harmonic conjugate, and turning by intermediate angles sweeps Bonnet's isometric family between the two, following Pinkall and Polthier (1993), https://doi.org/10.1080/10586458.1993.10504266.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga
from numga.sparse import spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-
# Mean curvature, measured against the triangles' own winding.
Curvature = ga.gatype.antipseudoscalar()                                     # Antipseudoscalar-


# --- math -----------------------------------------------------------------------------
def plateau_states(
    topology: Topology, initial_vertices: Point, step_size: float, n_steps: int,
) -> Iterator[tuple[Point, Scalar, Curvature]]:
    """Yields the surface (`Point-[V]`), total area (`Scalar+`), and interior mean curvature (`Curvature-[V]`) at each damped harmonic step with the boundary wire held."""
    corners = topology.incidence[2, 0]                                       # [3] Scalar+[F <- V]
    interior = topology.reading(topology.interior[0], 0)                     # Scalar+[V_interior <- V]
    exterior = topology.reading(topology.exterior[0], 0)                     # Scalar+[V_exterior <- V]
    wire = exterior * initial_vertices                                       # Point-[V_exterior]
    vertices = initial_vertices                                              # Point-[V]
    for _ in range(n_steps + 1):
        sides = topology.boundary[2, 1] * (topology.d[0] * vertices)         # [3] Point-[F]
        patches = (sides[0] & sides[1]) / 2                                  # Patch-[F]
        areas = patches.norm()                                               # Scalar+[F]
        normals = patches / areas                                            # Patch-[F]
        # Each corner's simplex gradient vanishes along the facing side and rises to one at the corner.
        gradients = (sides | patches.inverse()) / 2                          # [3] Plane+[F]
        # Turning each facing side by a quarter turn in the triangle's plane gives twice that corner's area gradient.
        turned_sides = sides.commutator(normals.dual())                      # [3] Point-[F]
        # Pairing each corner's simplex gradient with half every corner's turned side forms the cotangent Laplacian.
        local_laplacian = (gradients[:, None] & turned_sides[None, :]) / 2   # [3, 3] Scalar+[F]
        laplacian = (corners[:, None].adjugate() * spdiag(local_laplacian) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[V <- V]
        area_pull = interior * (corners.adjugate() * turned_sides).sum(axis=0) / 2  # Point-[V_interior]
        # Dividing a vertex's area pull by twice its third of the surrounding patches extracts the mean curvature.
        vertex_patches = interior * (corners.mean(axis=0).adjugate() * patches)  # Patch-[V_interior]
        mean_curvature = (area_pull & vertex_patches.inverse()) / 2          # Curvature-[V_interior]
        yield vertices, areas.sites.sum(), interior.adjugate() * mean_curvature
        # Solving for vanishing pull on the current triangles with the wire held gives the harmonic target surface.
        wire_pull = (interior * laplacian * exterior.adjugate()) * wire      # Point-[V_interior]
        harmonic_interior = ((interior * laplacian * interior.adjugate()) * Point).solve(-wire_pull)  # Point-[V_interior]
        harmonic = interior.adjugate() * harmonic_interior + exterior.adjugate() * wire  # Point-[V]
        # Taking a damped step towards the harmonic surface keeps the triangles and Laplacian well-conditioned.
        vertices = (1 - step_size) * vertices + step_size * harmonic         # Point-[V]


def conjugate_minimal_surface(topology: Topology, vertices: Point) -> tuple[Point, Point]:
    """Recovers the harmonic conjugate (`Point-[V]`) up to global translation from half the sum of the facing sides around each vertex (`Point-[V]`)."""
    corners = topology.incidence[2, 0]                                       # [3] Scalar+[F <- V]
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    areas = patches.norm()                                                   # Scalar+[F]
    normals = patches / areas                                                # Patch-[F]
    gradients = (sides | patches.inverse()) / 2                              # [3] Plane+[F]
    turned_sides = sides.commutator(normals.dual())                          # [3] Point-[F]
    local_laplacian = (gradients[:, None] & turned_sides[None, :]) / 2       # [3, 3] Scalar+[F]
    laplacian = (corners[:, None].adjugate() * spdiag(local_laplacian) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[V <- V]
    # Integrating the quarter-turned differential against the simplex gradients leaves half the bare facing side on each triangle.
    boundary_flux = (corners.adjugate() * sides).sum(axis=0) / 2             # Point-[V]
    # Because the Laplacian annihilates global translations, a least-squares solve selects one conjugate representative.
    conjugate = (laplacian * Point).lstsq(boundary_flux)                     # Point-[V]
    return conjugate, boundary_flux


def bonnet_family(
    topology: Topology, vertices: Point, conjugate: Point, angles: np.ndarray,
) -> Iterator[tuple[Point, Patch]]:
    """Yields each isometric minimal surface (`Point-[V]`) and its vertex Gauss normals (`Patch-[V]`) along Bonnet's family between a minimal surface and its harmonic conjugate."""
    corners = topology.incidence[2, 0]                                       # [3] Scalar+[F <- V]
    for angle in angles:
        surface = vertices * np.cos(angle) + conjugate * np.sin(angle)       # Point-[V]
        sides = topology.boundary[2, 1] * (topology.d[0] * surface)          # [3] Point-[F]
        normals = (corners.adjugate() * (sides[0] & sides[1])).sum(axis=0).normalized()  # Patch-[V]
        yield surface, normals
