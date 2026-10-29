r"""Minimal surfaces relax from triangle meshes, and their harmonic conjugates and Bonnet's family follow from them.

A soap film settles where no small displacement of its interior can shrink its area, and there its mean curvature vanishes. Turning every small piece of a minimal surface a quarter turn in its own tangent plane leaves the pieces fitting together into a second minimal surface, its harmonic conjugate, and turning them by any other angle gives Bonnet's family between the two.

The quarter turn in a triangle's plane is the commutator with the plane, `sides.commutator(normals.dual())`. It is both the area gradient, whose sums at the vertices are the cotangent Laplacian's pull `laplacian * vertices`, and the turned differential that integrates to the conjugate. The corner gradients read that differential over each triangle as half the bare facing side, and the sums of those sides cancel at every interior vertex, where neighbouring triangles run their shared edge in opposite senses, so the conjugate is solved from the boundary alone, following Pinkall and Polthier (1993), https://doi.org/10.1080/10586458.1993.10504266.

In the vector notation of differential geometry, the area gradient at corner $i$ of a triangle $T$ is $\nabla_{p_i} A_T = \frac{1}{2} N_T \times e_i$, the cotangent Laplacian is $(L X)_i = \frac{1}{2}\sum_{j} (\cot \alpha_{ij} + \cot \beta_{ij})(X_i - X_j) = -2 H_i A_i N_i$, the conjugate integrates $dX^* = N \times dX$, and Bonnet's family is $X_\theta = \cos\theta\, X + \sin\theta\, X^*$.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.algebras import VGA3D as ga
from numga.sparse import spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-
# Mean curvature, measured against the triangles' own winding.
Curvature = ga.gatype.antipseudoscalar()                                     # Antipseudoscalar-


# --- math -----------------------------------------------------------------------------
def relax_plateau(
    mesh: Mesh, initial_vertices: Point, step_size: float, n_steps: int,
) -> Iterator[tuple[Point, Scalar, Curvature]]:
    """The surface at each damped harmonic step towards least area, its boundary wire held, with its total area and
    its mean curvature at the vertices, zero on the wire."""
    corners = mesh.incidence[2, 0]                                           # [corners] Scalar+[faces <- vertices]
    interior = mesh.reading(mesh.interior(0))                                # Scalar+[interior <- vertices]
    exterior = mesh.reading(mesh.exterior(0))                                # Scalar+[exterior <- vertices]
    wire = exterior * initial_vertices                                       # Point-[exterior]
    vertices = initial_vertices                                              # Point-[vertices]
    for _ in range(n_steps + 1):
        sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                 # [corners] Point-[faces]
        patches = (sides[0] & sides[1]) / 2                                  # Patch-[faces]
        areas = patches.norm()                                               # Scalar+[faces]
        normals = patches / areas                                            # Patch-[faces]
        # Each corner's hat rises from zero along the facing side to one at the corner.
        corner_gradients = (sides | patches.inverse()) / 2                   # [corners] Plane+[faces]
        # Each triangle's corner gradients paired with one another over its area, pulled back to the vertices
        # through both corners of each pair.
        local_laplacian = (corner_gradients[:, None] | corner_gradients[None, :]) * areas  # [corners, corners] Scalar+[faces]
        laplacian = (~corners[:, None] * spdiag(local_laplacian) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[vertices <- vertices]
        # A quarter turn in each triangle's plane makes each facing side twice its corner's area gradient.
        turned_sides = sides.commutator(normals.dual())                      # [corners] Point-[faces]
        area_pull = interior * (~corners * turned_sides).sum(axis=0) / 2     # Point-[interior]
        # A vertex's area pull is twice its mean curvature times its third of the patches around it.
        vertex_patches = interior * (~corners.mean(axis=0) * patches)        # Patch-[interior]
        mean_curvature = (area_pull & vertex_patches.inverse()) / 2          # Curvature-[interior]
        yield vertices, areas.sites.sum(), ~interior * mean_curvature
        # The harmonic surface of this step's triangles: the interior where their pull vanishes, the wire held.
        wire_pull = (interior * laplacian * ~exterior) * wire                # Point-[interior]
        harmonic_interior = ((interior * laplacian * ~interior) * Point).solve(-wire_pull)  # Point-[interior]
        harmonic = ~interior * harmonic_interior + ~exterior * wire          # Point-[vertices]
        # Moving part of the way keeps the triangles, and with them the Laplacian, from changing too fast.
        vertices = (1 - step_size) * vertices + step_size * harmonic         # Point-[vertices]


def conjugate_minimal_surface(mesh: Mesh, vertices: Point) -> tuple[Point, Point]:
    """The harmonic conjugate of a minimal surface, up to a translation, and the sums of the sides facing each vertex
    that it is solved from."""
    corners = mesh.incidence[2, 0]                                           # [corners] Scalar+[faces <- vertices]
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                     # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[faces]
    areas = patches.norm()                                                   # Scalar+[faces]
    corner_gradients = (sides | patches.inverse()) / 2                       # [corners] Plane+[faces]
    local_laplacian = (corner_gradients[:, None] | corner_gradients[None, :]) * areas  # [corners, corners] Scalar+[faces]
    laplacian = (~corners[:, None] * spdiag(local_laplacian) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[vertices <- vertices]
    # The corner gradients read the quarter-turned differential over each triangle as half the bare facing side:
    # the triangle's area and its normal cancel.
    boundary_flux = (~corners * sides).sum(axis=0) / 2                       # Point-[vertices]
    # The Laplacian leaves translations free, and the least-squares solve picks one conjugate among them.
    conjugate = (laplacian * Point).lstsq(boundary_flux)                     # Point-[vertices]
    return conjugate, boundary_flux


def bonnet_family(
    mesh: Mesh, vertices: Point, conjugate: Point, angles: np.ndarray,
) -> Iterator[tuple[Point, Point]]:
    """The minimal surfaces of Bonnet's family between a minimal surface and its harmonic conjugate, one at each
    angle, each with its unit normals at the vertices, the patches of the faces around each vertex summed."""
    corners = mesh.incidence[2, 0]
    for angle in angles:
        surface = vertices * np.cos(angle) + conjugate * np.sin(angle)           # Point-[vertices]
        sides = mesh.boundary[2, 1] * (mesh.d[0] * surface)                  # [corners] Point-[faces]
        normals = (~corners * (sides[0] & sides[1])).sum(axis=0).normalized()   # Patch-[vertices]
        yield surface, normals
