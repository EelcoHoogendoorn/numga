"""Weighted Whitney fields carry steady diffusion on a fixed physical surface.

Positive vertex scales choose a rational interpolation on the same projective triangles. The surface measure and conductivity specify the physics independently of those scales.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, stack
from numga.backend.context import Context
from numga.sparse import spdiag
from examples import instantiate

# Instantiated in spherical or degenerate planar geometry.
ga: Algebra
ctx: Context
mv = ctx.multivector
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar = mesh_module.Mesh, mesh_module.as_scalar
Scalar = ga.gatype.scalar()                         # Scalar+
Plane = ga.gatype.vector()                          # Vector+
Point = ga.gatype.antivector()                      # Antivector-
Patch = ga.gatype.antibivector()                    # Antibivector-
Volume = ga.gatype.antitrivector()                  # Antitrivector-
Amount = ga.gatype.antiscalar()                     # Antiscalar-
Conductivity = ga.gatype((Point, Plane))


# --- math -----------------------------------------------------------------------------
def diffusion(mesh: Mesh, corners: Point, samples: Point, measures: Scalar,
              source: Scalar, conductivity: Conductivity) -> tuple[Scalar, Plane]:
    """The vertex potential solves steady diffusion under a prescribed source on a closed surface; the face planes interpolate it.

    Samples and positive quadrature measures belong to the physical surface; only the interpolation depends on the lifted corners.
    """
    first, second, third = corners                                   # Point-[faces], Point-[faces], Point-[faces]
    volume = first & second & third                                  # Volume-[faces]
    reciprocals = stack((second & third, third & first, first & second)) / volume  # [corners] Plane+[faces]
    section = reciprocals.sum(axis=0)                                # Plane+[faces]
    denominator = section & samples                                  # [samples] Scalar+[faces]
    hats = (reciprocals[None, :] & samples[:, None]) / denominator[:, None]  # [samples, corners] Scalar+[faces]
    # The derivative of the quotient annihilates the radial direction.
    gradients = (reciprocals[None, :] - hats * section) / denominator[:, None]  # [samples, corners] Plane+[faces]
    interpolation = (spdiag(hats * Scalar) * mesh.incidence[2, 0]).sum(axis=1)  # [samples] (Scalar+ <- Scalar+)[faces <- vertices]
    gradient = (spdiag(gradients * Scalar) * mesh.incidence[2, 0]).sum(axis=1)  # [samples] (Plane+ <- Scalar+)[faces <- vertices]

    # Pull the physical material response back to vertex potentials, integrating on the surface.
    response = spdiag(measures * conductivity)                       # [samples] (Point- <- Plane+)[faces <- faces]
    stiffness = gradient.adjugate()(response(gradient)).sum(axis=0)  # (Amount- <- Scalar+)[vertices <- vertices]
    load = interpolation.adjugate()((measures * source).dual()).sum(axis=0)  # Amount-[vertices]
    mass = interpolation.adjugate()(measures.dual()).sum(axis=0)     # Amount-[vertices]
    # A closed surface admits a solution only for balanced sources; remove the quadrature mean.
    load = load - mass * (load.dual_inverse().sites.sum() / mass.dual_inverse().sites.sum())
    pin = spdiag(as_scalar(np.arange(mesh.n_vertices) == 0).field() * Scalar.dual())  # (Amount- <- Scalar+)[vertices <- vertices]
    potential = (stiffness + pin).solve(load)                        # Scalar+[vertices]
    potential = potential - (potential & mass).sites.sum() / mass.dual_inverse().sites.sum()
    numerator = (reciprocals * (mesh.incidence[2, 0] * potential)).sum(axis=0)  # Plane+[faces]
    return potential, numerator
