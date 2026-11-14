r"""Steady heat diffusion on a triangulated sphere whose fields are functions of the rays through its centre.

A point of the sphere is modelled as a ray, any positive multiple of an antivector, and each corner of a triangle stands for its ray by one chosen representative. A corner moved along its ray keeps its place on the sphere but changes the rational interpolation across its triangles, while the heat sources, the surface measure and the conductivity stay those of the fixed unit sphere.

The corner functions are quotients of two readings of a point by planes, `(reciprocals & samples) / (section & samples)`, where each reciprocal plane is the join of the other two corners and `section` is their sum. Their simplex gradients `(reciprocals - barycentrics * section) / (section & samples)` pair to zero with the sample point itself, so no radial motion enters, and no metric is needed until the conductivity maps those gradients to currents. The gradient's adjugate carries the integrated currents back to the vertices, `gradient.adjugate()(response(gradient))`.

In spherical coordinates, steady diffusion reads $\frac{1}{\sin\theta}\,\partial_\theta(\sin\theta\,\partial_\theta T) + \frac{1}{\sin^2\theta}\,\partial_\phi^2 T = -f$, with metric factors that vanish at both poles, and a field interpolated on a flat chord triangle reaches the sphere only through the tangent projection $(\mathbf I - \hat{\mathbf r}\hat{\mathbf r}^{\mathsf T})\nabla$; in homogeneous coordinates the corner functions are the rational barycentric coordinates $\lambda_i(\mathbf x) = w_i\,b_i(\mathbf x)/\sum_j w_j\,b_j(\mathbf x)$ with weights $w_i = 1/s_i$.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.backend.context import Context
from numga.sparse import spdiag
from numga.topology import Topology

# Instantiated in spherical or degenerate planar geometry.
ga: Algebra
ctx: Context
mv = ctx.multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Plane = ga.gatype.vector()                          # Vector+
Point = ga.gatype.antivector()                      # Antivector-
Conductivity = ga.gatype((Point, Plane))            # Antivector- <- Vector+


# --- math -----------------------------------------------------------------------------
def diffusion(topology: Topology, vertices: Point, samples: Point, measures: Scalar,
              source: Scalar, conductivity: Conductivity) -> tuple[Scalar, Plane, Plane]:
    """The vertex potential of steady diffusion under a balanced source on a closed surface, with the face planes whose quotient interpolates it.

    Samples and positive quadrature measures belong to the physical surface; only the interpolation depends on the vertex representatives.
    """
    V, E, F = topology.n_elements
    corners = topology.incidence[2, 0] * vertices                    # [3] Point-[F]
    tails, heads = topology.incidence[1, 0] * vertices               # Point-[E], Point-[E]
    reciprocals = (topology.boundary[2, 1] * (tails & heads)) / (corners[0] & corners[1] & corners[2])  # [3] Plane+[F]
    section = reciprocals.sum(axis=0)                                # Plane+[F]
    denominator = section & samples                                  # [...] Scalar+[F]
    barycentrics = (reciprocals[None, :] & samples[:, None]) / denominator[:, None]  # [..., 3] Scalar+[F]
    # The derivative of the quotient annihilates the radial direction.
    gradients = (reciprocals[None, :] - barycentrics * section) / denominator[:, None]  # [..., 3] Plane+[F]
    interpolation = (spdiag(barycentrics * Scalar) * topology.incidence[2, 0]).sum(axis=1)  # [...] (Scalar+ <- Scalar+)[F <- V]
    gradient = (spdiag(gradients * Scalar) * topology.incidence[2, 0]).sum(axis=1)  # [...] (Plane+ <- Scalar+)[F <- V]

    # The gradient's adjugate pulls the physical material response back to the vertex potentials, integrated on the surface.
    response = measures * conductivity                               # [...] (Point- <- Plane+)[F]
    stiffness = gradient.adjugate()(response(gradient)).sum(axis=0)  # (Antiscalar- <- Scalar+)[V <- V]
    load = interpolation.adjugate()((measures * source).dual()).sum(axis=0)  # Antiscalar-[V]
    mass = interpolation.adjugate()(measures.dual()).sum(axis=0)             # Antiscalar-[V]
    # A closed surface admits a solution only for balanced sources; remove the quadrature mean.
    load = load - mass * (load.dual_inverse().sites.sum() / mass.dual_inverse().sites.sum())
    pin = mv.scalar((np.arange(V) == 0)[..., None]).field() * Scalar.dual()  # (Antiscalar- <- Scalar+)[V]
    potential = (stiffness + pin).solve(load)                        # Scalar+[V]
    potential = potential - (potential & mass).sites.sum() / mass.dual_inverse().sites.sum()
    numerator = (reciprocals * (topology.incidence[2, 0] * potential)).sum(axis=0)  # Plane+[F]
    return potential, numerator, section
