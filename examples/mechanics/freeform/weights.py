"""Reproducing kernels and scalar skinning modes for mesh-free deformation."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, Orientation, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Points and displacements are antivectors; gradients are covectors, vectors.
Antivector = ga.gatype.antivector()
Vector = ga.gatype.vector()
# A point with an intrinsic one prepended, on which affine functions are linear.
AffinePoint = ga.gatype(ga.subspace.from_grades([0, 2]), orientation=Orientation.intrinsic)
Evaluation = ga.gatype((Scalar, Scalar))
Gradient = ga.gatype((Vector, Scalar))
planes = stack((mv.x, mv.y, mv.z))                           # [3] Vector
directions = planes.dual()                                   # [3] Antivector
one = mv(Scalar.derive.flipped_orientation, [1.0])          # [] intrinsic Scalar


# --- math

def kernels(queries: Antivector, centers: Antivector, radius: float) -> tuple[Evaluation, Gradient]:
    """Maps from scalar nodal values to values and spatial gradients at query points."""
    points = one + centers                                       # [nodes] AffinePoint
    difference = queries[..., None] - centers                    # [queries, nodes] Antivector
    gaussian = (-difference.scalar_norm_squared() / radius**2).exp()  # [queries, nodes] Scalar

    # Correct the Gaussian kernels so their weighted sum reproduces every affine function.
    moment = (gaussian * (points * points.scalar_product(AffinePoint))).sum(axis=-1)  # [queries] AffinePoint <- AffinePoint
    correction = moment.solve(one + queries)                     # [queries] AffinePoint
    reproduction = points.scalar_product(correction[..., None])  # [queries, nodes] Scalar
    weights = gaussian * reproduction                           # [queries, nodes] Scalar

    # Differentiating the moment equation accounts for the correction changing with position.
    gaussian_derivative = gaussian * (-2 / radius**2) * difference.dot(Antivector)  # [queries, nodes] Scalar <- Antivector
    moment_change = (points * gaussian_derivative * reproduction).sum(axis=-1)  # [queries] AffinePoint <- Antivector
    correction_derivative = moment.solve(Antivector - moment_change)  # [queries] AffinePoint <- Antivector
    derivatives = gaussian_derivative * reproduction + gaussian * points.scalar_product(correction_derivative[..., None])  # [queries, nodes] Scalar <- Antivector
    # Each weight's gradient: the covector that reads its change along every direction.
    gradients = (planes * derivatives[..., None](directions)).sum(axis=-1)   # [queries, nodes] Vector

    # The node axis becomes the input field of the interpolation and gradient maps.
    return (weights * Scalar).field(1), (gradients * Scalar).field(1)


def modes(evaluation: Evaluation, gradient: Gradient, sample_volumes: np.ndarray,
          lame_lambda: np.ndarray, lame_mu: np.ndarray, n_modes: int) -> tuple[Scalar, Scalar]:
    """The lowest scalar skinning modes, orthonormal under volume integration."""
    # Assembling over the quadrature sites integrates all pairs of nodal responses together.
    values_at_samples = evaluation.field(0)                      # Scalar[samples] <- Scalar[nodes]
    gradients_at_samples = gradient.field(0)                     # Vector[samples] <- Scalar[nodes]
    mass = values_at_samples.adjoint()((evaluation * sample_volumes).field(0))  # Scalar[nodes] <- Scalar[nodes]
    stiffness = gradients_at_samples.adjoint()(
        (gradient * sample_volumes * (lame_lambda + 4 * lame_mu)).field(0)
    )                                                            # Scalar[nodes] <- Scalar[nodes]
    eigenvalues, fields = stiffness.eigh(mass)                    # [nodes] Scalar, Scalar[nodes]
    return eigenvalues[:n_modes], fields[:n_modes]
