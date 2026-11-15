"""Reproducing kernels and scalar skinning modes for solid deformation without a volume discretisation."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, Orientation, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Points and displacements are antivectors; spatial gradients are covectors, vectors.
Antivector = ga.gatype.antivector()
Vector = ga.gatype.vector()
# A point with an intrinsic one prepended, on which affine functions are linear.
AffinePoint = ga.gatype(ga.subspace.from_grades([0, 2]), orientation=Orientation.intrinsic)
Evaluation = ga.gatype((Scalar, Scalar))                     # Scalar+ <- Scalar+
Gradient = ga.gatype((Vector, Scalar))                       # Vector+ <- Scalar+
planes = stack((mv.x, mv.y, mv.z))                           # [3] Vector+
directions = planes.dual()                                   # [3] Antivector-
one = mv.antipseudoscalar([1])                               # Antiscalar-


# --- math -----------------------------------------------------------------------------
def kernels(queries: Antivector, centers: Antivector, radius: float) -> tuple[Evaluation, Gradient]:
    """Reproducing kernels map scalar nodal values at `centers` to interpolated values and spatial gradients at `queries`."""
    points = one + centers                                       # [...] AffinePoint-
    difference = queries[..., None] - centers                    # [..., ...] Antivector-
    gaussian = (-difference.scalar_norm_squared() / radius**2).exp()  # [..., ...] Scalar+

    # Correct the Gaussian kernels so their weighted sum reproduces every affine function.
    moment = (gaussian * (points * points.scalar_product(AffinePoint))).sum(axis=-1)  # [...] AffinePoint- <- AffinePoint-
    correction = moment.solve(one + queries)                     # [...] AffinePoint-
    reproduction = points.scalar_product(correction[..., None])  # [..., ...] Scalar+
    weights = gaussian * reproduction                            # [..., ...] Scalar+

    # Differentiating the moment equation accounts for the correction changing with position.
    gaussian_derivative = gaussian * (-2 / radius**2) * (difference.dual() & Antivector)  # [..., ...] Scalar+ <- Antivector-
    moment_change = (points * gaussian_derivative * reproduction).sum(axis=-1)  # [...] AffinePoint- <- Antivector-
    correction_derivative = moment.solve(Antivector - moment_change)  # [...] AffinePoint- <- Antivector-
    derivatives = gaussian_derivative * reproduction + gaussian * points.scalar_product(correction_derivative[..., None])  # [..., ...] Scalar+ <- Antivector-
    # Each weight's gradient is the covector that reads its change along every direction.
    gradients = (planes * derivatives[..., None](directions)).sum(axis=-1)   # [..., ...] Vector+

    # The node axis becomes the input field of the interpolation and gradient maps.
    return (weights * Scalar).field(1), (gradients * Scalar).field(1)


def skinning_modes(evaluation: Evaluation, gradient: Gradient, sample_volumes: np.ndarray,
                   lame_lambda: np.ndarray, lame_mu: np.ndarray,
                   n_modes: int) -> tuple[Scalar, Scalar]:
    """Generalized eigenmodes of the isotropic elastic stiffness against volume mass give the `n_modes` lowest scalar skinning modes (`[n_modes] Scalar+[...]`)."""
    # Assembling over the quadrature sites integrates all pairs of nodal responses together, each pulled back to the
    # nodes by the adjugate: the mass pairs values with sample volumes, and the stiffness pairs gradients with the
    # flux an isotropic material drives.
    values_at_samples = evaluation.field(0)                      # (Scalar+ <- Scalar+)[... <- ...]
    gradients_at_samples = gradient.field(0)                     # (Vector+ <- Scalar+)[... <- ...]
    mass = values_at_samples.adjugate()((sample_volumes * Scalar.dual()(evaluation)).field(0))   # (Antiscalar- <- Scalar+)[... <- ...]
    # Summing the rest elastic costs of three translation directions gives the isotropic flux along each gradient.
    fluxes = sample_volumes * (lame_lambda + 4 * lame_mu) * Vector.dual()(gradient)   # [...] (Antivector- <- Scalar+)[...]
    stiffness = gradients_at_samples.adjugate()(fluxes.field(0))                       # (Antiscalar- <- Scalar+)[... <- ...]
    eigenvalues, mode_fields = stiffness.eigh(mass)              # [...] Scalar+, [...] Scalar+[...]
    return eigenvalues[:n_modes], mode_fields[:n_modes]
