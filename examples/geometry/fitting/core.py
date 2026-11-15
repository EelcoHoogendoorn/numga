"""Least-squares fitting of projective primitives to noisy samples minimizes the squared join residual in one eigenproblem.

A point, a line, and a plane are fitted to noisy samples with the same three steps: joining the samples with the unknown as a linear map on its type, squaring and summing that residual into a quadratic form, and selecting its eigenvector of smallest eigenvalue. The unit constraint uses the scalar part of the unknown's reverse product, which is degenerate on the position components that least squares leaves free. Because the join treats points and lines symmetrically, fitting a point to a bundle of lines triangulates their point of closest approach with the same function.
"""

from __future__ import annotations

from numga import Extensor, GAType, NumpyContext
from numga.algebras import PGA3D


ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()                  # Antivector-
# A line through points is their join, intrinsic like them.
Line = ga.gatype.antibivector()                 # Antibivector-
Plane = ga.gatype.vector()                      # Vector+


def fitted_primitive(Unknown: GAType, samples: Extensor) -> Extensor:
    """The primitive of type `Unknown` fitted to `samples` (`[n_samples]`) is the flattest direction of their summed squared join residual."""
    samples = samples.normalized()
    # Joining each sample with the unknown measures its incidence error:
    # point & plane is a scalar, point & line a plane, point & point a line.
    residual = samples & Unknown
    misfit = residual.scalar_norm_squared().sum(axis=0)

    # The unknown's own metric, the scalar part of the primitive's reverse product, fixes its
    # geometric size, leaving its position free: plane normal, line direction, or point weight
    # has unit norm. For a line it leaves out the condition to be a line, the pseudoscalar part;
    # the fitted moment is the centroid's moment about the fitted direction, which meets that
    # condition. The metric is blind to the ideal components, so for each mode eigh chooses them
    # to minimize the misfit:
    eigenvalues, modes = misfit.eigh()
    return modes[eigenvalues.argmin()]
