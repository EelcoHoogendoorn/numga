"""A typed boundary to a reduced implicit elastic time step."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from examples.mechanics.freeform.core import Scalar, Antivector, Vector, Mass


def minimize_state(initial: Antivector, objective: Callable[[Antivector], tuple[Scalar, Vector]],
                   metric: Mass, n_iterations: int, tolerance: float) -> Antivector:
    """Minimize in coordinates whitened by a positive-definite control-space metric."""
    shape = initial.kernel.shape
    count = initial.kernel.size
    # A field of directions has site and blade axes on both sides of its metric.
    matrix = metric.kernel.transpose(0, 2, 1, 3).reshape(count, count)
    lower = np.linalg.cholesky(matrix)

    def evaluate(coordinates: np.ndarray) -> tuple[float, np.ndarray]:
        values = np.linalg.solve(lower.T, coordinates).reshape(shape)
        state = initial.context.extensor(initial.gatype, values)
        energy, gradient = objective(state)
        return float(energy.to_array()), np.linalg.solve(lower, gradient.kernel.reshape(count))

    result = minimize(evaluate, lower.T @ initial.kernel.reshape(count), jac=True,
                      method="L-BFGS-B", options={"maxiter": n_iterations, "ftol": tolerance, "gtol": tolerance})
    return initial.context.extensor(initial.gatype, np.linalg.solve(lower.T, result.x).reshape(shape))
