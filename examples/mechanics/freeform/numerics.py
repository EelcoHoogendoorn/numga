"""Typed L-BFGS minimization boundary for reduced implicit elastic time steps."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from examples.mechanics.freeform.core import Scalar, Antivector, Vector, Mass


def minimized_state(initial: Antivector, objective: Callable[[Antivector], tuple[Scalar, Vector]],
                    metric: Mass, n_iterations: int, tolerance: float) -> Antivector:
    """Whitening `initial` by Cholesky factorization of `metric` minimizes `objective` with L-BFGS-B and maps the solution back to `Antivector-`."""
    shape = initial.kernel.shape
    n_variables = initial.kernel.size
    # A field of directions has site and blade axes on both sides of its metric.
    matrix = metric.kernel.transpose(0, 2, 1, 3).reshape(n_variables, n_variables)
    lower = np.linalg.cholesky(matrix)

    def evaluate(coordinates: np.ndarray) -> tuple[float, np.ndarray]:
        values = np.linalg.solve(lower.T, coordinates).reshape(shape)
        state = initial.context.extensor(initial.gatype, values)
        energy, gradient = objective(state)
        return float(energy.to_array()), np.linalg.solve(lower, gradient.kernel.reshape(n_variables))

    result = minimize(evaluate, lower.T @ initial.kernel.reshape(n_variables), jac=True,
                      method="L-BFGS-B", options={"maxiter": n_iterations, "ftol": tolerance, "gtol": tolerance})
    return initial.context.extensor(initial.gatype, np.linalg.solve(lower.T, result.x).reshape(shape))
