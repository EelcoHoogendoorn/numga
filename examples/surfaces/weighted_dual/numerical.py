"""Numeric escapes compute 3D convex hulls and solve linear and quadratic weight programs via SciPy."""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize

from examples.util import shapes
from numga import stack
from numga.algebras import VGA3D as ga
from numga.topology import Topology

Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-


def convex_hull(points: Point) -> Topology:
    """SciPy builds the outward-oriented triangle `Topology` of the convex hull of `points` (`Point-[V]`)."""
    mv = points.context.multivector
    points_batch = points.batch()                                              # [V] Point-
    n_vertices = points_batch.shape[0]
    coordinates = (stack((mv.x, mv.y, mv.z), axis=-1) & points_batch[..., None]).to_array()  # [n_vertices, 3]
    topology, _ = shapes.convex_boundary(coordinates)
    return topology


def widest_margin(
    fold_columns: Scalar, inside_columns: Scalar, weight_bounds: tuple[float, float]
) -> tuple[Scalar, Scalar]:
    """A linear program finds bounded vertex weights (`Scalar+[V]`) summing to `V` that maximize the minimum normalized condition (`Scalar+`)."""
    mv = fold_columns.context.multivector
    n_vertices = fold_columns.shape[0]
    matrix = np.hstack([
        fold_columns.batch().to_array(),
        inside_columns.batch().to_array().reshape(n_vertices, -1),
    ]).T
    n_conditions = matrix.shape[0]
    unit_rows = matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)        # [n_conditions, n_vertices]
    objective = np.concatenate([np.zeros(n_vertices), [-1.0]])
    total = np.concatenate([np.ones(n_vertices), [0.0]])[None]
    result = linprog(
        objective,
        A_ub=np.hstack([-unit_rows, np.ones((n_conditions, 1))]),
        b_ub=np.zeros(n_conditions),
        A_eq=total,
        b_eq=[n_vertices],
        bounds=[weight_bounds] * n_vertices + [(None, None)],
    )
    return mv.scalar(result.x[:-1, None]).field(), mv.scalar([result.x[-1]])


def nearest_weights(
    fold_columns: Scalar,
    inside_columns: Scalar,
    margin: Scalar,
    start_weights: Scalar,
    weight_bounds: tuple[float, float],
) -> Scalar:
    """Sequential quadratic programming finds bounded weights (`Scalar+[V]`) closest to `start_weights` keeping every normalized condition at least `margin`."""
    mv = fold_columns.context.multivector
    n_vertices = fold_columns.shape[0]
    matrix = np.hstack([
        fold_columns.batch().to_array(),
        inside_columns.batch().to_array().reshape(n_vertices, -1),
    ]).T
    unit_rows = matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)
    initial = start_weights.batch().to_array()                                 # [n_vertices]
    target_margin = float(margin.to_array())
    result = minimize(
        lambda weights: ((weights - initial) ** 2).sum(),
        initial,
        jac=lambda weights: 2 * (weights - initial),
        method="SLSQP",
        constraints=[LinearConstraint(unit_rows, target_margin, np.inf)],
        bounds=[weight_bounds] * n_vertices,
    )
    return mv.scalar(result.x[:, None]).field()
