"""Coupled sparse linear solver bridging two-field numga maps to SciPy's direct sparse solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import spsolve

from numga.gatype import GAType
from numga.sparse import SparseExtensor

if TYPE_CHECKING:
    from numga.extensor import Extensor
    from examples.electromagnetism.piezoelectric.core import Plane, Point, Scalar


def solve_coupled(mechanical: SparseExtensor, coupling: SparseExtensor, dielectric: SparseExtensor,
                  force: Plane, charge: Extensor) -> tuple[Point, Scalar]:
    """Solves the coupled mechanical and electrical work equations for vertex displacements and potentials."""
    Point = mechanical.gatype.derive.axis(1)
    Scalar = dielectric.gatype.derive.axis(1)
    stiffness = _matrix(mechanical, Point)
    mixed = _matrix(coupling, Point)
    electric = _matrix(dielectric, Scalar)

    # Express both equations as scalar work forms on their input fields so the adjugate coupling is the matrix transpose.
    system = bmat([[stiffness, mixed], [mixed.T, electric]], format="csc")
    batch = np.broadcast_shapes(force.shape, charge.shape)
    mechanical_load = (force.batch() & Point).kernel.reshape(force.shape + (-1,))
    electric_load = (Scalar & charge.batch()).kernel.reshape(charge.shape + (-1,))
    rhs = np.concatenate([
        np.broadcast_to(mechanical_load, batch + (stiffness.shape[0],)),
        np.broadcast_to(electric_load, batch + (electric.shape[0],)),
    ], axis=-1)
    solution = spsolve(system, rhs.reshape(-1, system.shape[0]).T)
    solution = solution.reshape(system.shape[1], -1).T.reshape(batch + (system.shape[1],))

    displacement = mechanical.cells.context.extensor(
        Point, solution[..., :stiffness.shape[1]].reshape(batch + (mechanical.shape[1], len(Point.output_subspace))),
    ).field()
    potential = dielectric.cells.context.extensor(
        Scalar, solution[..., stiffness.shape[1]:].reshape(batch + (dielectric.shape[1], len(Scalar.output_subspace))),
    ).field()
    return displacement, potential


def _matrix(value: SparseExtensor, test_space: GAType) -> coo_matrix:
    """Packs a sparse extensor's bilinear pairing blocks into a compressed sparse column matrix."""
    form = test_space & value.cells
    height, width = (len(axis) for axis in form.gatype.input_subspaces)
    blocks = np.asarray(form.kernel).reshape(len(value.rows), height, width)
    rows = np.broadcast_to(value.rows[:, None, None] * height + np.arange(height)[None, :, None], blocks.shape)
    columns = np.broadcast_to(value.columns[:, None, None] * width + np.arange(width)[None, None, :], blocks.shape)
    return coo_matrix(
        (blocks.ravel(), (rows.ravel(), columns.ravel())),
        shape=(value.shape[0] * height, value.shape[1] * width),
    ).tocsc()
