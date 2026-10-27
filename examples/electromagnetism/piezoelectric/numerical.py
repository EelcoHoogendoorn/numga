"""The coupled solve for displacement and electric potential: an excursion from numga to SciPy, scoped to one
function that takes numga's maps and loads and returns numga's fields. numga's sparse solve takes a map of one field
type, and this system couples two, so the blocks leave the algebra as one sparse matrix and come back as fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import spsolve

from numga.gatype import GAType
from numga.sparse import SparseExtensor

if TYPE_CHECKING:
    from .core import Amount, Plane, Point, Scalar


def solve_coupled(mechanical: SparseExtensor, coupling: SparseExtensor, dielectric: SparseExtensor,
                  force: Plane, charge: Amount) -> tuple[Point, Scalar]:
    """The displacement and potential that solve the coupled work equations, with unbatched maps and any leading load
    batches: the maps and loads go in as numga values and the fields come out as numga fields, and only in between
    does the system leave the algebra, for SciPy's sparse solve. The dielectric block carries its electric-enthalpy
    sign; the reverse coupling is the adjugate. Essential boundary conditions have already been applied to the maps."""
    Point = mechanical.gatype.derive.axis(1)
    Scalar = dielectric.gatype.derive.axis(1)
    stiffness = _matrix(mechanical, Point)
    mixed = _matrix(coupling, Point)
    electric = _matrix(dielectric, Scalar)

    # Express both equations as work on their input fields. In these scalar forms, the adjugate
    # of the mixed map is exactly the transpose, with no choice of a displacement metric.
    system = bmat([[stiffness, mixed], [mixed.T, electric]], format="csc")
    batch = np.broadcast_shapes(force.shape, charge.shape)
    mechanical_load = (Point & force.batch()).kernel.reshape(force.shape + (-1,))
    electric_load = (Scalar & charge.batch()).kernel.reshape(charge.shape + (-1,))
    right = np.concatenate([
        np.broadcast_to(mechanical_load, batch + (stiffness.shape[0],)),
        np.broadcast_to(electric_load, batch + (electric.shape[0],)),
    ], axis=-1)
    solved = spsolve(system, right.reshape(-1, system.shape[0]).T)
    solved = solved.reshape(system.shape[1], -1).T.reshape(batch + (system.shape[1],))

    displacement = mechanical.cells.context.extensor(
        Point, solved[..., :stiffness.shape[1]].reshape(batch + (mechanical.shape[1], len(Point.output_subspace))),
    ).field()
    potential = dielectric.cells.context.extensor(
        Scalar, solved[..., stiffness.shape[1]:].reshape(batch + (dielectric.shape[1], len(Scalar.output_subspace))),
    ).field()
    return displacement, potential


def _matrix(value: SparseExtensor, test_space: GAType):
    """A map's scalar work form as a sparse matrix, one block per pair of sites."""
    form = test_space & value.cells
    height, width = (len(axis) for axis in form.gatype.input_subspaces)
    blocks = np.asarray(form.kernel).reshape(len(value.rows), height, width)
    rows = np.broadcast_to(value.rows[:, None, None] * height + np.arange(height)[None, :, None], blocks.shape)
    columns = np.broadcast_to(value.columns[:, None, None] * width + np.arange(width)[None, None, :], blocks.shape)
    return coo_matrix(
        (blocks.ravel(), (rows.ravel(), columns.ravel())),
        shape=(value.shape[0] * height, value.shape[1] * width),
    ).tocsc()
