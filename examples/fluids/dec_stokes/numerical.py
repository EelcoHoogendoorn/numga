"""The sparse constrained solve, returning the two fields in their algebraic types."""

from __future__ import annotations

import numpy as np
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import splu

from numga.extensor import Extensor
from numga.sparse import SparseExtensor


def solve_stokes(stiffness: SparseExtensor, divergence: SparseExtensor,
                 force: Extensor, balance: Extensor) -> tuple[Extensor, Extensor]:
    """Fluxes and pressure from viscous resistance and exact cellwise flux balance."""
    context = stiffness.cells.context
    Scalar = context.algebra.gatype.scalar()
    one = context.multivector.scalar([1])
    # Pair each force with its scalar flux argument before entering the sparse solve.
    work = Scalar & stiffness.cells
    K = coo_matrix((np.asarray(work.kernel).reshape(-1), (stiffness.rows, stiffness.columns)),
                   shape=stiffness.shape).tocsc()
    D = coo_matrix((np.asarray(divergence.cells.kernel).reshape(-1),
                    (divergence.rows, divergence.columns)), shape=divergence.shape).tocsc()
    # One pressure is the gauge; the omitted cell balance is implied by all the others
    # and the prescribed total boundary flux.
    reduced = D[1:]
    system = bmat([[K, -reduced.T], [-reduced, None]], format="csc")
    loads = np.asarray((one & force).kernel)[..., 0]
    target = np.asarray(balance.kernel)[..., 0]
    rhs = np.concatenate((loads, -target[..., 1:]), axis=-1)
    solved = splu(system).solve(rhs.reshape(-1, system.shape[0]).T).T.reshape(rhs.shape)
    flux = context.multivector.scalar(solved[..., :stiffness.shape[1], None]).field()
    pressure = np.concatenate((np.zeros(solved.shape[:-1] + (1,)), solved[..., stiffness.shape[1]:]), axis=-1)
    return flux, context.multivector.scalar(pressure[..., None]).field()
