"""The coupled solve for the stream function and the potential the liquid adds: an excursion from numga to SciPy,
scoped to one function that takes numga's maps and loads and returns numga's fields. numga's sparse solve takes a map
of one field type, and this system couples two, with the stream function tied along one wall and the potential free
up to a constant, so the blocks leave the algebra as one sparse matrix and come back as fields."""

from __future__ import annotations

import numpy as np
from scipy.sparse import bmat, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from numga.extensor import Extensor
from numga.sparse import SparseExtensor


def solve_coupled(viscous: SparseExtensor, coupling: SparseExtensor, conduction: SparseExtensor, flow_load: Extensor,
                  current_load: Extensor, free: np.ndarray, upper: np.ndarray, pinned: int) -> tuple[Extensor, Extensor]:
    """The stream function and the potential `[...] Scalar+[vertices]` where the viscous resistance to the flow balances
    the work the drives do on it, `viscous(stream) - coupling(potential) == flow_load`, and every potential's drive
    reads the channel's flux as nothing, `-coupling.adjugate()(stream) - conduction(potential) == current_load`. The
    stream function is free at the vertices `free`, nothing on the lower wall and one value, the volume flowing between
    the walls, along the upper wall `upper`; the potential is nothing at the vertex `pinned`. The maps' leading axes and
    the loads' broadcast against one another, one solve for each."""
    n_vertices = viscous.shape[1]
    # The stream function's unknowns as vertex values: one per free vertex, and one shared along the upper wall.
    tied = csc_matrix(coo_matrix((np.ones(len(free) + len(upper)),
                                  (np.concatenate([free, upper]),
                                   np.concatenate([np.arange(len(free)), np.full(len(upper), len(free))]))),
                                 shape=(n_vertices, len(free) + 1)))
    kept = np.setdiff1d(np.arange(n_vertices), [pinned])
    gauged = csc_matrix(coo_matrix((np.ones(len(kept)), (kept, np.arange(len(kept)))), shape=(n_vertices, len(kept))))
    batch = np.broadcast_shapes(_batch(viscous), _batch(coupling), _batch(conduction), flow_load.shape,
                                current_load.shape)
    flows, currents = _values(flow_load, batch), _values(current_load, batch)
    streams, potentials = np.empty(batch + (n_vertices,)), np.empty(batch + (n_vertices,))
    for case in np.ndindex(batch):
        resistance, mixed, conductance = (_matrix(value, batch, case) for value in (viscous, coupling, conduction))
        system = bmat([[tied.T @ resistance @ tied, -tied.T @ mixed @ gauged],
                       [-gauged.T @ mixed.T @ tied, -gauged.T @ conductance @ gauged]], format="csc")
        solved = spsolve(system, np.concatenate([tied.T @ flows[case], gauged.T @ currents[case]]))
        streams[case] = tied @ solved[:tied.shape[1]]
        potentials[case] = gauged @ solved[tied.shape[1]:]
    context = flow_load.context.multivector
    return context.scalar(streams[..., None]).field(), context.scalar(potentials[..., None]).field()


def _batch(value: SparseExtensor) -> tuple[int, ...]:
    return value.cells.shape[:-1]


def _values(field: Extensor, batch: tuple[int, ...]) -> np.ndarray:
    """A scalar field's values `[..., vertices]`, broadcast to the batch."""
    values = np.asarray(field.batch().to_array())
    return np.broadcast_to(values, batch + values.shape[-1:])


def _matrix(value: SparseExtensor, batch: tuple[int, ...], case: tuple[int, ...]):
    """One map of scalar cells, at a case of the batch, as a sparse matrix."""
    cells = np.asarray(value.cells.kernel)
    cells = np.broadcast_to(cells.reshape(value.cells.shape + (-1,))[..., 0], batch + value.cells.shape[-1:])[case]
    return csc_matrix(coo_matrix((cells, (value.rows, value.columns)), shape=value.shape))
