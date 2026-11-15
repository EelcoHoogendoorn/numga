"""SciPy solves the coupled sparse linear system for the stream function and induced electric potential across channel vertices. Because the stream function shares one unknown flux value along the upper wall while the potential is pinned at a reference vertex, the four scalar operator blocks are assembled into one sparse matrix and returned as scalar fields."""

from __future__ import annotations

import numpy as np
from scipy.sparse import bmat, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from numga.extensor import Extensor
from numga.sparse import SparseExtensor


def coupled_solution(
    viscous: SparseExtensor,
    coupling: SparseExtensor,
    conduction: SparseExtensor,
    flow_load: Extensor,
    current_load: Extensor,
    free_vertices: np.ndarray,
    upper_wall: np.ndarray,
    pinned_vertex: int,
) -> tuple[Extensor, Extensor]:
    """The coupled Stokes and current-balance equations `viscous(stream) - coupling(potential) == flow_load` and `-coupling.adjugate()(stream) - conduction(potential) == current_load` determine the stream function and induced potential (`[...] Scalar+[V]`), holding the lower wall at zero, tying `upper_wall` to one shared volume flux, and pinning `pinned_vertex` at zero potential."""
    n_vertices = viscous.shape[1]
    # Free interior vertices each take an unknown stream value, while upper-wall vertices share one total-flux unknown.
    n_free, n_upper = len(free_vertices), len(upper_wall)
    stream_basis = csc_matrix(coo_matrix(
        (np.ones(n_free + n_upper),
         (np.concatenate([free_vertices, upper_wall]), np.concatenate([np.arange(n_free), np.full(n_upper, n_free)]))),
        shape=(n_vertices, n_free + 1)))
    unpinned = np.setdiff1d(np.arange(n_vertices), [pinned_vertex])
    potential_basis = csc_matrix(coo_matrix(
        (np.ones(len(unpinned)), (unpinned, np.arange(len(unpinned)))),
        shape=(n_vertices, len(unpinned))))
    batch = np.broadcast_shapes(
        _batch_shape(viscous), _batch_shape(coupling), _batch_shape(conduction), flow_load.shape, current_load.shape)
    flows, currents = _field_values(flow_load, batch), _field_values(current_load, batch)
    streams = np.empty(batch + (n_vertices,))
    potentials = np.empty(batch + (n_vertices,))
    for case in np.ndindex(batch):
        resistance, mixed, conductance = (_case_matrix(op, batch, case) for op in (viscous, coupling, conduction))
        system = bmat([
            [stream_basis.T @ resistance @ stream_basis, -stream_basis.T @ mixed @ potential_basis],
            [-potential_basis.T @ mixed.T @ stream_basis, -potential_basis.T @ conductance @ potential_basis],
        ], format="csc")
        rhs = np.concatenate([stream_basis.T @ flows[case], potential_basis.T @ currents[case]])
        solution = spsolve(system, rhs)
        streams[case] = stream_basis @ solution[:stream_basis.shape[1]]
        potentials[case] = potential_basis @ solution[stream_basis.shape[1]:]
    factory = flow_load.context.multivector
    return factory.scalar(streams[..., None]).field(), factory.scalar(potentials[..., None]).field()


def _batch_shape(operator: SparseExtensor) -> tuple[int, ...]:
    """Leading batch dimensions precede the sparse nonzero axis of `operator`."""
    return operator.cells.shape[:-1]


def _field_values(field: Extensor, batch: tuple[int, ...]) -> np.ndarray:
    """Array readout broadcasts a scalar vertex field (`[...] Scalar+[V]`) to `batch + (V,)`."""
    values = np.asarray(field.batch().to_array())
    return np.broadcast_to(values, batch + values.shape[-1:])


def _case_matrix(operator: SparseExtensor, batch: tuple[int, ...], case: tuple[int, ...]) -> csc_matrix:
    """Entry extraction at `case` converts a scalar `SparseExtensor` into a sparse CSC matrix."""
    entries = np.asarray(operator.cells.to_array())
    entries = np.broadcast_to(entries, batch + operator.cells.shape[-1:])[case]
    return csc_matrix(coo_matrix((entries, (operator.rows, operator.columns)), shape=operator.shape))
