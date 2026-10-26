"""Loop subdivision of motors.

A mesh can carry a motor at every vertex, a rigid placement of a frame, and the Loop step averages them like
anything else at the sites: the same sparse map of shares, applied to the motors as they are. An average of
motors is not a motor, so each step is followed by normalizing every averaged motor back to a motor, and the
next step starts from there: average in the algebra, return to the group. Between two frames turned about an
axis, the average is the frame turned halfway, on the arc between them; the average of the two positions cuts
across the chord instead.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D as ga
from numga.sparse import SparseExtensor
from examples import instantiate

ctx = NumpyContext(ga)
mv = ctx.multivector
Mesh = instantiate("examples.util.mesh", ga).Mesh
Plane = ga.gatype.vector()
Line = ga.gatype.bivector()
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def refinements(mesh: Mesh, motors: Motor, pinned: np.ndarray, depth: int) -> Iterator[tuple[SparseExtensor, Mesh, Motor]]:
    """Each of so many Loop steps on a mesh with motors at its vertices, with the vertices pinned `[V]` staying
    where they are: the averaging map, built once from the faces, the finer mesh, and its motors, every averaged
    motor normalized back to a motor."""
    for _ in range(depth):
        step = mesh.loop(pinned)                                               # [finer, coarser] Scalar
        motors = (step * motors).normalized()                                  # Motor[finer]
        mesh, pinned = Mesh(step.shape[0], mesh.split_faces), mesh.split_pins(pinned)
        yield step, mesh, motors


# --- plumbing -------------------------------------------------------------------------
def grid_faces(rows: int, columns: int) -> np.ndarray:
    """The faces of a grid of `rows` by `columns` vertices, row by row, two triangles per cell, all turning
    the same way."""
    index = np.arange(rows * columns).reshape(rows, columns)
    corner, right, up, across = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    return np.concatenate([np.stack([corner, right, across], axis=-1).reshape(-1, 3),
                           np.stack([corner, across, up], axis=-1).reshape(-1, 3)])
