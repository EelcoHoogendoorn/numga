"""Loop subdivision of motors refines a triangle mesh whose vertices carry rigid placements of frames instead of points. A Loop step refines a mesh by averaging: every vertex of the finer mesh is a fixed average of vertices of the coarser one, with shares that depend only on how the mesh is connected. When neighbouring frames are twisted relative to one another, averaging their positions cuts across the chord of the turn and wrings the surface thin, whereas averaging their rigid placements and normalizing back to a rigid motion interpolates along the arc between them and keeps the surface wide.

Each placement is a motor, and one Loop step is a linear map from the coarser mesh's vertex sites to the finer one's that applies to whatever sits there, `step * motors`. A weighted sum of motors is not a motor, so every averaged motor is normalized back after each step, `(step * motors).normalized()`: average in the algebra, return to the group. The surface is the origin each motor carries, `motors >> origin`.

In the matrix and dual-quaternion notation of subdivision surfaces, a Loop step reads as a subdivision matrix $S$ acting on the stacked vertex positions $P' = S P$, while motor subdivision applies the same $S$ to the stacked unit dual quaternions and normalizes each row back to unit norm.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D as ga
from examples import instantiate

ctx = NumpyContext(ga)
mv = ctx.multivector
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, SubdividedMesh = mesh_module.Mesh, mesh_module.SubdividedMesh
Line = ga.gatype.bivector()                         # Bivector+
Point = ga.gatype.antivector()                      # Antivector-
Motor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def subdivide_motors(mesh: SubdividedMesh, motors: Motor) -> Motor:
    """Motor subdivision averages the motors at every Loop step of the mesh and normalizes each average back to a motor."""
    for step in mesh.steps:
        motors = (step * motors).normalized()                                  # [...] Motor+[finer]
    return motors


# --- plumbing -------------------------------------------------------------------------
def grid_faces(n_rows: int, n_columns: int) -> np.ndarray:
    """The counter-clockwise triangle faces of a grid of `n_rows` by `n_columns` vertices, two triangles per cell."""
    index = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
    corner, right, up, opposite = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    return np.concatenate([np.stack([corner, right, opposite], axis=-1).reshape(-1, 3),
                           np.stack([corner, opposite, up], axis=-1).reshape(-1, 3)])
