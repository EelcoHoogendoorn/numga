"""Numeric helpers for locating points in tetrahedral charts, reading sparse map scales, and building orbits and hulls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor
from numga.topology import Topology

from examples.util import shapes

ga = Algebra("x+y+z+w+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Plane = ga.gatype.vector()                                                     # Vector+
Point = ga.gatype.antivector()                                                 # Antivector-
Motor = ga.gatype.rotor()                                                      # Rotor+


def picking(topology: Topology, vertices: Point) -> Callable[[Point], np.ndarray]:
    """A k-d tree locator maps each sphere point `[...] Point-` to the index `[...]` of the tetrahedron holding it."""
    from scipy.spatial import cKDTree

    corners = topology.incidence[3, 0] * vertices                              # [4] Point-[T]
    facing = topology.off[3, 0] * vertices                                     # [3, 4] Point-[T]
    facets = facing[0] & facing[1] & facing[2]                                 # [4] Plane-[T]
    gradients = facets / (facets & corners)                                    # [4] Plane+[T]
    coordinate_planes = mv.vector(np.eye(4))                                   # [4] Plane+
    radial = gradients.sum(axis=0).batch()                                     # [T] Plane+
    planes = (radial[:, None] & coordinate_planes.dual()[None]).to_array()     # [T, 4]
    sizes = (planes ** 2).sum(axis=-1)                                         # [T]
    tree = cKDTree(np.concatenate([planes, np.sqrt(sizes.max() - sizes)[:, None]], axis=-1))

    def located(points: Point) -> np.ndarray:
        coordinates = (coordinate_planes & points.batch()[..., None]).to_array()   # [..., 4]
        return tree.query(np.pad(coordinates, [(0, 0)] * (coordinates.ndim - 1) + [(0, 1)]))[1]

    return located


def largest(value: SparseExtensor) -> float:
    """The largest absolute entry in `value`'s sparse block array sets the regularization scale."""
    return float(np.abs(np.asarray(value.cells.kernel)).max())


def scalar_value(value: Scalar) -> float:
    """Plain floating-point readout of a nullary `Scalar+` extensor sets scalar time-step parameters."""
    return float(value.to_array())


def vertex_orbits(vertices: Point, turns: Motor, n_sides: int) -> np.ndarray:
    """Repeated powers of `turns` carry `vertices` through vertex-index orbits of shape `[..., n_sides, V]`."""
    from scipy.spatial import cKDTree

    coordinate_planes = mv.vector(np.eye(4))                                   # [4] Plane+
    tree = cKDTree((coordinate_planes & vertices.batch()[:, None]).to_array())
    turned_vertices = (turns[:, None] >> vertices.batch()[None]).reshape(-1)   # [...] Point-
    following = tree.query((coordinate_planes & turned_vertices[:, None]).to_array())[1].reshape(turns.shape[0], -1)
    steps = [np.broadcast_to(np.arange(following.shape[1]), following.shape)]
    for _ in range(n_sides - 1):
        steps.append(np.take_along_axis(following, steps[-1], axis=-1))
    return np.stack(steps, axis=1)


def base_hull(steps: Point) -> Topology:
    """The convex hull of unit origin steps `[...] Point-` forms a closed triangular base `Topology`."""
    coordinates = (stack((mv.x, mv.y, mv.z)) & steps[:, None]).to_array()     # [..., 3]
    topology, _ = shapes.convex_boundary(coordinates)
    return topology
