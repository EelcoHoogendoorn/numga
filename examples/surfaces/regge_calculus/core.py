"""Prescribed edge lengths on a flat triangulation determine its intrinsic metric, curvature defects, and steady conduction.

Each triangle's intrinsic metric is modelled as a symmetric duality map `Plane <- Point` built from products of corner simplex gradients so that `metric(edge) & edge` reads each squared edge length. Combining side pairings with twice the intrinsic area in each face's turning plane forms unit corner rotors whose logarithms sum to the vertex angle defects, while cotangent-weighted side dyads assemble steady conduction across the triangulation.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D as ga
from numga.sparse import spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Rotor = ga.gatype.rotor()                                                    # Rotor+
Point = ga.gatype.antivector()                                               # Antivector-
Antiscalar = ga.gatype.antiscalar()                                          # Antiscalar-
Metric = ga.gatype((Plane, Point))                                           # Plane+ <- Point-
Conductivity = ga.gatype((Point, Plane))                                     # Point- <- Plane+
Gradient = ga.gatype((Plane, Scalar))                                        # Plane+ <- Scalar+
Conductance = ga.gatype((Antiscalar, Scalar))                                # Antiscalar- <- Scalar+
triangle = Topology(3, [[0, 1, 2]])


# --- math -----------------------------------------------------------------------------
def metric_basis(topology: Topology, vertices: Point) -> Metric:
    """Each triangle's three symmetric simplex-gradient products (`[3] (Plane+ <- Point-)[F]`) pair to one on their own side and zero on the other two."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    gradients = sides / (2 * patches)                                        # [3] Plane+[F]
    # Along the two sides meeting at a corner, one endpoint gradient vanishes while on the opposite
    # side the two endpoint gradients pair to -1 and +1, making their symmetric product dual to the squared edge lengths.
    tails, heads = topology.corners[2, 1]
    first, second = gradients[tails], gradients[heads]                       # [3] Plane+[F], [3] Plane+[F]
    return -(first * (second & Point) + second * (first & Point)) / 2        # [3] (Plane+ <- Point-)[F]


def intrinsic_areas(topology: Topology, vertices: Point, metrics: Metric) -> Scalar:
    """Each triangle's metric wedge paired with its reference patch determines its intrinsic area (`[...] Scalar+[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    # The wedge carries the duality map's area scaling; pairing it with the reference patch
    # and dividing by two gives the intrinsic area squared.
    return (((metrics(sides[0]) ^ metrics(sides[1])) & patches) / 2).square_root()  # [...] Scalar+[F]


def corner_rotors(topology: Topology, vertices: Point, metrics: Metric, areas: Scalar) -> Rotor:
    """Side pairings and twice the intrinsic area in each face's turning plane form unit corner rotors (`[..., 3] Rotor+[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    turning_planes = patches.normalized().dual_inverse()                     # Bivector+[F]
    tails, heads = topology.corners[2, 1]
    # Combining the metric pairing of the two sides meeting at each corner with twice the intrinsic
    # area in the face's turning plane forms a rotor whose norm is the product of those two side lengths.
    dots = -(metrics[..., None](sides[tails]) & sides[heads])                # [..., 3] Scalar+[F]
    return (dots + 2 * areas[..., None] * turning_planes).normalized()       # [..., 3] Rotor+[F]


def vertex_defects(topology: Topology, vertices: Point, corner_turns: Bivector) -> Bivector:
    """Interior vertices fall short of a full turn and boundary vertices of a half turn by the angle defect (`[...] Bivector+[V]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    turning_planes = ((sides[0] & sides[1]) / 2).normalized().dual_inverse()  # Bivector+[F]
    corner_mean = topology.incidence[2, 0].mean(axis=0)                      # Scalar+[F <- V]
    full_turns = 2 * np.pi * (corner_mean.adjugate() * turning_planes).normalized()  # Bivector+[V]
    # A flat surface turns through a full turn around an interior vertex, and through half of one at the rim.
    rim = topology.reading(topology.exterior[0], 0)                          # Scalar+[V_rim <- V]
    target_turns = full_turns - rim.adjugate() * (rim * full_turns) / 2      # Bivector+[V]
    return target_turns - (topology.incidence[2, 0].adjugate() * corner_turns).sum(axis=-1)  # [...] Bivector+[V]


def triangle_conductivities(topology: Topology, vertices: Point, metrics: Metric, areas: Scalar) -> Conductivity:
    """Corner cotangents weight the side dyads into each triangle's area-weighted inverse metric (`[...] (Point- <- Plane+)[F]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    tails, heads = topology.corners[2, 1]
    dots = -(metrics[..., None](sides[tails]) & sides[heads])                # [..., 3] Scalar+[F]
    # Each side dyad turns a potential slope into flow along that side;
    # the corner cotangents weight the isotropic response of the intrinsic triangle.
    cotangents = dots / (2 * areas[..., None])                               # [..., 3] Scalar+[F]
    return ((cotangents / 2) * (sides * (Plane & sides))).sum(axis=-1)       # [...] (Point- <- Plane+)[F]


def simplex_gradient(topology: Topology, vertices: Point) -> Gradient:
    """Corner simplex gradients assemble into the piecewise-linear gradient operator (`(Plane+ <- Scalar+)[F <- V]`) on the reference triangulation."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)             # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    gradients = sides / (2 * patches)                                        # [3] Plane+[F]
    return (spdiag(gradients) * topology.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[F <- V]


def steady_potential(topology: Topology, conductance: Conductance,
                     free_indices: np.ndarray, boundary_potential: Scalar) -> Scalar:
    """Current balance at every unpinned vertex determines the steady potential (`[...] Scalar+[V]`) from `boundary_potential`."""
    free = topology.reading(free_indices, 0).adjugate() * Scalar             # (Scalar+ <- Scalar+)[V <- V_free]
    rhs = -free.adjugate()(conductance(boundary_potential))                  # [...] Antiscalar-[V_free]
    return boundary_potential + free(free.adjugate()(conductance(free)).solve(rhs))  # [...] Scalar+[V]
