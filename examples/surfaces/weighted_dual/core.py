"""Polar duality about the unit sphere relates polyhedral face planes, dual poles, dihedral folds, and radial vertex weights.

Points around the origin span a triangulated polyhedron whose face planes each pair to one with their three corners.
Taking the dual of each face plane about the unit sphere yields the face pole, and projecting both radially onto the
sphere pairs a spherical triangulation with its polar dual complex. Varying each vertex's inverse distance along its
fixed ray tilts the incident faces while keeping both the dihedral folds and the face planes linear in those weights.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA3D as ga
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# Points and ray directions from the centre are intrinsic antivectors.
Point = ga.gatype.antivector()                      # Antivector-


def spiral_directions(n_points: int, jitter: float, seed: int) -> Point:
    """Unit ray directions (`Point-[V]`) follow a Fibonacci spiral perturbed by Gaussian angular jitter."""
    turns = np.arange(n_points) + 0.5                                          # [n_points]
    polar = np.arccos(1 - 2 * turns / n_points)                                # [n_points]
    azimuth = np.pi * (1 + 5**0.5) * turns                                     # [n_points]
    # Turn the north pole toward +x by the polar angle and around the z axis by the azimuth.
    polar_rotors = (mv.zx * (-polar / 2)).exp()                                # [n_points] Rotor+
    azimuth_rotors = (mv.xy * (-azimuth / 2)).exp()                            # [n_points] Rotor+
    spiral = (azimuth_rotors * polar_rotors) >> mv.z.dual()                    # [n_points] Point-
    nudges = mv.vector(np.random.default_rng(seed).normal(size=(n_points, 3))).dual()  # [n_points] Point-
    return (spiral + jitter * nudges).normalized().field()                     # Point-[V]


def face_poles(topology: Topology, points: Point) -> Point:
    """Each face plane sums its three corner simplex gradients, and its dual about the unit sphere gives the face pole (`Point-[F]`)."""
    corners = topology.incidence[2, 0] * points                                # [3] Point-[F]
    opposite_corners = topology.off[2, 0] * points                             # [2, 3] Point-[F]
    edge_planes = opposite_corners[0] & opposite_corners[1]                    # [3] Plane-[F]
    gradients = edge_planes / (edge_planes & corners)                          # [3] Plane+[F]
    return gradients.sum(axis=0).dual()                                        # Point-[F]


def pole_sides(topology: Topology, points: Point, duals: Point) -> Scalar:
    """Pairings with the three central edge planes (`[3] Scalar+[F]`) are all positive when each face pole projects inside its triangle."""
    corners = topology.incidence[2, 0] * points                                # [3] Point-[F]
    opposite_corners = topology.off[2, 0] * points                             # [2, 3] Point-[F]
    walls = opposite_corners[0] & opposite_corners[1]                          # [3] Plane-[F]
    return (walls & duals) * (walls & corners)                                 # [3] Scalar+[F]


def breath_states(
    topology: Topology, directions: Point, amplitude: float, phases: np.ndarray, n_frames: int
) -> Iterator[tuple[Topology, Point, Point]]:
    """Yields `(topology, points, duals)` across `n_frames` as vertices oscillate radially along fixed `directions` (`Point-[V]`)."""
    for angle in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        radii = mv.scalar((1 + amplitude * np.sin(angle + phases))[:, None]).field()  # Scalar+[V]
        points = directions * radii                                            # Point-[V]
        yield topology, points, face_poles(topology, points)


def edge_folds(topology: Topology, directions: Point, weights: Scalar) -> Scalar:
    """Signed dihedral folds (`[...] Scalar+[E_interior]`) across interior edges vary linearly with the four incident vertex weights."""
    interior = topology.reading(topology.interior[1], 1)                       # Scalar+[E_interior <- E]
    # Gather each interior edge's endpoints together with the opposite corners of its left and right triangles.
    ends = interior * topology.incidence[1, 0]                                 # [2] Scalar+[E_interior <- V]
    opposites = topology.reading(topology.facet_opposites.T, 0)                # [2] Scalar+[E_interior <- V]
    tail, head = ends * directions                                             # Point-[E_interior], Point-[E_interior]
    left_corner, right_corner = opposites * directions                         # Point-[E_interior], Point-[E_interior]
    left_volume = tail & head & left_corner                                    # Scalar-[E_interior]
    volumes = stack([
        head & left_corner & right_corner,
        tail & left_corner & right_corner,
        tail & head & right_corner,
        left_volume,
    ])                                                                         # [4] Scalar-[E_interior]
    # Weight each three-ray volume with alternating signs so the four-point determinant stays linear in the weights.
    corner_weights = concatenate([ends * weights[..., None], opposites * weights[..., None]], axis=-1)  # [..., 4] Scalar+[E_interior]
    signs = np.array([-1, 1, -1, 1])                                           # [4]
    return (volumes * (corner_weights * signs)).sum(axis=-1) * left_volume     # [...] Scalar+[E_interior]


def pole_insides(topology: Topology, directions: Point, weights: Scalar) -> Scalar:
    """Signed distances (`[..., 3] Scalar+[F]`) of each face pole inside its three central edge planes vary linearly with `weights`."""
    corners = topology.incidence[2, 0] * directions                            # [3] Point-[F]
    opposite_corners = topology.off[2, 0] * directions                         # [2, 3] Point-[F]
    edge_planes = opposite_corners[0] & opposite_corners[1]                    # [3] Plane-[F]
    gradients = edge_planes / (edge_planes & corners)                          # [3] Plane+[F]
    duals = (gradients * (topology.incidence[2, 0] * weights[..., None])).sum(axis=-1).dual()  # [...] Point-[F]
    walls = edge_planes.normalized()                                           # [3] Plane-[F]
    corner_senses = walls & corners                                            # [3] Scalar-[F]
    return (walls & duals[..., None]) * corner_senses / corner_senses.antinorm()  # [..., 3] Scalar+[F]


def dual_lengths(topology: Topology, points: Point) -> Scalar:
    """Signed steps across interior edges (`Scalar+[E_interior]`) between adjacent face poles turn negative where dual edges cross."""
    interior = topology.reading(topology.interior[1], 1)                       # Scalar+[E_interior <- E]
    tails, heads = interior * (topology.incidence[1, 0] * points)              # Point-[E_interior], Point-[E_interior]
    wall = tails & heads                                                       # Plane-[E_interior]
    right_sense = wall & (topology.reading(topology.facet_opposites[:, 1], 0) * points)  # Scalar-[E_interior]
    # Pulling face poles back by d[1].adjugate() gives left pole minus right pole, negated to step from left to right.
    pole_steps = -(interior * (topology.d[1].adjugate() * face_poles(topology, points)))  # Point-[E_interior]
    return (wall & pole_steps) * right_sense / right_sense.antinorm()          # Scalar+[E_interior]


def rescue_states(
    topology: Topology, directions: Point, start_weights: Scalar, target_weights: Scalar, schedule: np.ndarray
) -> Iterator[tuple[Topology, Point, Point]]:
    """Yields `(topology, points, duals)` as vertex weights (`Scalar+[V]`) interpolate linearly from `start_weights` to `target_weights`."""
    for progress in schedule:
        weights = start_weights + float(progress) * (target_weights - start_weights)  # Scalar+[V]
        points = directions / weights                                          # Point-[V]
        yield topology, points, face_poles(topology, points)
