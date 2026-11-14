"""Creases built from point and line coincidences split a sheet of paper into polygons and fold them rigidly into a kabuto helmet.

A crease that brings one point onto another is their perpendicular bisector plane `((first & second) | (first + second)).normalized()`, and a crease that brings one oriented line onto another through their intersection is `((first + second) | intersection).normalized()`. Each crease meets the sheet in a hinge line `crease ^ sheet` whose exponential `(hinge * (angle / 2)).exp()` turns the moving faces, while every polygon edge crossing the crease splits at `(tail & head) ^ crease`. Topological layer heights are carried as ideal direction offsets so closed folds stack cleanly without altering the exact sheet geometry. The construction follows the origami example of ganja.js by Steven De Keninck, https://github.com/enkimute/ganja.js/blob/master/examples/example_pga3d_origami.html.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack, where
from numga.algebras import PGA3D
from numga.sparse import Sparse
from numga.topology import Topology

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                      # Antivector-
Direction = Point.derive.degenerate                 # Antivector-
Line = ga.gatype.antibivector()                     # Antibivector-
Hinge = ga.gatype.bivector()                        # Bivector+
Plane = ga.gatype.vector()                          # Vector+
origin = mv.point([0, 0, 0])                        # [] Point-

PaperState = tuple[Point, np.ndarray, Direction, Plane]


# --- math -----------------------------------------------------------------------------
def point_bisector(first: Point, second: Point) -> Plane:
    """The perpendicular bisector plane of two unit-weight points folds one onto the other (`[...] Plane+`)."""
    return ((first & second) | (first + second)).normalized()                   # [...] Plane+


def line_bisector(first: Line, second: Line, intersection: Point) -> Plane:
    """The angle bisector plane of two oriented lines crossing at `intersection` folds one onto the other (`[...] Plane+`)."""
    first, second = first.normalized(), second.normalized()                     # [...] Line-, [...] Line-
    return ((first + second) | intersection).normalized()                       # [...] Plane+


def polygon_faces(counts: np.ndarray) -> tuple[np.ndarray, ...]:
    """Contiguous corner counts partition the flat corner sequence into one index array per face."""
    return tuple(np.split(np.arange(int(counts.sum())), np.cumsum(counts)[:-1]))


def split_polygons(points: Point, counts: np.ndarray, offsets: Direction, crease: Plane,
                   selected: tuple[int, ...], tolerance: float) -> tuple[Point, np.ndarray, Direction, np.ndarray]:
    """Cutting `selected` faces along `crease` inserts edge intersections and duplicates crease corners into moving then stationary polygons."""
    faces = polygon_faces(counts)
    F = len(counts)
    loops = Topology.loops(counts)
    V, E = loops.n_elements
    # Each polygon's closed boundary loop pairs every corner with the next corner around its face.
    tails, heads = loops.incidence[1, 0] * points                               # Point-[E], Point-[E]
    distances = crease & stack([tails, heads])                                  # [2] Scalar+[E]
    negative, positive = distances < -tolerance, distances > tolerance
    crossing = (negative[0] & positive[1]) | (positive[0] & negative[1])

    # Only crossing edges divide by their intersection weight, so edges parallel to the crease stay finite.
    cuts = (tails & heads) ^ crease                                             # Point-[E]
    unit_weights = mv.scalar(np.ones((V, 1))).field()                           # Scalar+[V]
    cut_points = where(crossing, cuts / where(crossing, mv.w & cuts, unit_weights), points)  # Point-[V]
    corners_and_cuts = stack([points, cut_points])                              # [2] Point-[V]
    vertex_mask = np.stack([np.ones_like(crossing), crossing], axis=-1)
    pair_distances = crease & corners_and_cuts                                  # [2] Scalar+[V]
    pair_negative, pair_positive = (pair_distances < -tolerance).T, (pair_distances > tolerance).T
    corners_only = np.array([True, False])

    # Duplicating crease vertices lets moving and stationary pieces turn independently about their shared edge.
    piece_masks, turning, parents = [], [], []
    for index, face in enumerate(faces):
        if index in selected:
            piece_masks.append(vertex_mask[face] & ~pair_positive[face])
            turning.append(True)
            parents.append(index)
            piece_masks.append(vertex_mask[face] & ~pair_negative[face])
            turning.append(False)
            parents.append(index)
        else:
            piece_masks.append(vertex_mask[face] & corners_only)
            turning.append(False)
            parents.append(index)

    parent_indices = np.array(parents)
    corner, at_cut = np.nonzero(np.concatenate(piece_masks))
    corner_indices = np.concatenate([faces[parent] for parent in parent_indices])[corner]
    split_counts = np.array([mask.sum() for mask in piece_masks])
    picked_pairs = Sparse.selection(corner_indices, V) * corners_and_cuts       # [2] Point-[V_split]
    split_points = where(at_cut == 1, picked_pairs[1], picked_pairs[0])         # Point-[V_split]
    split_offsets = Sparse.selection(parent_indices, F) * offsets               # Direction-[F_split]
    return split_points, split_counts, split_offsets, np.array(turning)


def folded_state(points: Point, counts: np.ndarray, offsets: Direction, sheet: Plane,
                 hinge: Hinge, turning: np.ndarray, angle: float) -> PaperState:
    """Turning the `turning` faces by `angle` about `hinge` transports their corners and carries their layer offsets past the stationary stack."""
    hinge = hinge.normalized()                                                  # [] Hinge+
    F = len(counts)
    face_motors = (hinge * (angle / 2 * turning)).exp().field()                 # Motor+[F]
    corner_motors = Sparse.selection(np.repeat(np.arange(F), counts), F) * face_motors  # Motor+[V]
    turned_points = corner_motors >> points                                     # Point-[V]

    # Signed layer heights along the sheet normal place the offset pivot halfway above the two stacks' top layers.
    normal_direction = (sheet | origin) ^ mv.w                                  # [] Direction-
    normal = normal_direction * (sheet & normal_direction)                      # [] Direction-
    heights = (sheet & offsets) / (sheet & normal) * np.sign(angle)             # Scalar+[F]
    on_turning = Sparse.selection(np.flatnonzero(turning), F)                   # Scalar+[F_turning <- F]
    on_stationary = Sparse.selection(np.flatnonzero(~turning), F)               # Scalar+[F_stationary <- F]
    pivot_height = ((on_turning * heights).max() + (on_stationary * heights).max() + 1) / 2  # [] Scalar+
    pivot = normal * (pivot_height * np.sign(angle))                            # [] Direction-
    turned_offsets = (face_motors >> (offsets - pivot)) + pivot                 # Direction-[F]
    return turned_points, counts, turned_offsets, sheet


def folding_states(initial_state: PaperState, creases: Plane, selections: tuple[tuple[int, ...], ...],
                   peaks: np.ndarray, ends: np.ndarray, n_half_steps: int, tolerance: float) -> Iterator[PaperState]:
    """Yields the paper state `(points, counts, offsets, sheet)` across every eased half-stage of the folding sequence."""
    points, counts, offsets, sheet = initial_state
    yield points, counts, offsets, sheet
    fractions = np.linspace(0, 1, n_half_steps + 1)[1:]
    fractions = fractions * fractions * (3 - 2 * fractions)
    for crease, selected, peak, end in zip(creases, selections, peaks, ends):
        split_points, split_counts, split_offsets, turning = split_polygons(
            points, counts, offsets, crease, selected, tolerance,
        )
        hinge = crease ^ sheet                                                  # [] Hinge+
        for angle in np.concatenate([peak * fractions, peak + (end - peak) * fractions]):
            points, counts, offsets, sheet = folded_state(
                split_points, split_counts, split_offsets, sheet, hinge, turning, angle,
            )
            yield points, counts, offsets, sheet
