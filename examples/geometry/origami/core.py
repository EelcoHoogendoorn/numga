"""Crease construction, polygon splitting and rigid folds of a sheet of paper.

The construction follows the origami example of ganja.js by Steven De Keninck,
https://github.com/enkimute/ganja.js/blob/master/examples/example_pga3d_origami.html.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Direction = ga.gatype.from_blades("yzw zxw xyw")
Plane = ga.gatype.vector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Scalar = ga.gatype.scalar()


# --- math -----------------------------------------------------------------------------
def point_bisector(first: Point, second: Point) -> Plane:
    """The perpendicular bisector plane of two equally weighted finite points."""
    return ((first & second) | ((first + second) / 2)).normalized()


def line_bisector(first: Line, second: Line, normal: Point) -> Plane:
    """The external angle bisector of two intersecting oriented lines in a sheet."""
    first, second = first.normalized(), second.normalized()
    intersection = (first & normal) ^ second
    return ((first + second) | intersection).normalized()


@dataclass(frozen=True)
class Paper:
    """Polygon corners in one point batch; counts delimit the separate paper faces."""

    points: Point
    counts: np.ndarray
    offsets: Direction
    normal: Direction

    def faces(self) -> tuple[np.ndarray, ...]:
        return tuple(np.split(np.arange(len(self.points)), np.cumsum(self.counts)[:-1]))

    def split(self, plane: Plane, selected: tuple[int, ...], tolerance: float) -> tuple[Paper, np.ndarray]:
        """Cut selected faces, retaining moving then stationary pieces in face order.

        The tolerance is a distance to the unit crease plane, used to keep vertices
        already on a crease there despite round-off from preceding half-turns.
        """
        faces = self.faces()
        following = np.concatenate([np.roll(face, -1) for face in faces])
        distance = plane & self.points
        negative, positive = distance < -tolerance, distance > tolerance
        crossing = (negative & positive[following]) | (positive & negative[following])

        # Insert an intersection after each edge that crosses the crease plane.
        # Only crossing edges are evaluated, so parallel edges never require division.
        cuts = (self.points[crossing] & self.points[following[crossing]]) ^ plane
        cuts = cuts / (mv.w & cuts)
        expanded = stack([self.points, self.points], axis=-1).reshape(-1)
        expanded = expanded.at[2 * np.flatnonzero(crossing) + 1].set(cuts)
        inserted = np.column_stack([np.ones(len(self.points), dtype=bool), crossing]).reshape(-1)
        distances = plane & expanded
        negative, positive = distances < -tolerance, distances > tolerance

        # Face indices encode the folding recipe. Splitting duplicates crease vertices:
        # the two faces can then move independently while their common edge stays fixed.
        def pieces() -> Iterator[tuple[np.ndarray, bool, int]]:
            for index, face in enumerate(faces):
                if index in selected:
                    candidates = np.column_stack([2 * face, 2 * face + 1]).reshape(-1)
                    candidates = candidates[inserted[candidates]]
                    yield candidates[~positive[candidates]], True, index
                    yield candidates[~negative[candidates]], False, index
                else:
                    yield 2 * face, False, index

        polygons, turning, parents = zip(*pieces())
        counts = np.array([len(face) for face in polygons])
        points = expanded[np.concatenate(polygons)]
        return Paper(points, counts, self.offsets[np.array(parents)], self.normal), np.repeat(turning, counts)

    def folded(self, hinge: Line, moving: np.ndarray, angle: float) -> Paper:
        """Turn the selected polygon corners as one batch about their common crease."""
        motor = (hinge.normalized() * (angle / 2)).exp()
        points = self.points.at[moving].set(motor >> self.points[moving])
        return Paper(points, self.counts, self.folded_offsets(moving, motor, angle), self.normal)

    def folded_offsets(self, moving: np.ndarray, motor: Motor, angle: float) -> Direction:
        """Carry infinitesimal layer separations rigidly through a fold.

        Offsets are measured in paper layers. The renderer sets their visible thickness;
        the crease geometry itself stays exactly on the mathematical sheet.
        """
        face_index = np.repeat(np.arange(len(self.counts)), self.counts)
        turning = np.bincount(face_index, weights=moving, minlength=len(self.counts)) > 0
        heights = self.offsets.dual().scalar_product(self.normal.dual()) * np.sign(angle)
        moving_heights, fixed_heights = heights[turning], heights[~turning]
        # Offset the display hinge so a closed stack lands one layer beyond the fixed
        # stack. The same rotation transports every separation continuously, reversing
        # their order only as the paper turns over. Negative folds land underneath.
        height = (moving_heights[moving_heights.argmax()] + fixed_heights[fixed_heights.argmax()] + 1) / 2
        pivot = self.normal * (height * np.sign(angle))
        carried = (motor >> (self.offsets[turning] - pivot)) + pivot
        return self.offsets.at[turning].set(carried)


def folding(paper: Paper, planes: Plane, selections: tuple[tuple[int, ...], ...],
            peaks: np.ndarray, ends: np.ndarray, sheet: Plane, steps: int,
            tolerance: float) -> Iterator[Paper]:
    """Execute crease-and-fold stages, including folds that reopen to mark a crease."""
    yield paper
    half_steps = steps // 2
    fractions = np.linspace(0, 1, half_steps + 1)[1:]
    fractions = fractions * fractions * (3 - 2 * fractions)
    for plane, selected, peak, end in zip(planes, selections, peaks, ends):
        split, moving = paper.split(plane, selected, tolerance)
        hinge = plane ^ sheet
        angles = np.concatenate([peak * fractions, peak + (end - peak) * fractions])
        for angle in angles:
            yield split.folded(hinge, moving, angle)
        paper = split.folded(hinge, moving, end)
