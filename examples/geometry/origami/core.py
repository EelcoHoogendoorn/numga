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
    return ((first & second) | (first + second)).normalized()


def line_bisector(first: Line, second: Line, sheet: Plane) -> Plane:
    """The external angle bisector of two intersecting oriented lines in a sheet."""
    first, second = first.normalized(), second.normalized()
    intersection = (sheet | first) ^ second
    return ((first + second) | intersection).normalized()


@dataclass(frozen=True)
class Paper:
    """Polygon corners in one point batch; counts delimit the separate paper faces."""

    points: Point
    counts: np.ndarray
    offsets: Direction
    sheet: Plane

    def faces(self) -> tuple[np.ndarray, ...]:
        return tuple(np.split(np.arange(len(self.points)), np.cumsum(self.counts)[:-1]))

    @staticmethod
    def sides(plane: Plane, points: Point, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
        """Which points lie beyond the tolerance on the negative and on the positive side of a unit plane."""
        distance = plane & points
        return distance < -tolerance, distance > tolerance

    def split(self, plane: Plane, selected: tuple[int, ...],
              tolerance: float) -> tuple[Paper, np.ndarray, np.ndarray]:
        """Cut selected faces, retaining moving then stationary pieces in face order.

        Returns the cut paper, which of its faces turn, and the face of this paper each came from.
        The tolerance is a distance to the unit crease plane, used to keep vertices
        already on a crease there despite round-off from preceding half-turns.
        """
        faces = self.faces()
        following = np.concatenate([np.roll(face, -1) for face in faces])
        negative, positive = Paper.sides(plane, self.points, tolerance)
        crossing = (negative & positive[following]) | (positive & negative[following])

        # Pair each corner with the intersection of its outgoing edge and the crease plane.
        # Only crossing edges are evaluated, so parallel edges never require division.
        cuts = (self.points[crossing] & self.points[following[crossing]]) ^ plane
        cuts = cuts / (mv.w & cuts)
        slots = stack([self.points, self.points.at[crossing].set(cuts)], axis=-1)  # [corners, 2] Point
        inserted = np.stack([np.ones_like(crossing), crossing], axis=-1)
        negative, positive = Paper.sides(plane, slots, tolerance)
        corners_only = np.array([True, False])

        # Face indices encode the folding recipe. Splitting duplicates crease vertices:
        # the two faces can then move independently while their common edge stays fixed.
        def pieces() -> Iterator[tuple[np.ndarray, bool, int]]:
            for index, face in enumerate(faces):
                if index in selected:
                    yield inserted[face] & ~positive[face], True, index
                    yield inserted[face] & ~negative[face], False, index
                else:
                    yield inserted[face] & corners_only, False, index

        kept, turning, parents = zip(*pieces())
        parents = np.array(parents)
        # Kept slots in boundary order, piece after piece.
        corner, slot = np.nonzero(np.concatenate(kept))
        corners = np.concatenate([faces[parent] for parent in parents])
        counts = np.array([piece.sum() for piece in kept])
        paper = Paper(slots[corners[corner], slot], counts, self.offsets[parents], self.sheet)
        return paper, np.array(turning), parents

    def folded(self, hinge: Line, turning: np.ndarray, angle: float) -> Paper:
        """Turn the turning faces as one batch about their common crease."""
        hinge = hinge.normalized()
        # Faces that stay put turn by the zero angle, the identity motor.
        points = (hinge * (angle / 2 * np.repeat(turning, self.counts))).exp() >> self.points
        return Paper(points, self.counts, self.folded_offsets(hinge, turning, angle), self.sheet)

    def folded_offsets(self, hinge: Line, turning: np.ndarray, angle: float) -> Direction:
        """Carry infinitesimal layer separations rigidly through a fold.

        Offsets are measured in paper layers. The renderer sets their visible thickness;
        the crease geometry itself stays exactly on the mathematical sheet.
        """
        heights = (self.sheet & self.offsets) * np.sign(angle)
        moving_heights, fixed_heights = heights[turning], heights[~turning]
        # Offset the display hinge so a closed stack lands one layer beyond the fixed
        # stack. The same rotation transports every separation continuously, reversing
        # their order only as the paper turns over. Negative folds land underneath.
        height = (moving_heights[moving_heights.argmax()] + fixed_heights[fixed_heights.argmax()] + 1) / 2
        # The sheet times the pseudoscalar is its normal direction.
        pivot = self.sheet * mv.xyzw * (height * np.sign(angle))
        motors = (hinge * (angle / 2 * turning)).exp()  # [faces] Motor
        return (motors >> (self.offsets - pivot)) + pivot


def folding(paper: Paper, planes: Plane, selections: tuple[tuple[int, ...], ...],
            peaks: np.ndarray, ends: np.ndarray, half_steps: int, tolerance: float) -> Iterator[Paper]:
    """Execute crease-and-fold stages, including folds that reopen to mark a crease."""
    yield paper
    fractions = np.linspace(0, 1, half_steps + 1)[1:]
    fractions = fractions * fractions * (3 - 2 * fractions)
    for plane, selected, peak, end in zip(planes, selections, peaks, ends):
        split, turning, _ = paper.split(plane, selected, tolerance)
        hinge = plane ^ paper.sheet
        for angle in np.concatenate([peak * fractions, peak + (end - peak) * fractions]):
            paper = split.folded(hinge, turning, angle)
            yield paper
