"""Creases built from point and line coincidences split a sheet of paper into polygons and fold them rigidly into a kabuto helmet.

A crease that brings one point onto another is their perpendicular bisector plane `((first & second) | (first + second)).normalized()`, and a crease that brings one line onto another through their intersection is `((first + second) | intersection).normalized()`. Each crease meets the sheet in a hinge line `crease ^ sheet` that generates the fold's motor `(hinge * (angle / 2)).exp()`, and splits every crossed polygon edge at `(edge_start & edge_end) ^ crease`. The construction follows the origami example of ganja.js by Steven De Keninck, https://github.com/enkimute/ganja.js/blob/master/examples/example_pga3d_origami.html.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                      # Antivector-
# Directions, points at infinity: the layer offsets and the sheet's normal.
Direction = Point.derive.degenerate                 # Antivector-
Line = ga.gatype.bivector()                         # Bivector+
Plane = ga.gatype.vector()                          # Vector+
origin = mv.point([0, 0, 0])                        # [] Point-


# --- math -----------------------------------------------------------------------------
def point_bisector(first: Point, second: Point) -> Plane:
    """The perpendicular bisector plane of two equally weighted finite points folds one onto the other."""
    return ((first & second) | (first + second)).normalized()                   # [...] Plane+


def line_bisector(first: Line, second: Line, intersection: Point) -> Plane:
    """The external angle bisector plane of two oriented lines crossing at a point folds one onto the other."""
    first, second = first.normalized(), second.normalized()                     # [...] Line-, Line-
    return ((first + second) | intersection).normalized()                       # [...] Plane+


@dataclass(frozen=True)
class Paper:
    """A folded sheet is its faces' corners in one point batch, delimited by the corner counts, with each face's layer offset and the sheet's plane."""

    points: Point
    counts: np.ndarray
    offsets: Direction
    sheet: Plane

    def faces(self) -> tuple[np.ndarray, ...]:
        """The corner indices fall into one index array per face."""
        return tuple(np.split(np.arange(len(self.points)), np.cumsum(self.counts)[:-1]))

    def split(self, crease: Plane, selected: tuple[int, ...],
              tolerance: float) -> tuple[Paper, np.ndarray, np.ndarray]:
        """Cutting the selected faces along a crease plane gives the paper with moving then stationary pieces in
        face order, which of its faces turn, and the face of this paper each came from.
        The tolerance is a distance to the unit crease plane, used to keep vertices
        already on a crease there despite round-off from preceding half-turns.
        """
        faces = self.faces()
        next_corners = np.concatenate([np.roll(face, -1) for face in faces])
        distances = crease & self.points                                        # [corners] Scalar+
        negative, positive = distances < -tolerance, distances > tolerance
        crossing = (negative & positive[next_corners]) | (positive & negative[next_corners])

        # Pair each corner with the intersection of its outgoing edge and the crease plane.
        # Only crossing edges are evaluated, so parallel edges never require division.
        cuts = (self.points[crossing] & self.points[next_corners[crossing]]) ^ crease  # [crossings] Point-
        cuts = cuts / (mv.w & cuts)                                             # [crossings] Point-
        corners_and_cuts = stack([self.points, self.points.at[crossing].set(cuts)], axis=-1)  # [corners, pair] Point-
        vertex_mask = np.stack([np.ones_like(crossing), crossing], axis=-1)
        distances = crease & corners_and_cuts                                   # [corners, pair] Scalar+
        negative, positive = distances < -tolerance, distances > tolerance
        corners_only = np.array([True, False])

        # Face indices encode the folding recipe. Splitting duplicates crease vertices:
        # the two faces can then move independently while their common edge stays fixed.
        def pieces() -> Iterator[tuple[np.ndarray, bool, int]]:
            for index, face in enumerate(faces):
                if index in selected:
                    yield vertex_mask[face] & ~positive[face], True, index
                    yield vertex_mask[face] & ~negative[face], False, index
                else:
                    yield vertex_mask[face] & corners_only, False, index

        piece_masks, turning, parents = zip(*pieces())
        parents = np.array(parents)
        # The kept corners and cuts in boundary order, piece after piece.
        corner, at_cut = np.nonzero(np.concatenate(piece_masks))
        corners = np.concatenate([faces[parent] for parent in parents])
        counts = np.array([mask.sum() for mask in piece_masks])
        paper = Paper(corners_and_cuts[corners[corner], at_cut], counts, self.offsets[parents], self.sheet)
        return paper, np.array(turning), parents

    def folded(self, hinge: Line, turning: np.ndarray, angle: float) -> Paper:
        """The turning faces turn as one batch about their common hinge."""
        hinge = hinge.normalized()                                              # [] Line+
        # Faces that stay put turn by the zero angle, the identity motor.
        motors = (hinge * (angle / 2 * np.repeat(turning, self.counts))).exp()  # [corners] Motor+
        points = motors >> self.points                                          # [corners] Point-
        return Paper(points, self.counts, self.folded_offsets(hinge, turning, angle), self.sheet)

    def folded_offsets(self, hinge: Line, turning: np.ndarray, angle: float) -> Direction:
        """The layer offsets are infinitesimal separations carried rigidly through a fold.

        Offsets are measured in paper layers. The renderer sets their visible thickness;
        the crease geometry itself stays exactly on the mathematical sheet.
        """
        # The sheet's normal: the direction of the line perpendicular to it, signed towards its
        # positive side by its pairing with the sheet. An offset's height is its pairing with the
        # sheet, in units of the normal's.
        normal_direction = (self.sheet | origin) ^ mv.w                         # [] Direction-
        normal = normal_direction * (self.sheet & normal_direction)             # [] Direction-
        heights = (self.sheet & self.offsets) / (self.sheet & normal) * np.sign(angle)  # [faces] Scalar+
        turning_heights, stationary_heights = heights[turning], heights[~turning]
        # Offset the display hinge so a closed stack lands one layer beyond the stationary
        # stack. The same rotation transports every separation continuously, reversing
        # their order only as the paper turns over. Negative folds land underneath.
        height = (turning_heights[turning_heights.argmax()] + stationary_heights[stationary_heights.argmax()] + 1) / 2  # [] Scalar+
        pivot = normal * (height * np.sign(angle))                              # [] Direction-
        motors = (hinge * (angle / 2 * turning)).exp()                          # [faces] Motor+
        return (motors >> (self.offsets - pivot)) + pivot                       # [faces] Direction-


def folding(paper: Paper, creases: Plane, selections: tuple[tuple[int, ...], ...],
            peaks: np.ndarray, ends: np.ndarray, n_half_steps: int, tolerance: float) -> Iterator[Paper]:
    """The crease-and-fold stages yield the paper at every frame, including folds that reopen to mark a crease."""
    yield paper
    fractions = np.linspace(0, 1, n_half_steps + 1)[1:]
    fractions = fractions * fractions * (3 - 2 * fractions)
    for crease, selected, peak, end in zip(creases, selections, peaks, ends):
        split_paper, turning, _ = paper.split(crease, selected, tolerance)
        hinge = crease ^ paper.sheet                                            # [] Line+
        for angle in np.concatenate([peak * fractions, peak + (end - peak) * fractions]):
            paper = split_paper.folded(hinge, turning, angle)
            yield paper
