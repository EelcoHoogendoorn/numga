"""Three marker correspondences determine the rigid motor that carries an object onto its measured pose.

Matching one marker fixes the position, matching the line through two markers fixes the direction, and matching the plane through all three fixes the roll about that line. In projective geometric algebra, each match is the square root of the ratio of two elements of the same kind, `(target / source).square_root()`, applied to a point, a line, and a plane, and the pose is the product of the three. The construction follows the motor reconstruction example of ganja.js by Steven De Keninck (https://github.com/enkimute/ganja.js/blob/master/examples/example_chapter11_motor_reconstruction.html).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                      # Antivector-
Motor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def alignment_motors(source: Point, target: Point) -> Motor:
    """Three incremental motors (`[3, ...] Motor+`) carry `source` (`[..., 3] Point-`) onto `target` by matching one marker, one edge line, and the marker plane in turn."""
    # Two point reflections translate by twice the required displacement.
    translation = (target[..., 0] / source[..., 0]).square_root()                      # [...] Motor+
    at_first_marker = translation >> source                                            # [..., 3] Point-

    # Turn the joining line while keeping the matched first point fixed.
    target_line = (target[..., 0] & target[..., 1]).normalized()                       # [...] Line-
    current_line = (target[..., 0] & at_first_marker[..., 1]).normalized()             # [...] Line-
    turn = (target_line / current_line).square_root()                                  # [...] Motor+
    along_first_edge = turn >> at_first_marker                                         # [..., 3] Point-

    # Only rotation about that line remains; matching the marker planes removes it.
    target_plane = (target_line & target[..., 2]).normalized()                         # [...] Plane-
    current_plane = (target_line & along_first_edge[..., 2]).normalized()              # [...] Plane-
    roll = (target_plane / current_plane).square_root()                                # [...] Motor+
    return stack([translation, turn, roll])                                            # [3, ...] Motor+


def alignments(source: Point, increments: Motor, fractions: np.ndarray) -> Iterator[Point]:
    """Yields the marker triple (`[..., 3] Point-`) along each incremental motor's screw in turn."""
    current = source
    yield current
    for increment in increments:
        start = current
        for fraction in fractions:
            # A fraction of the motor's logarithm moves at a constant rate along its screw.
            step_motor = (increment.log() * fraction).exp()                            # [...] Motor+
            current = step_motor >> start                                              # [..., 3] Point-
            yield current
