r"""Three marker correspondences determine the rigid motor that carries an object onto its measured pose.

Matching one marker fixes the position, matching the line through two of them fixes the direction, and matching the plane through all three fixes the roll about that line. In projective geometric algebra each match is the square root of the ratio of two elements of the same kind, `(target / source).square_root()`, applied to a point, a line, and a plane, and the pose is the product of the three. The construction follows the motor reconstruction example of ganja.js by Steven De Keninck, https://github.com/enkimute/ganja.js/blob/master/examples/example_chapter11_motor_reconstruction.html.

In matrix notation each triple of markers $\mathbf a, \mathbf b, \mathbf c$ gives an orthonormal frame, $\mathbf t_1 = (\mathbf b - \mathbf a)/|\mathbf b - \mathbf a|$, $\mathbf t_3 = \mathbf t_1 \times (\mathbf c - \mathbf a)/|\mathbf t_1 \times (\mathbf c - \mathbf a)|$, and $\mathbf t_2 = \mathbf t_3 \times \mathbf t_1$, and the pose is the rotation $R = T' T^\mathsf{T}$ between the frames' matrices with the translation $\mathbf a' - R\,\mathbf a$.
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
def reconstruct(source: Point, target: Point) -> Motor:
    """The three incremental motors carrying the source marker triple onto the target triple match the first point, the first edge line, and the marker plane in turn."""
    # Two point reflections translate by twice the required displacement.
    translation = (target[..., 0] / source[..., 0]).square_root()                          # [...] Motor+

    # Turn the joining line while keeping the matched first point fixed.
    target_line = (target[..., 0] & target[..., 1]).normalized()                           # [...] Line-
    current_line = (target[..., 0] & (translation >> source[..., 1])).normalized()         # [...] Line-
    turn = (target_line / current_line).square_root()                                      # [...] Motor+

    # Only rotation about that line remains. Match the marker planes to remove it.
    target_plane = (target_line & target[..., 2]).normalized()                             # [...] Plane-
    current_plane = (target_line & ((turn * translation) >> source[..., 2])).normalized()  # [...] Plane-
    roll = (target_plane / current_plane).square_root()                                    # [...] Motor+
    return stack([translation, turn, roll])                                                # [stages, ...] Motor+


def placements(source: Point, increments: Motor) -> Iterator[Point]:
    """Yield the marker triple after each of the three incremental alignments."""
    for increment in increments:
        source = increment >> source
        yield source


def alignments(source: Point, increments: Motor, fractions: np.ndarray) -> Iterator[Point]:
    """Yield the marker triple along each incremental motor's screw in turn."""
    current = source
    yield current
    for increment in increments:
        start = current
        for fraction in fractions:
            # A fraction of the motor's logarithm moves at a constant rate along its screw.
            current = (increment.log() * fraction).exp() >> start
            yield current
