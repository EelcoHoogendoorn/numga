"""Thirteen central planes sit perpendicular to the cube's rotation axes and permute under its generating rotations."""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Plane = ga.gatype.vector()                          # Vector+
Point = ga.gatype.antivector()                      # Antivector-

# Integer direction vectors along the three face, six edge, and four corner rotation axes of the cube.
face_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
edge_axes = [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]]
corner_axes = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]

# Unit central planes perpendicular to the thirteen rotation axes of the cube.
planes = mv.vector(face_axes + edge_axes + corner_axes).normalized()            # [13] Plane+

# A quarter turn in a face plane and a third turn around a body diagonal generate the cube's rotation group.
quarter_turn = (mv.xy * (np.pi / 4)).exp()                                      # [] Rotor+
third_turn = ((mv.xy + mv.yz + mv.zx).normalized() * (np.pi / 3)).exp()         # [] Rotor+
generators = stack([quarter_turn, third_turn])                                  # [2] Rotor+
turned_planes = generators[:, None] >> planes                                   # [2, 13] Plane+
alignments = (turned_planes[:, :, None] | planes).abs()                         # [2, 13, 13] Scalar+
