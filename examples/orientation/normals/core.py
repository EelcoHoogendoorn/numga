"""Surface normals distinguish a chosen side from the winding of ordered vertices.

Projective geometric algebra represents both by planes. A plane supplied as a boundary
carries extrinsic orientation; joining three points gives it intrinsic orientation. A line
through two points lying in a mirror is its own mirror image: reflecting it with `>>` leaves
its direction, while the plain sandwich `mirror * line * mirror.inverse()` reverses it. The
same line, built as the meet of two planes that are each other's mirror images, comes out of the
mirror negated: the reflected planes swap places, and the outer product of the swapped pair is
minus the original.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                                                  # Antivector-
Plane = ga.gatype.vector()                                                      # Vector+
Surface = Plane.derive.flipped_orientation                                      # Vector-
# A line oriented along itself, as the join of two points is, and one oriented around itself, as the meet of two
# planes is: a spear and an axis, in Dorst's names.
Spear = (Point & Point).output_gatype                                           # Antibivector-
Axis = (Plane ^ Plane).output_gatype                                            # Bivector+
