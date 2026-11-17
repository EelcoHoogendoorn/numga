"""Surface normals distinguish a chosen side of a surface from the winding of its ordered vertices.

Projective geometric algebra represents both by planes of grade 1 with opposite orientations. A
boundary plane carries extrinsic orientation, whereas joining three ordered points gives a plane
with intrinsic orientation. Joining two points in a mirror gives a line with intrinsic orientation
that reflection with `>>` preserves, whereas the plain sandwich `mirror * line * mirror.inverse()`
reverses it. Intersecting two planes symmetric across the mirror gives the same geometric line with
extrinsic orientation, and reflection swaps the two planes so their outer product changes sign.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                                                  # Antivector-
Plane = ga.gatype.vector()                                                      # Vector+
Surface = ga.gatype(Point & Point & Point)                                      # Antitrivector-
# Joining two points orients a line along itself (Study's spear, as in Gunn's projective duality),
# whereas intersecting two planes orients a line around itself (an axis).
Spear = ga.gatype(Point & Point)                                                # Antibivector-
Axis = ga.gatype(Plane ^ Plane)                                                 # Bivector+
