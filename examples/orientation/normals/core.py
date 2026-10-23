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

from numga import NumpyContext, stack
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Surface = Plane.derive.flipped_orientation
Spear = (Point & Point).output_gatype
Axis = (Plane ^ Plane).output_gatype


# --- math -----------------------------------------------------------------------------
def reflect_normals(triangle: Point, side: Plane, mirror: Plane) -> tuple[Point, Plane, Surface]:
    """The triangle and its two plane orientations, before and after reflection."""
    # Joining the vertices in order gives the plane its winding orientation.
    winding = (triangle[0] & triangle[1] & triangle[2]).normalized()            # Surface

    # The same reflection acts on both planes, respecting their different orientations.
    vertices = stack([triangle, mirror >> triangle])                            # Point
    sides = stack([side, mirror >> side])                                       # Plane
    windings = stack([winding, mirror >> winding])                              # Surface
    return vertices, sides, windings


def reflect_line(ends: Point, mirror: Plane) -> Spear:
    """The line through two points, then reflected by `>>`, then by the plain sandwich."""
    line = ends[0] & ends[1]                                                    # Spear
    return stack([line, mirror >> line, mirror * line * mirror.inverse()])


def reflect_crossing(planes: Plane, ends: Point, mirror: Plane) -> tuple[Plane, Point, Axis, Spear]:
    """Two planes crossing in a line, and two points on it: each pair before and after the mirror,
    with the meet of the planes and the join of the points built again from each pair."""
    reflected_planes = mirror >> planes                                         # Plane
    reflected_ends = mirror >> ends                                             # Point
    meets = stack([planes[0] ^ planes[1], reflected_planes[0] ^ reflected_planes[1]])  # Axis
    joins = stack([ends[0] & ends[1], reflected_ends[0] & reflected_ends[1]])   # Spear
    return stack([planes, reflected_planes]), stack([ends, reflected_ends]), meets, joins
