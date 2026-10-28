"""The scenes of the surface normals example: a vertical mirror reflects a horizontal triangle, a line lying in the
mirror, and two planes crossing in that line."""

from __future__ import annotations

from numga import stack

from examples.orientation.normals.core import Axis, Plane, Point, Spear, Surface, mv


# --- math -----------------------------------------------------------------------------
def normals() -> tuple[Point, Plane, Surface, Plane]:
    """A horizontal triangle, its plane's positive side and its winding, each beside its image in a vertical mirror,
    and the mirror."""
    triangle = mv.point([[-2.4, -0.65, 0], [-0.9, -0.45, 0], [-1.8, 0.8, 0]])  # [vertices] Point-
    side = mv.z                                                                 # [] Plane+
    mirror = mv.x                                                               # [] Plane+

    # Joining the vertices in order gives the plane its winding orientation.
    winding = (triangle[0] & triangle[1] & triangle[2]).normalized()            # [] Surface-

    # The same reflection acts on both planes, respecting their different orientations.
    vertices = stack([triangle, mirror >> triangle])                            # [scenes, vertices] Point-
    sides = stack([side, mirror >> side])                                       # [scenes] Plane+
    windings = stack([winding, mirror >> winding])                              # [scenes] Surface-
    return vertices, sides, windings, mirror


def line_in_mirror() -> tuple[Point, Spear, Plane]:
    """Two points lying in a vertical mirror; the line through them, its mirror image and its plain sandwich with the
    mirror; and the mirror."""
    ends = mv.point([[0, -1, 0.2], [0, 1, 0.8]])                                # [ends] Point-
    mirror = mv.x                                                               # [] Plane+
    line = ends[0] & ends[1]                                                    # [] Spear-
    # The plain sandwich gives every element of a grade the same sign, whatever its orientation.
    lines = stack([line, mirror >> line, mirror * line * mirror.inverse()])     # [cases] Spear-
    return ends, lines, mirror


def pairs_in_mirror() -> tuple[Plane, Point, Axis, Spear, Plane]:
    """Two planes at right angles to each other and at half a right angle to a vertical mirror, crossing in a line
    that lies in the mirror, and two points on that line: each pair beside its mirror image, with the meet of the
    planes and the join of the points built from each; and the mirror."""
    mirror = mv.x                                                               # [] Plane+
    crossing = stack([mv.x + mv.y, mv.y - mv.x])                                # [planes] Plane+
    on_line = mv.point([[0, 0, -0.5], [0, 0, 0.5]])                             # [ends] Point-

    crossings = stack([crossing, mirror >> crossing])                           # [scenes, planes] Plane+
    both_ends = stack([on_line, mirror >> on_line])                             # [scenes, ends] Point-
    meets = crossings[:, 0] ^ crossings[:, 1]                                   # [scenes] Axis+
    joins = both_ends[:, 0] & both_ends[:, 1]                                   # [scenes] Spear-
    return crossings, both_ends, meets, joins, mirror


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.orientation.normals import render

    save_figure(render.draw_normals(*normals()), "normal_orientation")
    save_figure(render.draw_line_in_mirror(*line_in_mirror()), "line_in_mirror")
    save_figure(render.draw_pairs_in_mirror(*pairs_in_mirror()), "pairs_in_mirror")


if __name__ == "__main__":
    main()
