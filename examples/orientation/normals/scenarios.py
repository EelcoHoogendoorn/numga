"""A vertical mirror reflecting a horizontal triangle, a line lying in the mirror, and two planes crossing in it."""

from __future__ import annotations

from numga import stack

from examples.orientation.normals import core
from examples.orientation.normals.core import Axis, Plane, Point, Spear, Surface, mv


def normals() -> tuple[Point, Plane, Surface, Plane]:
    """A plane's positive side and a triangle's winding under the same reflection."""
    triangle: Point = mv.antivector([
        [-2.4, -0.65, 0, 1],
        [-0.9, -0.45, 0, 1],
        [-1.8, 0.8, 0, 1],
    ])
    side: Plane = mv.z
    mirror: Plane = mv.x
    return *core.reflect_normals(triangle, side, mirror), mirror


def line_in_mirror() -> tuple[Point, Spear, Plane]:
    """Two points lying in a vertical mirror, and the line through them: as it is, and reflected two ways."""
    ends: Point = mv.point([[0.0, -1.0, 0.2], [0.0, 1.0, 0.8]])
    mirror: Plane = mv.x
    return ends, core.reflect_line(ends, mirror), mirror


def pairs_in_mirror() -> tuple[Plane, Point, Axis, Spear, Plane]:
    """Two planes at right angles to each other and at half a right angle to a vertical mirror, crossing in
    a line that lies in the mirror, and two points on that line."""
    mirror: Plane = mv.x
    planes: Plane = stack([mv.x + mv.y, mv.y - mv.x])
    ends: Point = mv.point([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    return *core.reflect_crossing(planes, ends, mirror), mirror


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.orientation.normals import render

    save_figure(render.draw_normals(*normals()), "normal_orientation")
    save_figure(render.draw_line_in_mirror(*line_in_mirror()), "line_in_mirror")
    save_figure(render.draw_pairs_in_mirror(*pairs_in_mirror()), "pairs_in_mirror")
