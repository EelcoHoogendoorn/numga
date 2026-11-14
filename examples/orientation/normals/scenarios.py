"""Three mirror scenes compare extrinsic and intrinsic orientations on a horizontal triangle, a line
in the mirror, and two planes crossing in that line."""

from __future__ import annotations

from examples.orientation.normals.core import Axis, Plane, Point, Spear, Surface, mv
from numga import stack


# --- math -----------------------------------------------------------------------------
def normals() -> tuple[Point, Plane, Surface, Plane]:
    """A vertical mirror reflects a horizontal triangle (`[2, 3] Point-`), its positive boundary side
    (`[2] Plane+`), and its vertex winding (`[2] Surface-`)."""
    triangle = mv.point([[-2.4, -0.65, 0], [-0.9, -0.45, 0], [-1.8, 0.8, 0]])   # [3] Point-
    side = mv.z                                                                 # [] Plane+
    mirror = mv.x                                                               # [] Plane+

    # Joining the ordered vertices gives the triangle's plane its intrinsic winding orientation.
    winding = (triangle[0] & triangle[1] & triangle[2]).normalized()            # [] Surface-

    # The same reflection acts on points, the boundary side, and the winding plane alike.
    vertices = stack([triangle, mirror >> triangle])                            # [2, 3] Point-
    sides = stack([side, mirror >> side])                                       # [2] Plane+
    windings = stack([winding, mirror >> winding])                              # [2] Surface-
    return vertices, sides, windings, mirror


def line_in_mirror() -> tuple[Point, Spear, Plane]:
    """Two points in a vertical mirror (`[2] Point-`) join into a line compared across the original,
    its `>>` reflection, and its plain mirror sandwich (`[3] Spear-`)."""
    ends = mv.point([[0, -1, 0.2], [0, 1, 0.8]])                                # [2] Point-
    mirror = mv.x                                                               # [] Plane+
    line = ends[0] & ends[1]                                                    # [] Spear-
    # The plain sandwich applies one grade sign regardless of orientation, reversing the fixed line.
    lines = stack([line, mirror >> line, mirror * line * mirror.inverse()])     # [3] Spear-
    return ends, lines, mirror


def pairs_in_mirror() -> tuple[Plane, Point, Axis, Spear, Plane]:
    """Two orthogonal planes (`[2, 2] Plane+`) and two points (`[2, 2] Point-`) meeting in a mirror
    line reflect across the mirror so their meets (`[2] Axis+`) negate while their joins (`[2] Spear-`)
    stay fixed."""
    mirror = mv.x                                                               # [] Plane+
    crossing_planes = stack([mv.x + mv.y, mv.y - mv.x])                         # [2] Plane+
    line_points = mv.point([[0, 0, -0.5], [0, 0, 0.5]])                         # [2] Point-

    plane_pairs = stack([crossing_planes, mirror >> crossing_planes])           # [2, 2] Plane+
    point_pairs = stack([line_points, mirror >> line_points])                   # [2, 2] Point-
    meets = plane_pairs[:, 0] ^ plane_pairs[:, 1]                               # [2] Axis+
    joins = point_pairs[:, 0] & point_pairs[:, 1]                               # [2] Spear-
    return plane_pairs, point_pairs, meets, joins, mirror


# --- plumbing -------------------------------------------------------------------------
def pinned() -> None:
    """Writes the tracked orientation figures for the triangle normals, the mirror line, and the
    crossing planes."""
    from examples.orientation.normals import render
    from examples.util.animation import pin_figure

    pin_figure(render.normals_figure(*normals()), "normal_orientation")
    pin_figure(render.line_in_mirror_figure(*line_in_mirror()), "line_in_mirror")
    pin_figure(render.pairs_in_mirror_figure(*pairs_in_mirror()), "pairs_in_mirror")


def main() -> None:
    """Saves the three orientation figures to the shared plots directory and verifies the reflection
    identities."""
    import matplotlib.pyplot as plt

    from examples.orientation.normals import render
    from examples.util.animation import save_figure

    vertices, sides, windings, mirror = normals()
    ends, lines, line_mirror = line_in_mirror()
    plane_pairs, point_pairs, meets, joins, pair_mirror = pairs_in_mirror()

    for figure, name in [
        (render.normals_figure(vertices, sides, windings, mirror), "normal_orientation"),
        (render.line_in_mirror_figure(ends, lines, line_mirror), "line_in_mirror"),
        (render.pairs_in_mirror_figure(plane_pairs, point_pairs, meets, joins, pair_mirror), "pairs_in_mirror"),
    ]:
        save_figure(figure, name)
        plt.close(figure)

    # checks
    witness = vertices[0].mean(axis=0) + mv.z.dual()                            # [] Point-
    reflected_witness = mirror >> witness                                       # [] Point-
    rebuilt_winding = (vertices[1, 0] & vertices[1, 1] & vertices[1, 2]).normalized()  # [] Surface-

    assert ((sides[0] & witness) - 1).abs() < 1e-6
    assert ((sides[1] & reflected_witness) - 1).abs() < 1e-6
    assert ((windings[0] & witness) + (windings[1] & reflected_witness)).scalar_norm_squared() < 1e-12
    rebuilt_diff = rebuilt_winding - windings[1]                                # [] Surface-
    assert rebuilt_diff.scalar_norm_squared() + rebuilt_diff.dual().scalar_norm_squared() < 1e-12

    reflected_line_diff = lines[1] - lines[0]                                   # [] Spear-
    sandwich_line_sum = lines[2] + lines[0]                                     # [] Spear-
    assert reflected_line_diff.scalar_norm_squared() + reflected_line_diff.dual().scalar_norm_squared() < 1e-12
    assert sandwich_line_sum.scalar_norm_squared() + sandwich_line_sum.dual().scalar_norm_squared() < 1e-12

    swapped_diff = plane_pairs[1] - plane_pairs[0, ::-1]                        # [2] Plane+
    fixed_ends_diff = point_pairs[1] - point_pairs[0]                           # [2] Point-
    meets_sum = meets[1] + meets[0]                                             # [] Axis+
    joins_diff = joins[1] - joins[0]                                            # [] Spear-
    assert ((swapped_diff.scalar_norm_squared() + swapped_diff.dual().scalar_norm_squared()) < 1e-12).all()
    assert ((fixed_ends_diff.scalar_norm_squared() + fixed_ends_diff.dual().scalar_norm_squared()) < 1e-12).all()
    assert meets_sum.scalar_norm_squared() + meets_sum.dual().scalar_norm_squared() < 1e-12
    assert joins_diff.scalar_norm_squared() + joins_diff.dual().scalar_norm_squared() < 1e-12


if __name__ == "__main__":
    main()
