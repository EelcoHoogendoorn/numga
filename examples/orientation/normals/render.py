"""Drawing for the surface normals example, ray traced on the GPU. The mirror, the planes and the triangles are
translucent plates with their rims, lines are rods, points are balls, and orientations are arrows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.orientation.normals.core import Axis, Plane, Point, Spear, Surface
from examples.util.wgsl import show, solids

side_colour = np.array(to_rgb("#218c9d"))
winding_colour = np.array(to_rgb("#ce713f"))
face_colour = np.array(to_rgb("#b8c1c8"))
face_edge_colour = np.array(to_rgb("#59626a"))
mirror_colour = np.array(to_rgb("#8095aa"))
line_colour = np.array(to_rgb("#3b4a57"))
first_colour = np.array(to_rgb("#c8553d"))
second_colour = np.array(to_rgb("#2f6690"))
background = np.ones(3)
# The margin around each scene as a share of the larger side, and the pixels across a panel: the triangles', the line
# in the mirror's, three side by side, and the crossing planes', whose panels are tall.
margin, n_pixels, n_line_pixels, n_pair_pixels = 0.25, 300, 200, 175
# How much of what lies behind them each surface of the mirror, the triangles and the crossing planes hides, and how
# strongly the mirror's rim shows over the white page.
mirror_opacity, face_opacity, plane_opacity, mirror_rim_opacity = 0.06, 0.55, 0.23, 0.6
mirror_rim_colour = mirror_rim_opacity * mirror_colour + (1 - mirror_rim_opacity)
# The widths of rims, lines, normal arrows and edge arrows, the arrowheads' widths, and the points' radius, in pixels
# of the triangles' panels.
rim_width, line_width, normal_width, edge_arrow_width, head_width = 2.0, 2.3, 3.3, 2.6, 9.1
dot_radius = 4.6
# The line in the mirror's widths and the crossing planes' as a share of those above, their scenes that much smaller.
line_thickness, pair_thickness = 0.67, 0.79


# --- plumbing -------------------------------------------------------------------------
def arrows(starts: Point, ends: Point, width: float, head_share: float, view: solids.View,
           thickness: float) -> tuple[solids.Quadric, solids.Quadric]:
    """Arrows between points of unit weight `[arrows]`, their shafts `thickness` times `width` pixels across, and their
    heads `thickness` times the width set above and a share of each arrow's length."""
    # The weight of the line through two points of unit weight is the distance between them.
    lengths = (starts & ends).norm().to_array()                               # [arrows]
    return solids.arrows(starts, ends, thickness * width / 2 * view.pixel, head_share * lengths,
                         thickness * head_width / 2 * view.pixel)


def polygons(corners: Point, view: solids.View, thickness: float
             ) -> tuple[solids.Plane, solids.Quadric, solids.Quadric]:
    """Flat polygons of corners `[polygons, corners]`: plates a pixel thick, bounded by planes `[polygons,
    corners + 2]`, and their rims, rods `[polygons * corners, 3]` and the balls joining them `[polygons * corners * 2,
    1]`, the rims' width a share `thickness` of the one set above."""
    following = corners[:, np.roll(np.arange(corners.shape[1]), -1)]          # [polygons, corners] Point-
    rims, joints = solids.edges(corners.reshape(-1), following.reshape(-1), thickness * rim_width / 2 * view.pixel)
    return solids.plates(corners, view.pixel), rims, joints


def panels(figure: plt.Figure, images: list[np.ndarray], view: solids.View, n_columns: int) -> None:
    """Images of one view in a grid of `n_columns`, filling the figure."""
    n_rows = -(-len(images) // n_columns)
    for index, image in enumerate(images):
        row, column = divmod(index, n_columns)
        show(figure.add_axes((column / n_columns, 1 - (row + 1) / n_rows, 1 / n_columns, 1 / n_rows)), image, view)


def grid_figure(view: solids.View, n_rows: int, n_columns: int) -> plt.Figure:
    return plt.figure(figsize=(n_columns * view.shape[1] / 100, n_rows * view.shape[0] / 100), dpi=100)


def draw_normals(vertices: Point, side: Plane, winding: Surface, mirror: Plane) -> plt.Figure:
    """Left, the plane's positive side over both triangles; right, the winding of both, its normal and the
    vertex order along their edges."""
    mv = side.context.multivector
    origin = mv.w.dual()                                                       # [] Point-
    centres = vertices.mean(axis=-1)                                           # [scenes] Point-
    # The winding as the plane it orients, extrinsic like the chosen side, so that one construction draws both normals.
    planes = stack([side, winding * mv.antipseudoscalar([1.0])]).normalized()  # [panels, scenes] Plane+
    # Each plane moved to the origin: its dual is the unit direction square to it.
    normals = (planes - mv.w * (planes & origin)).dual()                       # [panels, scenes] Point-

    # The scene's middle and its span: the centroid of the vertices, and the largest distance between two of them.
    places = vertices.reshape(-1)                                              # [scenes * vertices] Point-
    centre = places.mean(axis=0)                                               # [] Point-
    span = (places[:, None] & places[None, :]).norm().to_array().max()
    arrow_length = span * 0.22
    corners = mirror_sheet(mirror, centre, span * 0.32)                        # [corners] Point-
    tips = centres + normals * arrow_length                                    # [panels, scenes] Point-

    view = solids.view([corners, vertices, tips], 0.0, np.radians(38.0), np.radians(22.0), margin, n_pixels)
    triangles, rims, joints = polygons(vertices, view, 1.0)
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, 1.0)
    common = [(rims, face_edge_colour), (joints, face_edge_colour), (mirror_rims, mirror_rim_colour),
              (mirror_joints, mirror_rim_colour)]
    sheets = [(mirror_plates, mirror_colour, mirror_opacity), (triangles, face_colour, face_opacity)]

    # Follow the supplied vertex order, so reflection visibly reverses the winding.
    edges = vertices[:, np.roll(np.arange(vertices.shape[1]), -1)] - vertices  # [scenes, vertices] Point-
    edge_arrows = arrows((vertices + edges * 0.25).reshape(-1), (vertices + edges * 0.75).reshape(-1),
                         edge_arrow_width, 0.4, view, 1.0)
    side_arrows = arrows(centres, tips[0], normal_width, 0.25, view, 1.0)
    winding_arrows = arrows(centres, tips[1], normal_width, 0.25, view, 1.0)
    side_panel = [(side_arrows[0], side_colour), (side_arrows[1], side_colour)]
    winding_panel = [(winding_arrows[0], winding_colour), (winding_arrows[1], winding_colour),
                     (edge_arrows[0], winding_colour), (edge_arrows[1], winding_colour)]

    figure = grid_figure(view, 1, 2)
    panels(figure, [solids.draw(common + side_panel, view, background, inks=[], translucent=sheets, meshes=[]),
                    solids.draw(common + winding_panel, view, background, inks=[], translucent=sheets,
                                meshes=[])], view, 2)
    return figure


def draw_line_in_mirror(ends: Point, lines: Spear, mirror: Plane) -> plt.Figure:
    """Two points in the mirror and the line through them, one panel per line: as it is, and reflected two ways."""
    mv = lines.context.multivector
    middle = ends.mean(axis=0)                                                 # [] Point-
    # Each line's unit direction, from its first point to its second: where the line of unit weight meets the plane
    # at infinity, turned to face away from the points of unit weight.
    forward = lines.normalized() ^ -mv.w                                       # [cases] Point-
    span = (ends[0] & ends[1]).norm().to_array()
    arrow_starts = middle - forward * (span * 0.2)                             # [cases] Point-
    arrow_ends = middle + forward * (span * 0.2)                               # [cases] Point-

    corners = mirror_sheet(mirror, middle, span * 0.7)                         # [corners] Point-
    view = solids.view([corners], 0.0, np.radians(24.0), np.radians(62.0), margin, n_line_pixels)
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, line_thickness)
    line = solids.rods(ends[0], ends[1], line_thickness * line_width / 2 * view.pixel)[None]
    dots = solids.ball(ends, line_thickness * dot_radius * view.pixel)[:, None]
    common = [(mirror_rims, mirror_rim_colour), (mirror_joints, mirror_rim_colour), (line, line_colour),
              (dots, line_colour)]

    images = []
    for case in range(forward.shape[0]):
        shaft, head = arrows(arrow_starts[case:case + 1], arrow_ends[case:case + 1], normal_width, 0.3, view,
                             line_thickness)
        images.append(solids.draw(common + [(shaft, winding_colour), (head, winding_colour)], view, background,
                                  inks=[], translucent=[(mirror_plates, mirror_colour, mirror_opacity)], meshes=[]))
    figure = grid_figure(view, 1, len(images))
    panels(figure, images, view, len(images))
    return figure


def draw_pairs_in_mirror(planes: Plane, ends: Point, meets: Axis, joins: Spear, mirror: Plane) -> plt.Figure:
    """Two points on a line in the mirror, and two planes crossing in that line, before and after the
    mirror: points in the top row, planes in the bottom row, each coloured by which one it is."""
    mv = planes.context.multivector
    origin = mv.w.dual()                                                       # [] Point-
    middle = ends[0].mean(axis=0)                                              # [] Point-
    span = joins[0].norm().to_array()
    # The line's unit direction, from its first point to its second, as in `draw_line_in_mirror`.
    along = joins[0].normalized() ^ -mv.w                                      # [] Point-
    # Within each plane, the direction square to the line: the unit direction square to the plane through the line
    # that stands square to the plane.
    upright = (planes | meets[:, None]).normalized()                           # [scenes, planes] Plane+
    across = (upright - mv.w * (upright & origin)).dual()                      # [scenes, planes] Point-
    corners = mirror_sheet(mirror, middle, span * 0.95)                        # [corners] Point-
    colours = np.stack([first_colour, second_colour])

    # Each plane's patch: a square about the middle, along the line and across it within the plane.
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    reach = span * 0.6
    patches = middle + across[..., None] * (reach * square[:, 0]) + along * (reach * square[:, 1])   # [scenes, planes, corners] Point-

    view = solids.view([corners, patches], 0.0, np.radians(22.0), np.radians(28.0), margin, n_pair_pixels)
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, pair_thickness)
    mirror_plate = (mirror_plates, mirror_colour, mirror_opacity)
    top, bottom = [], []
    for scene in range(ends.shape[0]):
        line = solids.rods(ends[scene, 0], ends[scene, 1], pair_thickness * line_width / 2 * view.pixel)[None]
        marks = [(mirror_rims, mirror_rim_colour), (mirror_joints, mirror_rim_colour), (line, line_colour)]
        dots = solids.ball(ends[scene], pair_thickness * dot_radius * view.pixel)[:, None]
        top.append(solids.draw(marks + [(dots, colours)], view, background, inks=[], translucent=[mirror_plate],
                               meshes=[]))
        # Each plane's patch and its rim in its own colour, rim by rim and joint by joint.
        sheets, rims, joints = polygons(patches[scene], view, pair_thickness)
        n_corners = patches.shape[2]
        crossing = [(rims, np.repeat(colours, n_corners, axis=0)), (joints, np.repeat(colours, 2 * n_corners, axis=0))]
        bottom.append(solids.draw(marks + crossing, view, background,
                                  inks=[], translucent=[mirror_plate, (sheets, colours, plane_opacity)], meshes=[]))
    figure = grid_figure(view, 2, ends.shape[0])
    panels(figure, top + bottom, view, ends.shape[0])
    return figure


def mirror_sheet(mirror: Plane, centre: Point, half_width: float) -> Point:
    """The corners `[corners]` of a square piece of the mirror plane around the foot of `centre`, its sides along where
    the turn carrying the plane z onto the mirror carries x and y."""
    mv = mirror.context.multivector
    unit = mirror.normalized()                                                 # [] Plane+
    # The foot of the centre: where the line through it square to the mirror meets the mirror.
    foot = (unit | centre) ^ unit                                              # [] Point-
    # The mirror moved to the origin, and the turn carrying the plane z onto it.
    through_origin = unit - mv.w * (unit & mv.w.dual())                        # [] Plane+
    turn = (1 + through_origin * mv.z).normalized()                            # [] Motor+
    width_direction = turn >> mv.x.dual()                                      # [] Point-
    height_direction = turn >> mv.y.dual()                                     # [] Point-
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return foot + width_direction * (half_width * corners[:, 0]) + height_direction * (half_width * corners[:, 1])
