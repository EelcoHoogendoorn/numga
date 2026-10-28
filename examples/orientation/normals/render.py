"""Drawing for the surface normals example, ray traced on the GPU. The mirror, the planes and the triangles are
translucent plates with their rims, lines are rods, points are balls, and orientations are arrows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.orientation.normals.core import Axis, Plane, Point, Spear, Surface
from examples.util.wgsl import show, solids

sm = solids.mv
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
def coordinates(points: Point) -> np.ndarray:
    """x, y and z of points `[..., 3]`: their pairings with the planes x, y and z, at unit weight."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & points.normalized()[..., None]).to_array()


def arrows(starts: np.ndarray, ends: np.ndarray, width: float, head_share: float, view: solids.View,
           thickness: float) -> tuple[solids.Quadric, solids.Quadric]:
    """Arrows between places `[arrows, 3]`, their shafts `thickness` times `width` pixels across, and their heads
    `thickness` times the width set above and a share of each arrow's length."""
    lengths = np.linalg.norm(ends - starts, axis=-1)
    return solids.arrows(sm.point(starts), sm.point(ends), thickness * width / 2 * view.pixel, head_share * lengths,
                         thickness * head_width / 2 * view.pixel)


def polygons(corners: np.ndarray, view: solids.View, thickness: float
             ) -> tuple[solids.Plane, solids.Quadric, solids.Quadric]:
    """Flat polygons of corners `[polygons, corners, 3]`: plates a pixel thick, bounded by planes `[polygons,
    corners + 2]`, and their rims, rods `[polygons * corners, 3]` and the balls joining them `[polygons * corners * 2,
    1]`, the rims' width a share `thickness` of the one set above."""
    places = sm.point(corners)                                                 # [polygons, corners] Point-
    following = places[:, np.roll(np.arange(corners.shape[1]), -1)]
    rims, joints = solids.edges(places.reshape(-1), following.reshape(-1), thickness * rim_width / 2 * view.pixel)
    return solids.plates(places, view.pixel), rims, joints


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
    points = coordinates(vertices)
    centres = points.mean(axis=1)
    # The points at infinity along x, y and z, whose pairings with a plane are its normal.
    directions = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))
    side_normals = (side[..., None] & directions).to_array()
    winding_normals = (winding[..., None] & directions).to_array()
    normals = np.stack([side_normals, winding_normals])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    lower = points.min(axis=(0, 1))
    upper = points.max(axis=(0, 1))
    centre = (lower + upper) / 2
    span = np.max(upper - lower)
    arrow_length = span * 0.22
    corners = mirror_sheet(mirror, centre, span * 0.32)
    tips = centres + normals * arrow_length                                   # [panels, triangles, 3]

    view = solids.view([sm.point(corners), sm.point(points), sm.point(tips)], 0.0, np.radians(38.0),
                       np.radians(22.0), margin, n_pixels)
    triangles, rims, joints = polygons(points, view, 1.0)
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, 1.0)
    common = [(rims, face_edge_colour), (joints, face_edge_colour), (mirror_rims, mirror_rim_colour),
              (mirror_joints, mirror_rim_colour)]
    sheets = [(mirror_plates, mirror_colour, mirror_opacity), (triangles, face_colour, face_opacity)]

    # Follow the supplied vertex order, so reflection visibly reverses the winding.
    edges = np.roll(points, -1, axis=1) - points
    edge_arrows = arrows((points + edges * 0.25).reshape(-1, 3), (points + edges * 0.75).reshape(-1, 3),
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
    points = coordinates(ends)
    middle = points.mean(axis=0)
    # A line's direction, from its first point to its second: its pairings with the lines where the planes x, y and
    # z meet the plane at infinity.
    directions = (stack((mv.wx, mv.wy, mv.wz)) & lines[..., None]).to_array()
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    span = np.linalg.norm(points[1] - points[0])
    corners = mirror_sheet(mirror, middle, span * 0.7)
    view = solids.view([sm.point(corners)], 0.0, np.radians(24.0), np.radians(62.0), margin, n_line_pixels)
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, line_thickness)
    line = solids.rods(sm.point(points[0]), sm.point(points[1]), line_thickness * line_width / 2 * view.pixel)[None]
    dots = solids.ball(sm.point(points), line_thickness * dot_radius * view.pixel)[:, None]
    common = [(mirror_rims, mirror_rim_colour), (mirror_joints, mirror_rim_colour), (line, line_colour),
              (dots, line_colour)]

    images = []
    for direction in directions:
        shaft, head = arrows(middle[None] - direction * span * 0.2, middle[None] + direction * span * 0.2,
                             normal_width, 0.3, view, line_thickness)
        images.append(solids.draw(common + [(shaft, winding_colour), (head, winding_colour)], view, background,
                                  inks=[], translucent=[(mirror_plates, mirror_colour, mirror_opacity)], meshes=[]))
    figure = grid_figure(view, 1, len(directions))
    panels(figure, images, view, len(directions))
    return figure


def draw_pairs_in_mirror(planes: Plane, ends: Point, meets: Axis, joins: Spear, mirror: Plane) -> plt.Figure:
    """Two points on a line in the mirror, and two planes crossing in that line, before and after the
    mirror: points in the top row, planes in the bottom row, each coloured by which one it is."""
    mv = planes.context.multivector
    points = coordinates(ends)                                              # [panels, 2, 3]
    # The points at infinity along x, y and z, whose pairings with a plane are its normal.
    directions = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))
    normals = (planes[..., None] & directions).to_array()                   # [panels, 2, 3]
    middle = points[0].mean(axis=0)
    line_direction = points[0, 1] - points[0, 0]
    span = np.linalg.norm(line_direction)
    line_direction /= span
    corners = mirror_sheet(mirror, middle, span * 0.95)
    colours = np.stack([first_colour, second_colour])

    # Each plane's patch: a square about the middle, along the line and across it within the plane.
    transverse = np.cross(normals, line_direction)
    transverse /= np.linalg.norm(transverse, axis=-1, keepdims=True)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    patches = middle + span * 0.6 * (square[:, :1] * transverse[..., None, :]
                                     + square[:, 1:] * line_direction)     # [panels, 2, 4, 3]

    view = solids.view([sm.point(corners), sm.point(patches)], 0.0, np.radians(22.0), np.radians(28.0), margin,
                       n_pair_pixels)
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, pair_thickness)
    mirror_plate = (mirror_plates, mirror_colour, mirror_opacity)
    top, bottom = [], []
    for panel in range(len(points)):
        line = solids.rods(sm.point(points[panel, 0]), sm.point(points[panel, 1]),
                           pair_thickness * line_width / 2 * view.pixel)[None]
        marks = [(mirror_rims, mirror_rim_colour), (mirror_joints, mirror_rim_colour), (line, line_colour)]
        dots = solids.ball(sm.point(points[panel]), pair_thickness * dot_radius * view.pixel)[:, None]
        top.append(solids.draw(marks + [(dots, colours)], view, background, inks=[], translucent=[mirror_plate],
                               meshes=[]))
        # Each plane's patch and its rim in its own colour, rim by rim and joint by joint.
        sheets, rims, joints = polygons(patches[panel], view, pair_thickness)
        n_corners = patches.shape[2]
        crossing = [(rims, np.repeat(colours, n_corners, axis=0)), (joints, np.repeat(colours, 2 * n_corners, axis=0))]
        bottom.append(solids.draw(marks + crossing, view, background,
                                  inks=[], translucent=[mirror_plate, (sheets, colours, plane_opacity)], meshes=[]))
    figure = grid_figure(view, 2, len(points))
    panels(figure, top + bottom, view, len(points))
    return figure


def mirror_sheet(mirror: Plane, centre: np.ndarray, half_width: float) -> np.ndarray:
    """The corners `[4, 3]` of a square piece of the mirror plane, around the foot of `centre`."""
    mv = mirror.context.multivector
    normal = (mirror & stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))).to_array()
    # The mirror's pairing with the centre is the centre's distance from it, times the length of the normal.
    offset = (mirror & mv.point(centre)).to_array()
    length = np.linalg.norm(normal)
    normal /= length
    foot = centre - normal * offset / length
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    width_direction = np.cross(normal, reference)
    width_direction /= np.linalg.norm(width_direction)
    height_direction = np.cross(normal, width_direction)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return foot + half_width * (corners[:, :1] * width_direction + corners[:, 1:] * height_direction)
