"""Plane sides and triangle winding under reflection, ray traced on the GPU: the mirror, the planes and the triangles
translucent plates with their rims, lines rods, points balls and orientations arrows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.orientation.normals.core import Axis, Plane, Point, Spear, Surface, ga
from examples.wgsl import show, solids

sm = solids.mv
SIDE = np.array(to_rgb("#218c9d"))
WINDING = np.array(to_rgb("#ce713f"))
FACE = np.array(to_rgb("#b8c1c8"))
FACE_EDGE = np.array(to_rgb("#59626a"))
MIRROR = np.array(to_rgb("#8095aa"))
LINE = np.array(to_rgb("#3b4a57"))
FIRST = np.array(to_rgb("#c8553d"))
SECOND = np.array(to_rgb("#2f6690"))
background = np.ones(3)
# The margin around each scene as a share of the larger side, and the pixels across a panel, fewer for the tall
# panels of the crossing planes.
MARGIN, N_PIXELS, N_PAIR_PIXELS = 0.25, 460, 340
# How much of what lies behind them each surface of the mirror, the triangles and the crossing planes hides, and how
# strongly the mirror's rim shows over the white page.
MIRROR_OPACITY, FACE_OPACITY, PLANE_OPACITY, MIRROR_RIM_OPACITY = 0.06, 0.55, 0.23, 0.6
MIRROR_RIM = MIRROR_RIM_OPACITY * MIRROR + (1 - MIRROR_RIM_OPACITY)
# The widths of rims, lines, normal arrows and edge arrows, the arrowheads' widths, and the points' radius, in pixels.
RIM_WIDTH, LINE_WIDTH, NORMAL_WIDTH, EDGE_ARROW_WIDTH, HEAD_WIDTH = 3.0, 3.5, 5.0, 4.0, 14.0
DOT_RADIUS = 7.0


# --- plumbing -------------------------------------------------------------------------
def arrows(starts: np.ndarray, ends: np.ndarray, width: float, head_share: float, view: solids.View
           ) -> tuple[solids.Quadric, solids.Quadric]:
    """Arrows between places `[arrows, 3]`, their shafts `width` pixels across and their heads a share of each
    arrow's length."""
    lengths = np.linalg.norm(ends - starts, axis=-1)
    return solids.arrows(sm.point(starts), sm.point(ends), width / 2 * view.pixel, head_share * lengths,
                         HEAD_WIDTH / 2 * view.pixel)


def polygons(corners: np.ndarray, view: solids.View) -> tuple[solids.Quadric, solids.Quadric, solids.Quadric]:
    """Flat polygons of corners `[polygons, corners, 3]`: plates a pixel thick `[polygons, corners + 2]`, and their
    rims, rods `[polygons * corners, 3]` and the balls joining them `[polygons * corners * 2, 1]`."""
    places = sm.point(corners)                                                 # [polygons, corners] Point
    following = places[:, np.roll(np.arange(corners.shape[1]), -1)]
    rims, joints = solids.edges(places.reshape(-1), following.reshape(-1), RIM_WIDTH / 2 * view.pixel)
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
    values = vertices.cast(ga.subspace("wzy wxz wyx xyz")).kernel
    points = values[..., :3] / values[..., 3:]
    centres = points.mean(axis=1)
    side_normals = side.cast(ga.subspace("x y z w")).kernel[..., :3]
    winding_normals = winding.cast(ga.subspace("x y z w")).kernel[..., :3]
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
                       np.radians(22.0), MARGIN, N_PIXELS)
    triangles, rims, joints = polygons(points, view)
    mirror, mirror_rims, mirror_joints = polygons(corners[None], view)
    common = [(rims, FACE_EDGE), (joints, FACE_EDGE), (mirror_rims, MIRROR_RIM), (mirror_joints, MIRROR_RIM)]
    sheets = [(mirror, MIRROR, MIRROR_OPACITY), (triangles, FACE, FACE_OPACITY)]

    # Follow the supplied vertex order, so reflection visibly reverses the winding.
    edges = np.roll(points, -1, axis=1) - points
    edge_arrows = arrows((points + edges * 0.25).reshape(-1, 3), (points + edges * 0.75).reshape(-1, 3),
                         EDGE_ARROW_WIDTH, 0.4, view)
    side_arrows = arrows(centres, tips[0], NORMAL_WIDTH, 0.25, view)
    winding_arrows = arrows(centres, tips[1], NORMAL_WIDTH, 0.25, view)
    side_panel = [(side_arrows[0], SIDE), (side_arrows[1], SIDE)]
    winding_panel = [(winding_arrows[0], WINDING), (winding_arrows[1], WINDING), (edge_arrows[0], WINDING),
                     (edge_arrows[1], WINDING)]

    figure = grid_figure(view, 1, 2)
    panels(figure, [solids.draw(common + side_panel, view, background, translucent=sheets),
                    solids.draw(common + winding_panel, view, background, translucent=sheets)], view, 2)
    return figure


def draw_line_in_mirror(ends: Point, lines: Spear, mirror: Plane) -> plt.Figure:
    """Two points in the mirror and the line through them, one panel per line: as it is, and reflected two ways."""
    values = ends.cast(ga.subspace("wzy wxz wyx xyz")).kernel
    points = values[..., :3] / values[..., 3:]
    middle = points.mean(axis=0)
    directions = lines.cast(ga.subspace("yz zx xy")).kernel
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    span = np.linalg.norm(points[1] - points[0])
    corners = mirror_sheet(mirror, middle, span * 0.7)
    view = solids.view([sm.point(corners)], 0.0, np.radians(24.0), np.radians(62.0), MARGIN, N_PIXELS)
    mirror, mirror_rims, mirror_joints = polygons(corners[None], view)
    line = solids.rods(sm.point(points[0]), sm.point(points[1]), LINE_WIDTH / 2 * view.pixel)[None]
    dots = solids.ball(sm.point(points), DOT_RADIUS * view.pixel)[:, None]
    common = [(mirror_rims, MIRROR_RIM), (mirror_joints, MIRROR_RIM), (line, LINE), (dots, LINE)]

    images = []
    for direction in directions:
        shaft, head = arrows(middle[None] - direction * span * 0.2, middle[None] + direction * span * 0.2,
                             NORMAL_WIDTH, 0.3, view)
        images.append(solids.draw(common + [(shaft, WINDING), (head, WINDING)], view, background,
                                  translucent=[(mirror, MIRROR, MIRROR_OPACITY)]))
    figure = grid_figure(view, 1, len(directions))
    panels(figure, images, view, len(directions))
    return figure


def draw_pairs_in_mirror(planes: Plane, ends: Point, meets: Axis, joins: Spear, mirror: Plane) -> plt.Figure:
    """Two points on a line in the mirror, and two planes crossing in that line, before and after the
    mirror: points in the top row, planes in the bottom row, each coloured by which one it is."""
    values = ends.cast(ga.subspace("wzy wxz wyx xyz")).kernel
    points = values[..., :3] / values[..., 3:]                              # [panels, 2, 3]
    normals = planes.cast(ga.subspace("x y z w")).kernel[..., :3]           # [panels, 2, 3]
    middle = points[0].mean(axis=0)
    along = points[0, 1] - points[0, 0]
    span = np.linalg.norm(along)
    along /= span
    corners = mirror_sheet(mirror, middle, span * 0.95)
    colours = np.stack([FIRST, SECOND])

    # Each plane's patch: a square about the middle, along the line and across it within the plane.
    across = np.cross(normals, along)
    across /= np.linalg.norm(across, axis=-1, keepdims=True)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    patches = middle + span * 0.6 * (square[:, :1] * across[..., None, :] + square[:, 1:] * along)   # [panels, 2, 4, 3]

    view = solids.view([sm.point(corners), sm.point(patches)], 0.0, np.radians(22.0), np.radians(28.0), MARGIN,
                       N_PAIR_PIXELS)
    mirror, mirror_rims, mirror_joints = polygons(corners[None], view)
    mirror_plate = (mirror, MIRROR, MIRROR_OPACITY)
    top, bottom = [], []
    for panel in range(len(points)):
        line = solids.rods(sm.point(points[panel, 0]), sm.point(points[panel, 1]), LINE_WIDTH / 2 * view.pixel)[None]
        marks = [(mirror_rims, MIRROR_RIM), (mirror_joints, MIRROR_RIM), (line, LINE)]
        dots = solids.ball(sm.point(points[panel]), DOT_RADIUS * view.pixel)[:, None]
        top.append(solids.draw(marks + [(dots, colours)], view, background, translucent=[mirror_plate]))
        # Each plane's patch and its rim in its own colour, rim by rim and joint by joint.
        sheets, rims, joints = polygons(patches[panel], view)
        n_corners = patches.shape[2]
        crossing = [(rims, np.repeat(colours, n_corners, axis=0)), (joints, np.repeat(colours, 2 * n_corners, axis=0))]
        bottom.append(solids.draw(marks + crossing, view, background,
                                  translucent=[mirror_plate, (sheets, colours, PLANE_OPACITY)]))
    figure = grid_figure(view, 2, len(points))
    panels(figure, top + bottom, view, len(points))
    return figure


def mirror_sheet(mirror: Plane, centre: np.ndarray, half_width: float) -> np.ndarray:
    """The corners `[4, 3]` of a square piece of the mirror plane, around the foot of `centre`."""
    values = mirror.cast(ga.subspace("x y z w")).kernel
    length = np.linalg.norm(values[:3])
    normal = values[:3] / length
    foot = centre - normal * ((centre @ values[:3] + values[3]) / length)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    across = np.cross(normal, reference)
    across /= np.linalg.norm(across)
    up = np.cross(normal, across)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return foot + half_width * (corners[:, :1] * across + corners[:, 1:] * up)
