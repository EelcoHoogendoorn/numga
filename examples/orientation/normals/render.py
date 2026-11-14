"""Ray-traced GPU views draw the mirror, crossing planes, and triangles as translucent rimmed plates,
lines as rods, points as balls, and orientations as chevron arrows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.orientation.normals.core import Axis, Plane, Point, Spear, Surface
from examples.util.panels import show
from numga import stack
from numga.render import simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology

side_colour = np.array(to_rgb("#218c9d"))
winding_colour = np.array(to_rgb("#ce713f"))
face_colour = np.array(to_rgb("#b8c1c8"))
face_edge_colour = np.array(to_rgb("#59626a"))
mirror_colour = np.array(to_rgb("#8095aa"))
line_colour = np.array(to_rgb("#3b4a57"))
first_colour = np.array(to_rgb("#c8553d"))
second_colour = np.array(to_rgb("#2f6690"))
background = np.ones(3)
# Scene margin as a share of the larger side, and panel widths in pixels for the three figures.
margin, n_pixels, n_line_pixels, n_pair_pixels = 0.25, 300, 200, 175
# Surface opacities and blended rim opacity for the mirror plate over a white background.
mirror_opacity, face_opacity, plane_opacity, mirror_rim_opacity = 0.06, 0.55, 0.23, 0.6
mirror_rim_colour = mirror_rim_opacity * mirror_colour + (1 - mirror_rim_opacity)
# Stroke widths and point radius in pixels of the triangle panels, and chevron length in stroke widths.
rim_width, line_width, normal_width, edge_arrow_width = 2.0, 2.3, 3.3, 2.6
dot_radius = 4.6
chevron_length = 4.0
# Relative stroke scale for the smaller panels of the line and crossing-plane figures.
line_thickness, pair_thickness = 0.67, 0.79


# --- plumbing -------------------------------------------------------------------------
def chevron_arrows(
    tails: Point,
    heads: Point,
    colour: np.ndarray,
    width: float,
    view: solids.View,
    thickness: float,
) -> Draw:
    """Tubes of width `thickness * width` pixels connect `tails` to `heads` (`[n_arrows] Point-`) and
    end in chevrons for `solids.draw`."""
    n_arrows = tails.shape[0]
    return simplices.tubes(
        stack((tails, heads), axis=-1).reshape(-1),
        Topology.chains([2] * n_arrows),
        colour,
        thickness * width,
        view,
        gloss=solids.gloss,
        chevrons=np.ones(n_arrows),
        chevron=chevron_length * thickness * width,
    )


def polygons(
    corners: Point, view: solids.View, thickness: float
) -> tuple[solids.Plane, solids.Quadric, solids.Quadric]:
    """Each polygon in `corners` (`[n_polygons, n_corners] Point-`) forms a pixel-thick plate together
    with rod and ball rims scaled by `thickness`."""
    n_polygons, n_corners = corners.shape
    outlines = Topology.loops([n_corners] * n_polygons)
    tails, heads = (outlines.incidence[1, 0] * corners.reshape(-1).field()).batch()  # [n_polygons * n_corners] Point-, Point-
    rims, joints = solids.edges(tails, heads, thickness * rim_width / 2 * view.pixel)
    return solids.plates(corners, view.pixel), rims, joints


def panel_grid_figure(images: list[np.ndarray], view: solids.View, n_columns: int) -> plt.Figure:
    """Equal-sized `images` of `view` tile a borderless figure in `n_columns` columns."""
    n_rows = -(-len(images) // n_columns)
    figure = plt.figure(
        figsize=(n_columns * view.shape[1] / 100, n_rows * view.shape[0] / 100), dpi=100
    )
    for index, image in enumerate(images):
        row, column = divmod(index, n_columns)
        axes = figure.add_axes(
            (column / n_columns, 1 - (row + 1) / n_rows, 1 / n_columns, 1 / n_rows)
        )
        show(axes, image, view)
    return figure


def normals_figure(
    vertices: Point, sides: Plane, windings: Surface, mirror: Plane
) -> plt.Figure:
    """Two side-by-side panels show the positive boundary side on the left and the vertex winding with
    edge arrows on the right across the mirror."""
    mv = sides.context.multivector
    n_scenes, n_vertices = vertices.shape
    origin = mv.w.dual()                                                        # [] Point-
    centres = vertices.mean(axis=-1)                                            # [n_scenes] Point-
    # Multiply the intrinsic winding by the ambient antiscalar at the display boundary to draw it as a 3D arrow.
    planes = stack([sides, windings * mv.antipseudoscalar([1])]).normalized()   # [2, n_scenes] Plane+
    # Shift each plane to the origin so its dual gives the unit normal direction.
    normals = (planes - mv.w * (planes & origin)).dual()                        # [2, n_scenes] Point-

    places = vertices.reshape(-1)                                               # [n_scenes * n_vertices] Point-
    centre = places.mean(axis=0)                                                # [] Point-
    distances = (places[:, None] & places[None, :]).norm().reshape(-1)
    span = distances.max()
    arrow_length = span * 0.22
    corners = mirror_sheet(mirror, centre, span * 0.32)                         # [4] Point-
    tips = centres + normals * arrow_length                                     # [2, n_scenes] Point-

    view = solids.view(
        [corners, vertices, tips], 0.0, np.radians(38.0), np.radians(22.0), margin, n_pixels
    )
    triangles, rims, joints = polygons(vertices, view, 1.0)
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, 1.0)
    common = [
        (rims, face_edge_colour),
        (joints, face_edge_colour),
        (mirror_rims, mirror_rim_colour),
        (mirror_joints, mirror_rim_colour),
    ]
    sheets = [(mirror_plates, mirror_colour, mirror_opacity), (triangles, face_colour, face_opacity)]

    # Edge arrows follow the ordered vertices so reflection visibly reverses the circulation.
    edge_offsets = (Topology.loops([n_vertices]).d[0] * vertices.field()).batch()  # [n_scenes, n_vertices] Point-
    edge_arrows = chevron_arrows(
        (vertices + edge_offsets * 0.25).reshape(-1),
        (vertices + edge_offsets * 0.75).reshape(-1),
        winding_colour,
        edge_arrow_width,
        view,
        1.0,
    )
    side_arrows = chevron_arrows(centres, tips[0], side_colour, normal_width, view, 1.0)
    winding_arrows = chevron_arrows(centres, tips[1], winding_colour, normal_width, view, 1.0)

    return panel_grid_figure(
        [
            solids.draw(common, view, background, inks=[], translucent=sheets, meshes=[(side_arrows, 1.0)]),
            solids.draw(
                common,
                view,
                background,
                inks=[],
                translucent=sheets,
                meshes=[(winding_arrows, 1.0), (edge_arrows, 1.0)],
            ),
        ],
        view,
        2,
    )


def line_in_mirror_figure(ends: Point, lines: Spear, mirror: Plane) -> plt.Figure:
    """Three side-by-side panels show a line between two points in the mirror, its reflection, and its
    plain mirror sandwich."""
    mv = lines.context.multivector
    n_cases = lines.shape[0]
    middle = ends.mean(axis=0)                                                  # [] Point-
    # Intersecting each unit-weight line with the ideal plane extracts its unit direction from first point to second.
    line_directions = lines.normalized() ^ -mv.w                                # [n_cases] Point-
    span = (ends[0] & ends[1]).norm().to_array()
    arrow_tails = middle - line_directions * (span * 0.2)                       # [n_cases] Point-
    arrow_heads = middle + line_directions * (span * 0.2)                       # [n_cases] Point-

    corners = mirror_sheet(mirror, middle, span * 0.7)                          # [4] Point-
    view = solids.view([corners], 0.0, np.radians(24.0), np.radians(62.0), margin, n_line_pixels)
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, line_thickness)
    line = solids.rods(ends[0], ends[1], line_thickness * line_width / 2 * view.pixel)[None]
    dots = solids.ball(ends, line_thickness * dot_radius * view.pixel)[:, None]
    common = [
        (mirror_rims, mirror_rim_colour),
        (mirror_joints, mirror_rim_colour),
        (line, line_colour),
        (dots, line_colour),
    ]

    images = []
    for case in range(n_cases):
        arrow = chevron_arrows(
            arrow_tails[case : case + 1],
            arrow_heads[case : case + 1],
            winding_colour,
            normal_width,
            view,
            line_thickness,
        )
        images.append(
            solids.draw(
                common,
                view,
                background,
                inks=[],
                translucent=[(mirror_plates, mirror_colour, mirror_opacity)],
                meshes=[(arrow, 1.0)],
            )
        )
    return panel_grid_figure(images, view, len(images))


def pairs_in_mirror_figure(
    planes: Plane, ends: Point, meets: Axis, joins: Spear, mirror: Plane
) -> plt.Figure:
    """Four panels in a 2x2 grid show two points on a mirror line in the top row and two planes
    crossing in that line in the bottom row, before and after reflection."""
    mv = planes.context.multivector
    n_scenes, n_planes = planes.shape
    origin = mv.w.dual()                                                        # [] Point-
    middle = ends[0].mean(axis=0)                                               # [] Point-
    span = joins[0].norm().to_array()
    line_direction = joins[0].normalized() ^ -mv.w                              # [] Point-
    # Within each plane, form the orthogonal plane through the intersection line to read the in-plane transverse direction.
    perpendicular_planes = (planes | meets[:, None]).normalized()               # [n_scenes, n_planes] Plane+
    plane_directions = (perpendicular_planes - mv.w * (perpendicular_planes & origin)).dual()  # [n_scenes, n_planes] Point-
    corners = mirror_sheet(mirror, middle, span * 0.95)                         # [4] Point-
    colours = np.stack([first_colour, second_colour])

    # Span a square patch on each plane along and across the intersection line.
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    reach = span * 0.6
    patches = (
        middle + plane_directions[..., None] * (reach * square[:, 0]) + line_direction * (reach * square[:, 1])
    )                                                                           # [n_scenes, n_planes, 4] Point-

    view = solids.view(
        [corners, patches], 0.0, np.radians(22.0), np.radians(28.0), margin, n_pair_pixels
    )
    mirror_plates, mirror_rims, mirror_joints = polygons(corners[None], view, pair_thickness)
    mirror_plate = (mirror_plates, mirror_colour, mirror_opacity)
    top, bottom = [], []
    for scene in range(n_scenes):
        line = solids.rods(ends[scene, 0], ends[scene, 1], pair_thickness * line_width / 2 * view.pixel)[None]
        marks = [(mirror_rims, mirror_rim_colour), (mirror_joints, mirror_rim_colour), (line, line_colour)]
        dots = solids.ball(ends[scene], pair_thickness * dot_radius * view.pixel)[:, None]
        top.append(
            solids.draw(
                marks + [(dots, colours)], view, background, inks=[], translucent=[mirror_plate], meshes=[]
            )
        )
        sheets, rims, joints = polygons(patches[scene], view, pair_thickness)
        n_corners = patches.shape[2]
        crossing = [
            (rims, np.repeat(colours, n_corners, axis=0)),
            (joints, np.repeat(colours, 2 * n_corners, axis=0)),
        ]
        bottom.append(
            solids.draw(
                marks + crossing,
                view,
                background,
                inks=[],
                translucent=[mirror_plate, (sheets, colours, plane_opacity)],
                meshes=[],
            )
        )
    return panel_grid_figure(top + bottom, view, n_scenes)


def mirror_sheet(mirror: Plane, centre: Point, half_width: float) -> Point:
    """Four corners (`[4] Point-`) bound a square patch of `mirror` of half-side `half_width` centred
    on the orthogonal foot of `centre`."""
    mv = mirror.context.multivector
    unit = mirror.normalized()                                                  # [] Plane+
    # Intersect the perpendicular line through the scene centre with the mirror to find its foot.
    foot = (unit | centre) ^ unit                                               # [] Point-
    # Shift the mirror to the origin and rotate the horizontal basis directions onto it.
    through_origin = unit - mv.w * (unit & mv.w.dual())                         # [] Plane+
    rotor = (1 + through_origin * mv.z).normalized()                            # [] Motor+
    width_direction = rotor >> mv.x.dual()                                      # [] Point-
    height_direction = rotor >> mv.y.dual()                                     # [] Point-
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return foot + width_direction * (half_width * corners[:, 0]) + height_direction * (half_width * corners[:, 1])
