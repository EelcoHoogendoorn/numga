"""Drawing for the stressed cube: the cube seen from each frame with its face tractions, on the GPU, and Mohr's
circles.

The deformed cube is a translucent solid, its edges dark rods, and the undeformed cube a ghost of thin pale rods; the normal traction on each face is a blue arrow from the face's centre, the shear a red one, and the principal
directions are thin green rods through the centre."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from numga import stack
from matplotlib.colors import to_rgb

from examples.util import charts
from examples.util.animation import capture
from examples.mechanics.stress import core
from examples.util.wgsl import fitted, show, solids
from examples.util.wgsl.scene import Scene

# The cube's edges and faces, by the corners' indices: corner 4 i + 2 j + k sits at the i-th, j-th and
# k-th of -0.5 and 0.5 along x, y and z.
EDGES = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
FACES = np.array([(4, 5, 7, 6), (0, 2, 3, 1), (2, 6, 7, 3), (0, 1, 5, 4), (1, 3, 7, 5), (0, 4, 6, 2)])
# The arrows' lengths per unit traction, and the colours of the frames' diameters on Mohr's circle.
NORMAL_SCALE, SHEAR_SCALE = 0.4, 0.5
COLOURS = ("#0284c7", "#10b981", "#e11d48")
# The principal directions are drawn out to this distance from the centre, just past the cube; the room the view of
# a turning frame leaves around the first frame's cube and arrows, for them to move in.
REACH = 0.75
TURNING_ROOM = 0.05
# The colours of the cube, its edges, the undeformed cube, the tractions and the principal directions.
cube_colour, edge_colour, ghost_colour = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#0f172a", "#94a3b8"))
normal_colour, shear_colour, principal_colour = (np.array(to_rgb(colour)) for colour in ("#0284c7", "#e11d48", "#10b981"))
background = np.ones(3)
# The view, elevation and azimuth in radians, azimuth zero looking along +y, and the margin around the cubes as a
# share of the larger side.
elevation, azimuth = np.radians([22.0, 25.0 + 90.0])
margin = 0.14
# How much of what lies behind them each of the cube's faces hides, and how strongly the undeformed cube and the
# principal directions show over the white page; the radii of the
# edges, the undeformed cube's edges, the principal directions and the arrows' shafts, in pixels; an arrow's
# head, its length a share of the arrow's, and its radius a share of its own length.
face_opacity, ghost_opacity, principal_opacity = 0.14, 0.7, 0.8
ghost_tint, principal_tint = (opacity * colour + (1 - opacity) for opacity, colour in
                              ((ghost_opacity, ghost_colour), (principal_opacity, principal_colour)))
edge_radius, ghost_radius, principal_radius, shaft_radius = 0.9, 0.6, 0.6, 1.1
# The turning animation's radii as a share of those above, its picture that much narrower than a frame's.
turning_thickness = 0.62
head_share = 0.28
head_width = 0.4


def gpu_points(points: core.Point) -> solids.Point:
    """The GPU library's own points at the places of the cube's: their coordinates, read by pairing with the planes
    x, y and z, are the one hand-off from this algebra to the renderer's."""
    return solids.mv.point((stack((core.mv.x, core.mv.y, core.mv.z)) & points[..., None]).to_array())


def view(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point,
         principal: core.Point, room: float) -> solids.View:
    """A view framing every cube, arrow and principal direction given, with `room` to spare around each."""
    points = [core.corners, corners, centres + normal * NORMAL_SCALE, centres + shear * SHEAR_SCALE,
              REACH * principal]
    return solids.view([gpu_points(batch) for batch in points], room, elevation, azimuth, margin, 1)


def arrows(starts: core.Point, offsets: core.Point, pixel: float,
           thickness: float) -> tuple[solids.Quadric, solids.Quadric]:
    """Arrows from points along offsets `[arrows]`, their heads in proportion to them, their shafts' radius a share
    `thickness` of the one set above; those shorter than a pixel, a traction that vanishes, left out."""
    lengths = offsets.norm()                                                   # [arrows] Scalar+
    shown = lengths > pixel
    heads = head_share * lengths[shown].to_array()
    return solids.arrows(gpu_points(starts[shown]), gpu_points(starts[shown] + offsets[shown]),
                         thickness * shaft_radius * pixel, heads, head_width * heads)


def ghost(view: solids.View, thickness: float) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The undeformed cube, a ghost of thin pale rods, their radius a share `thickness` of the one set above."""
    corners = gpu_points(core.corners)                                         # [8] Point-
    edges, joints = solids.edges(corners[EDGES[:, 0]], corners[EDGES[:, 1]], thickness * ghost_radius * view.pixel)
    return [(edges, ghost_tint), (joints, ghost_tint)]


def cube(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point, principal: core.Point,
         view: solids.View, thickness: float) -> tuple[list[tuple[solids.Quadric, np.ndarray]],
                                                       list[tuple[solids.Quadric, np.ndarray, float]]]:
    """The deformed cube seen from one frame, its edges solid and its faces translucent, the normal traction on each
    face in blue and the shear in red, and the principal directions in green; the radii a share `thickness` of those
    set above."""
    places = gpu_points(corners)                                               # [8] Point-
    directions = REACH * principal                                             # [3] Point-
    normal_shafts, normal_heads = arrows(centres, normal * NORMAL_SCALE, view.pixel, thickness)
    shear_shafts, shear_heads = arrows(centres, shear * SHEAR_SCALE, view.pixel, thickness)
    edges, joints = solids.edges(places[EDGES[:, 0]], places[EDGES[:, 1]], thickness * edge_radius * view.pixel)
    # Each principal direction a rod through the centre, both ways.
    principal_rods = solids.rods(gpu_points(-directions), gpu_points(directions),
                                 thickness * principal_radius * view.pixel)
    parts = [(edges, edge_colour), (joints, edge_colour), (normal_shafts, normal_colour), (normal_heads, normal_colour),
             (shear_shafts, shear_colour), (shear_heads, shear_colour), (principal_rods, principal_tint)]
    body = solids.polyhedra(places, FACES)[None]                              # [1, 6] Plane+
    return parts, [(body, cube_colour, face_opacity)]


def draw_cube(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point, principal: core.Point,
              view: solids.View) -> np.ndarray:
    """The cube seen from one frame: the undeformed cube a ghost, the deformed one translucent, the normal
    traction on each face in blue and the shear in red, and the principal directions in green."""
    parts, translucent = cube(corners, centres, normal, shear, principal, view, 1.0)
    return solids.draw(ghost(view, 1.0) + parts, view, background, inks=[], translucent=translucent, meshes=[])


def draw_circles(ax, values: core.Scalar) -> None:
    """Mohr's circles of the principal stresses, on axes framing the largest."""
    stresses = np.sort(values.to_array())
    for low, high, style in [(0, 2, "-"), (0, 1, ":"), (1, 2, ":")]:
        centre, radius = (stresses[low] + stresses[high]) / 2, (stresses[high] - stresses[low]) / 2
        # The outer circle at a chart's line width, the inner two as thin as its frame.
        ax.add_patch(Circle((centre, 0), radius, fill=False, edgecolor="#1e293b" if style == "-" else "#94a3b8",
                            linestyle=style, linewidth=charts.LINE_WIDTH if style == "-" else charts.FRAME_WIDTH))
    ax.scatter(stresses, np.zeros(3), color="#1e293b", zorder=3)
    # The outer two outside the circle that passes through them, the middle one below its dot.
    for stress, name, shift, align in zip(stresses, ["$\\sigma_1$", "$\\sigma_2$", "$\\sigma_3$"], (-5, 0, 5),
                                          ("right", "center", "left")):
        ax.annotate(name, (stress, 0), xytext=(shift / 2, -2), textcoords="offset points", ha=align, va="top",
                    color="#1e293b")
    centre, radius = (stresses[0] + stresses[2]) / 2, (stresses[2] - stresses[0]) / 2
    margin = 1.3 * radius
    ax.set_xlim(centre - margin, centre + margin)
    ax.set_ylim(-margin, margin)
    ax.axhline(0, color="#94a3b8", linewidth=charts.FRAME_WIDTH, zorder=2)
    ax.axvline(0, color="#94a3b8", linewidth=charts.FRAME_WIDTH, zorder=2)
    ax.text(centre + 0.95 * margin, 0.04, "$\\sigma$", ha="right", va="bottom", color="#64748b")
    ax.text(0.04, 0.95 * margin, "$\\tau$", ha="left", va="top", color="#64748b")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def diameter(normal: core.Point, shear: core.Point) -> np.ndarray:
    """The ends of a frame's diameter, from the normal and shear traction on its faces `[6]`, read by pairing: the
    face facing x, its normal traction along x and its shear along y, and the face facing y, with the opposite shear,
    `[2, 2]`."""
    x, y, _ = core.axes
    along_x, along_y, sliding = ((x & normal[0]).to_array(), (y & normal[1]).to_array(),
                                 (y & shear[0]).to_array())
    return np.array([(along_x, sliding), (along_y, -sliding)])


def draw_diameter(ax, ends: np.ndarray, colour: str) -> tuple:
    """A frame's diameter between its ends `[2, 2]`, the face facing x filled and the face facing y open: the line
    and the two markers, for an animation to move."""
    line, = ax.plot(*ends.T, color=colour, zorder=4)
    filled = ax.scatter(*ends[0], color=colour, zorder=5)
    hollow = ax.scatter(*ends[1], zorder=5, facecolors="none", edgecolors=colour)
    return line, filled, hollow


def draw_mohr(ax, values: core.Scalar, normal: core.Point, shear: core.Point) -> None:
    """Mohr's circles of the principal stresses, and for each frame `[frames]` the diameter joining the
    normal and shear traction on the faces facing x, filled, and y, open, in the frame's colour."""
    draw_circles(ax, values)
    for frame, colour in zip(range(normal.shape[0]), COLOURS * normal.shape[0]):
        draw_diameter(ax, diameter(normal[frame], shear[frame]), colour)


@charts.style()
def draw_frames(views: tuple[core.Point, core.Point, core.Point, core.Point, core.Point],
                values: core.Scalar) -> plt.Figure:
    """The cube seen from each frame `[frames]`, all from one camera, and Mohr's circles with each frame's
    diameter."""
    corners, centres, normal, shear, principal = views
    count = corners.shape[0]
    # Nearly the whole column: a square panel a frame's cube, and Mohr's circles beside them.
    figure, panels = charts.subplots(190 * (count + 1), 190, ncols=count + 1)
    cubes = panels[:count]
    draw_mohr(panels[count], values, normal, shear)
    # The figure laid out with the cubes' axes bare, as their pictures show them; their rooms alike, one view fitted
    # to the first holds them all.
    for ax in cubes:
        ax.set_axis_off()
    figure.tight_layout()
    seen = fitted(view(*views, 0.0), cubes[0])
    for frame, ax in enumerate(cubes):
        show(ax, draw_cube(corners[frame], centres[frame], normal[frame], shear[frame], principal[frame], seen), seen)
    return figure


def animate_turning(values: core.Scalar, frames: Iterable[tuple[core.Point, core.Point, core.Point, core.Point,
                                                                   core.Point]]) -> Iterator[np.ndarray]:
    """Each frame of the cube seen from a turning frame, all from one camera framing the first with room around it,
    and its diameter on Mohr's circles; the undeformed cube and the circles set up once."""
    frames = iter(frames)
    first = next(frames)
    # Three fifths of the column: the cube beside Mohr's circles, which are drawn once under the charts' style; only
    # the cube's picture and the diameter change between frames.
    with charts.style():
        figure, (cube_ax, circles) = charts.subplots(480, 230, ncols=2)
        seen = fitted(view(*first, TURNING_ROOM), cube_ax)
        scene = Scene(seen, background, ghost(seen, turning_thickness), [], inks=[], translucent=[])
        picture = show(cube_ax, np.zeros((*seen.shape, 3), dtype=np.uint8), seen)
        draw_circles(circles, values)
        line, filled, hollow = draw_diameter(circles, np.zeros((2, 2)), COLOURS[0])
    for corners, centres, normal, shear, principal in chain([first], frames):
        parts, translucent = cube(corners, centres, normal, shear, principal, seen, turning_thickness)
        picture.set_data(scene.draw(parts, inks=[], translucent=translucent, meshes=[]))
        ends = diameter(normal, shear)
        line.set_data(*ends.T)
        filled.set_offsets(ends[:1])
        hollow.set_offsets(ends[1:])
        yield capture(figure)
    plt.close(figure)
