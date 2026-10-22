"""Drawing for the stressed cube: the cube seen from each frame with its face tractions, on the GPU, and Mohr's
circles.

The deformed cube is a translucent solid, its edges dark rods, and the undeformed cube a ghost of thin pale rods; the normal traction on each face is a blue arrow from the face's centre, the shear a red one, and the principal
directions are thin green rods through the centre."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.animation import capture
from examples.mechanics.stress import core
from examples.wgsl import show, solids

# The cube's edges and faces, by the corners' indices: corner 4 i + 2 j + k sits at the i-th, j-th and
# k-th of -0.5 and 0.5 along x, y and z.
EDGES = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
FACES = np.array([(4, 5, 7, 6), (0, 2, 3, 1), (2, 6, 7, 3), (0, 1, 5, 4), (1, 3, 7, 5), (0, 4, 6, 2)])
# The arrows' lengths per unit traction, and the colours of the frames' diameters on Mohr's circle.
NORMAL_SCALE, SHEAR_SCALE = 0.4, 0.5
COLOURS = ("#0284c7", "#10b981", "#e11d48")
# The principal directions are drawn out to this distance from the centre, just past the cube.
REACH = 0.75
# The colours of the cube, its edges, the undeformed cube, the tractions and the principal directions.
cube_colour, edge_colour, ghost_colour = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#0f172a", "#94a3b8"))
normal_colour, shear_colour, principal_colour = (np.array(to_rgb(colour)) for colour in ("#0284c7", "#e11d48", "#10b981"))
background = np.ones(3)
# The view, elevation and azimuth in radians, azimuth zero looking along +y; the margin around the cubes as a share
# of the larger side, and the pixels across a picture.
elevation, azimuth = np.radians([22.0, 25.0 + 90.0])
margin = 0.14
n_pixels = 540
# How much of what lies behind them each of the cube's faces hides, and how strongly the undeformed cube and the
# principal directions show over the white page; the radii of the
# edges, the undeformed cube's edges, the principal directions and the arrows' shafts, in pixels; an arrow's
# head, its length a share of the arrow's, and its radius a share of its own length.
face_opacity, ghost_opacity, principal_opacity = 0.14, 0.7, 0.8
ghost_tint, principal_tint = (opacity * colour + (1 - opacity) for opacity, colour in
                              ((ghost_opacity, ghost_colour), (principal_opacity, principal_colour)))
edge_radius, ghost_radius, principal_radius, shaft_radius = 1.6, 1.0, 1.0, 2.0
head_share = 0.28
head_width = 0.4


def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def view(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point,
         principal: core.Point) -> solids.View:
    """A view framing every cube, arrow and principal direction given."""
    centres = coordinates(centres)
    points = [coordinates(core.corners), coordinates(corners), centres + coordinates(normal) * NORMAL_SCALE,
              centres + coordinates(shear) * SHEAR_SCALE, REACH * coordinates(principal)]
    return solids.view([solids.mv.point(batch) for batch in points], 0.0, elevation, azimuth, margin, n_pixels)


def arrows(starts: np.ndarray, offsets: np.ndarray, pixel: float) -> tuple[solids.Quadric, solids.Quadric]:
    """Arrows from points along offsets `[arrows, 3]`, their heads in proportion to them; those shorter than a pixel,
    a traction that vanishes, left out."""
    lengths = np.linalg.norm(offsets, axis=-1)
    shown = lengths > pixel
    return solids.arrows(solids.mv.point(starts[shown]), solids.mv.point(starts[shown] + offsets[shown]),
                         shaft_radius * pixel, head_share * lengths[shown], head_width * head_share * lengths[shown])


def draw_cube(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point, principal: core.Point,
              view: solids.View) -> np.ndarray:
    """The cube seen from one frame: the undeformed cube a ghost, the deformed one translucent, the normal
    traction on each face in blue and the shear in red, and the principal directions in green."""
    ghost = solids.mv.point(coordinates(core.corners))                         # [8] Point
    places = solids.mv.point(coordinates(corners))                             # [8] Point
    centres = coordinates(centres)                                             # [6, 3]
    directions = REACH * coordinates(principal)                                # [3, 3]
    normal_shafts, normal_heads = arrows(centres, coordinates(normal) * NORMAL_SCALE, view.pixel)
    shear_shafts, shear_heads = arrows(centres, coordinates(shear) * SHEAR_SCALE, view.pixel)
    edges, joints = solids.edges(places[EDGES[:, 0]], places[EDGES[:, 1]], edge_radius * view.pixel)
    ghost_edges, ghost_joints = solids.edges(ghost[EDGES[:, 0]], ghost[EDGES[:, 1]], ghost_radius * view.pixel)
    # Each principal direction a rod through the centre, both ways.
    principal_rods = solids.rods(solids.mv.point(-directions), solids.mv.point(directions),
                                 principal_radius * view.pixel)
    parts = [(edges, edge_colour), (joints, edge_colour), (normal_shafts, normal_colour), (normal_heads, normal_colour),
             (shear_shafts, shear_colour), (shear_heads, shear_colour), (ghost_edges, ghost_tint),
             (ghost_joints, ghost_tint), (principal_rods, principal_tint)]
    cube = solids.polyhedra(coordinates(corners), FACES)[None]                # [1, 6] Quadric
    return solids.draw(parts, view, background, translucent=[(cube, cube_colour, face_opacity)])


def draw_mohr(ax, values: core.Scalar, normal: core.Point, shear: core.Point) -> None:
    """Mohr's circles of the principal stresses, and for each frame `[frames]` the diameter joining the
    normal and shear traction on the faces facing x, filled, and y, open, in the frame's colour."""
    stresses = np.sort(values.to_array())
    angle = np.linspace(0, 2 * np.pi, 200)
    for low, high, style in [(0, 2, "-"), (0, 1, ":"), (1, 2, ":")]:
        centre, radius = (stresses[low] + stresses[high]) / 2, (stresses[high] - stresses[low]) / 2
        ax.plot(centre + radius * np.cos(angle), radius * np.sin(angle), color="#1e293b" if style == "-" else "#94a3b8",
                linestyle=style, linewidth=1.8 if style == "-" else 1.1)
    ax.scatter(stresses, np.zeros(3), color="#1e293b", s=18, zorder=3)
    # The outer two outside the circle that passes through them, the middle one below its dot.
    for stress, name, shift, align in zip(stresses, ["$\\sigma_1$", "$\\sigma_2$", "$\\sigma_3$"], (-5, 0, 5),
                                          ("right", "center", "left")):
        ax.annotate(name, (stress, 0), xytext=(shift, -5), textcoords="offset points", ha=align, va="top",
                    fontsize=9, color="#1e293b")
    normal, shear = coordinates(normal), coordinates(shear)                  # [frames, 6, 3]
    for frame, colour in zip(range(len(normal)), COLOURS * len(normal)):
        # The face facing x: its normal traction along x and its shear along y; the face facing y, with
        # the opposite shear.
        x_face = (normal[frame, 0, 0], shear[frame, 0, 1])
        y_face = (normal[frame, 1, 1], -shear[frame, 0, 1])
        ax.plot(*zip(x_face, y_face), color=colour, linewidth=1.6, zorder=4)
        ax.scatter(*x_face, color=colour, s=65, zorder=5)
        ax.scatter(*y_face, s=65, zorder=5, facecolors="none", edgecolors=colour, linewidth=1.6)
    centre, radius = (stresses[0] + stresses[2]) / 2, (stresses[2] - stresses[0]) / 2
    margin = 1.3 * radius
    ax.set_xlim(centre - margin, centre + margin)
    ax.set_ylim(-margin, margin)
    ax.axhline(0, color="#94a3b8", linewidth=0.9, zorder=2)
    ax.axvline(0, color="#94a3b8", linewidth=0.9, zorder=2)
    ax.text(centre + 0.95 * margin, 0.04, "$\\sigma$", ha="right", va="bottom", fontsize=11, color="#64748b")
    ax.text(0.04, 0.95 * margin, "$\\tau$", ha="left", va="top", fontsize=11, color="#64748b")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def draw_frames(views: tuple[core.Point, core.Point, core.Point, core.Point, core.Point],
                values: core.Scalar) -> plt.Figure:
    """The cube seen from each frame `[frames]`, all from one camera, and Mohr's circles with each frame's
    diameter."""
    corners, centres, normal, shear, principal = views
    count = corners.shape[0]
    seen = view(*views)
    figure = plt.figure(figsize=(4.5 * (count + 1), 4.5), dpi=120)
    for frame in range(count):
        show(figure.add_subplot(1, count + 1, frame + 1),
             draw_cube(corners[frame], centres[frame], normal[frame], shear[frame], principal[frame], seen), seen)
    draw_mohr(figure.add_subplot(1, count + 1, count + 1), values, normal, shear)
    figure.tight_layout()
    return figure


def animate_turning(values: core.Scalar, frames: Iterable[tuple[core.Point, core.Point, core.Point, core.Point,
                                                                   core.Point]]) -> list[np.ndarray]:
    """Each frame of the cube seen from a turning frame, all from one camera, and its diameter on Mohr's
    circles."""
    frames = list(frames)
    seen = view(*(stack(list(parts)) for parts in zip(*frames)))   # [frames, ...] each
    images = []
    for corners, centres, normal, shear, principal in frames:
        figure = plt.figure(figsize=(9.5, 4.5), dpi=100)
        show(figure.add_subplot(1, 2, 1), draw_cube(corners, centres, normal, shear, principal, seen), seen)
        draw_mohr(figure.add_subplot(1, 2, 2), values, normal[None], shear[None])
        images.append(capture(figure))
        plt.close(figure)
    return images
