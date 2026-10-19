"""Drawing for thin-walled bending and twisting: the sections' walls with their centroids and shear
centres; each section with its stress at the clamp, its principal axes, its neutral axis, the load and
the way the tip moves; the bent cantilevers beside the straight beam; a load turning or moving across
the tip; and each section pushed along the beam, with the polars of its nodes and its kern.

Sections are seen from beyond the tip, looking back at the clamp: across to the right and up. Walls are
drawn as thick lines along their midlines. The beams are drawn as each wall swept along the beam, in an
orthographic view. Coordinates are read out here alone:
points through their weight, points at infinity as headings, and lines by the foot of the centroid on
them and their heading."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.mechanics.bending import core

ga, mv = core.ga, core.mv
# Colours of the beam, of the straight beam, of the load, of the neutral axis, of the principal axes,
# and of the kern, and the stress's colour map. The width of a drawn wall and the pieces
# each wall's stress is drawn in. The half width of the section views, the drawn length of the arrows
# in a section and at the tip, the pieces between drawn cross-sections, the view as (elevation,
# azimuth) in degrees, and the extent of the three-dimensional views along the beam, across and up.
BEAM, STRAIGHT, LOAD, NEUTRAL, PRINCIPAL, KERN = "#0369a1", "#cbd5e1", "#f97316", "#16a34a", "#64748b", "#a855f7"
STRESS = "RdBu_r"
WALL_WIDTH = 5.0
WALL_PIECES = 12
SECTION_LIMIT = 1.2
ARROW, TIP_ARROW = 0.45, 1.4
OUTLINE_EVERY = 10
# The pieces between the stubby beam's drawn cross-sections.
ELEVATION_EVERY = 5
VIEW = (22, -58)
SPACE = ((-0.3, 10.3), (-3.8, 3.8), (-2.6, 2.0))
# The lamp's direction along the beam, across and up, and the share of a colour lit where no lamp reaches.
LAMP = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
AMBIENT = 0.45


def flat(points: core.Point) -> np.ndarray:
    """Points in the section views `[..., 2]`: across and up, any part along the beam projected away."""
    coordinates = points.cast(ga.subspace("ywz wxz xyz")).kernel
    return coordinates[..., :2] / coordinates[..., 2:]


def heading(directions: core.Point) -> np.ndarray:
    """Points at infinity in the section views `[..., 2]`: across and up."""
    return directions.cast(ga.subspace("ywz wxz")).kernel


def space(points: core.Point) -> np.ndarray:
    """Points in the three-dimensional views `[..., 3]`: along the beam, across, and up."""
    return points.cast(ga.subspace("xwy ywz wxz")).kernel


def twist(frames: core.Motor) -> np.ndarray:
    """How far each frame turns a cross-section about the beam, in radians: the turn of the direction
    across the section."""
    turned = heading(frames >> mv.x.dual())                                  # [..., 2]
    return np.arctan2(turned[..., 1], turned[..., 0])


def unit(directions: np.ndarray) -> np.ndarray:
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def arrow(ax, start: np.ndarray, direction: np.ndarray, length: float, colour: str) -> None:
    ax.annotate("", xy=start + length * unit(direction), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2, shrinkA=0, shrinkB=0))


def line_through(ax, line: core.Strain, centre: core.Point, half: float, **style) -> None:
    """A line of the section, drawn for `half` either side of the foot of `centre` on it."""
    foot = flat((centre | line) ^ line)
    ax.plot(*(foot + np.outer([-half, half], unit(heading(line ^ mv.w)))).T, **style)


def draw_stress(ax, walls: core.Point, stress: core.Scalar, largest: float) -> None:
    """The walls as thick lines coloured by the stress along the beam, linear along each wall between
    its ends."""
    ends = flat(walls)                                                       # [walls, 2, 2]
    values = stress.to_array()                                               # [walls, 2]
    fractions = np.linspace(0, 1, WALL_PIECES + 1)[:, None]                  # [pieces + 1, 1]
    along = ends[:, None, 0] * (1 - fractions) + ends[:, None, 1] * fractions   # [walls, pieces + 1, 2]
    pieces = np.stack([along[:, :-1], along[:, 1:]], axis=-2).reshape(-1, 2, 2)
    middles = (fractions[:-1, 0] + fractions[1:, 0]) / 2
    colours = (values[:, None, 0] * (1 - middles) + values[:, None, 1] * middles).reshape(-1)
    lines = LineCollection(pieces, array=colours, cmap=STRESS, norm=Normalize(-largest, largest),
                           linewidths=WALL_WIDTH, capstyle="round")
    ax.add_collection(lines)


def frame_section(ax) -> None:
    ax.set_xlim(-SECTION_LIMIT, SECTION_LIMIT)
    ax.set_ylim(-SECTION_LIMIT, SECTION_LIMIT)
    ax.set_aspect("equal")
    ax.set_axis_off()


def mark_centres(ax, centre: core.Point, shear: core.Point) -> None:
    """The centroid as a dot, and the shear centre as a ring."""
    ax.plot(*flat(centre), "o", color="#0f172a", markersize=3)
    ax.plot(*flat(shear), "o", markerfacecolor="none", markeredgecolor="#0f172a", markersize=7, markeredgewidth=1.4)


def draw_walls(sections: list[tuple[core.Point, np.ndarray, np.ndarray]], centres: list[core.Point],
               shears: list[core.Point]) -> plt.Figure:
    """The sections' walls along their midlines, each with its centroid and its shear centre."""
    figure, axes = plt.subplots(1, len(sections), figsize=(4 * len(sections), 3.8))
    for ax, (nodes, ends, _), centre, shear in zip(axes, sections, centres, shears):
        ax.add_collection(LineCollection(flat(nodes[ends]), colors=STRAIGHT, linewidths=WALL_WIDTH, capstyle="round"))
        mark_centres(ax, centre, shear)
        frame_section(ax)
    figure.tight_layout()
    return figure


def draw_section(ax, walls: core.Point, stress: core.Scalar, principal: core.Strain, centre: core.Point,
                 shear: core.Point, through: core.Point, load: core.Point, tip: core.Point, strain: core.Twist,
                 largest: float) -> None:
    """One section: the stress along the beam over its walls at the clamp, the walls at the tip as the
    tip has moved, its principal axes dashed through the centroid, the neutral axis, and the load's
    direction where it acts."""
    draw_stress(ax, walls, stress, largest)
    ax.add_collection(LineCollection(flat(tip), colors=BEAM, linewidths=1.6, capstyle="round"))
    for axis in principal:
        line_through(ax, axis, centre, 2 * SECTION_LIMIT, color=PRINCIPAL, linestyle="--", linewidth=1.0)
    line_through(ax, strain, centre, 2 * SECTION_LIMIT, color=NEUTRAL, linewidth=1.8)
    arrow(ax, flat(through), heading(load), ARROW, LOAD)
    mark_centres(ax, centre, shear)
    frame_section(ax)


def shaded(quads: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """Colours of quads `[quads, 4, 3]` lit by the lamp from either side, by their normals."""
    normals = np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
    light = np.abs(normals @ LAMP) / np.maximum(np.linalg.norm(normals, axis=-1), 1e-12)
    return np.concatenate([colours[:, :3] * (AMBIENT + (1 - AMBIENT) * light[:, None]), colours[:, 3:]], axis=-1)


def draw_beam(ax, walls: core.Point, shears: core.Point, through: core.Point, load: core.Point) -> None:
    """One cantilever: each wall swept along the beam as shaded strips, one collection so that the strips
    are drawn in depth order; the straight beam faintly in the same collection; the line of shear
    centres the beam twists about, dashed; and the tip load where it acts."""
    bent = space(walls)                                                      # [pieces + 1, walls, 2, 3]
    middle = bent.mean(axis=(1, 2))                                          # [pieces + 1, 3]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(middle, axis=0), axis=-1))])
    straight = bent[0] + arc[:, None, None, None] * np.array([1.0, 0.0, 0.0])   # [pieces + 1, walls, 2, 3]
    strips = [np.stack([shape[:-1, :, 0], shape[:-1, :, 1], shape[1:, :, 1], shape[1:, :, 0]], axis=-2).reshape(-1, 4, 3)
              for shape in (straight, bent)]                                 # [pieces * walls, 4, 3] each
    colours = np.concatenate([np.tile(to_rgba(STRAIGHT, 0.35), (len(strips[0]), 1)),
                              np.tile(to_rgba(BEAM), (len(strips[1]), 1))])
    quads = np.concatenate(strips)
    ax.add_collection3d(Poly3DCollection(quads, facecolors=shaded(quads, colours), edgecolors="none"))
    ax.plot(*space(shears).T, color="#0f172a", linestyle="--", linewidth=1.0)
    wall = np.array([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, -1.0, -1.0]])
    ax.plot(*wall.T, color="#475569", linewidth=1.4)
    ax.quiver(*space(through), *(TIP_ARROW * unit(space(load))), color=LOAD, linewidth=2, arrow_length_ratio=0.25)
    ax.set_xlim(*SPACE[0])
    ax.set_ylim(*SPACE[1])
    ax.set_zlim(*SPACE[2])
    ax.set_box_aspect([high - low for low, high in SPACE], zoom=1.2)
    ax.set_proj_type("ortho")
    ax.view_init(*VIEW)
    ax.set_axis_off()


def draw_beams(sections: list[tuple[core.Point, np.ndarray, np.ndarray]], centres: list[core.Point],
               shears: list[core.Point], load: core.Point,
               beams: list[tuple[core.Twist, core.Motor, core.Strain, core.Scalar]]) -> plt.Figure:
    """Top, each section with its stress at the clamp; below, each cantilever bent under the load
    through its centroid."""
    figure = plt.figure(figsize=(4.4 * len(sections), 8.6), dpi=110)
    largest = max(np.abs(stress.to_array()).max() for _, _, _, stress in beams)
    for case, ((nodes, ends, _), centre, shear, (strains, frames, principal, stress)) in enumerate(
            zip(sections, centres, shears, beams)):
        draw_section(figure.add_subplot(2, len(sections), case + 1), nodes[ends], stress, principal, centre,
                             shear, centre, load, frames[-1] >> nodes[ends], strains[0], largest)
        draw_beam(figure.add_subplot(2, len(sections), len(sections) + case + 1, projection="3d"),
                  frames[..., None, None] >> nodes[ends], frames >> shear, frames[-1] >> centre, load)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.0, top=0.97, hspace=0.0, wspace=0.05)
    return figure


def draw_stubby(section: tuple[core.Point, np.ndarray, np.ndarray], frames: list[core.Motor], centre: core.Point,
                load: core.Point) -> plt.Figure:
    """A stubby beam seen from the side, along the beam to the right and up: the lines along it through its
    walls' ends, and its cross-sections, solid with the walls' shear stiffness and dashed with the slides
    held rigid, beside the straight beam, with the clamp and the load at the tip."""
    nodes, ends, _ = section
    figure, ax = plt.subplots(figsize=(7.0, 4.2), dpi=110)
    for placed, colour, style in zip(frames, (BEAM, PRINCIPAL), ("-", "--")):
        lines = space(placed[:, None] >> nodes)[..., [0, 2]]                 # [pieces + 1, nodes, 2]
        sections = space(placed[::ELEVATION_EVERY, None, None] >> nodes[ends])[..., [0, 2]]   # [drawn, walls, 2, 2]
        ax.plot(*lines.transpose(2, 0, 1), color=colour, linestyle=style, linewidth=1.4)
        ax.add_collection(LineCollection(sections.reshape(-1, 2, 2), colors=colour, linestyles=style, linewidths=1.4))
    middle = space(frames[0][:, None] >> nodes).mean(axis=1)                  # [pieces + 1, 3]
    span = np.linalg.norm(np.diff(middle, axis=0), axis=-1).sum()
    straight = space(nodes)[..., [0, 2]]                                      # [nodes, 2]
    ax.plot(*np.stack([straight, straight + [span, 0.0]], axis=0).transpose(2, 0, 1), color=STRAIGHT, linewidth=1.0)
    arrow(ax, space(frames[0][-1] >> centre)[[0, 2]], space(load)[[0, 2]], ARROW, LOAD)
    ax.plot([0.0, 0.0], [-0.7, 0.7], color="#475569", linewidth=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    figure.tight_layout()
    return figure


def animate_loads(section: tuple[core.Point, np.ndarray, np.ndarray], centre: core.Point, shear: core.Point,
                  through: core.Point, loads: core.Point,
                  beam: tuple[core.Twist, core.Motor, core.Strain, core.Scalar]) -> list[np.ndarray]:
    """Each frame of a changing tip load, through points `through` in directions `loads`, either batched
    over the frames: the bent beam, and its section's stress at the clamp with the tip's cross-section."""
    nodes, ends, _ = section
    strains, frames, principal, stress = beam
    count = frames.shape[0]
    through, loads = through.broadcast_to((count,)), loads.broadcast_to((count,))
    walls = frames[..., None, None] >> nodes[ends]                           # [frames, pieces + 1, walls, 2] Point
    largest = np.abs(stress.to_array()).max()
    images = []
    for frame in range(count):
        figure = plt.figure(figsize=(9.0, 4.6), dpi=100)
        draw_beam(figure.add_subplot(1, 2, 1, projection="3d"), walls[frame], frames[frame] >> shear,
                  frames[frame, -1] >> through[frame], loads[frame])
        draw_section(figure.add_subplot(1, 2, 2), nodes[ends], stress[frame], principal, centre, shear,
                     through[frame], loads[frame], walls[frame, -1], strains[frame, 0], largest)
        figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
        images.append(capture(figure))
        plt.close(figure)
    return images


def kern(polars: core.Strain, centre: core.Point) -> np.ndarray:
    """The kern as a polygon `[corners, 2]`: the view cut down to the centroid's side of every node's
    polar. Each polar is read as the stretch it gives the fibres, affine across the section."""
    probes = core.point([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])               # [3] Point
    values = (polars[:, None] & (core.beam_direction & probes)).to_array()  # [nodes, 3]
    lines = np.stack([values[:, 1] - values[:, 0], values[:, 2] - values[:, 0], values[:, 0]], axis=-1)
    lines *= np.sign(lines @ np.append(flat(centre), 1.0))[:, None]          # positive on the centroid's side
    polygon = SECTION_LIMIT * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for line in lines:
        # Keep each corner on the centroid's side, and where an edge crosses the polar, the crossing.
        side = polygon @ line[:2] + line[2]                                  # [corners]
        kept = []
        for corner, following, here, there in zip(polygon, np.roll(polygon, -1, axis=0), side, np.roll(side, -1)):
            if here >= 0:
                kept.append(corner)
            if here * there < 0:
                kept.append(corner + (following - corner) * here / (here - there))
        polygon = np.array(kept)
    return polygon


def draw_push(ax, walls: core.Point, point: core.Point, strain: core.Strain, stress: core.Scalar,
              polars: core.Strain, centre: core.Point, largest: float) -> None:
    """One section pushed through one point: the stress over its walls, the kern, and the neutral axis,
    the push point's polar."""
    draw_stress(ax, walls, stress, largest)
    ax.fill(*kern(polars, centre).T, facecolor=to_rgba(KERN, 0.3), edgecolor=KERN, linewidth=1.4, zorder=3)
    line_through(ax, strain, centre, 4 * SECTION_LIMIT, color=NEUTRAL, linewidth=1.8)
    ax.plot(*flat(point), "X", color=LOAD, markersize=9, markeredgecolor="white", zorder=4)
    ax.plot(*flat(centre), "o", color="#0f172a", markersize=3, zorder=4)
    frame_section(ax)


def draw_pushes(sections: list[tuple[core.Point, np.ndarray, np.ndarray]], centres: list[core.Point],
                points: core.Point, pushes: list[tuple[core.Strain, core.Scalar, core.Strain]]) -> plt.Figure:
    """A row per push point and a column per section: the section pushed along the beam through the
    point, with its kern."""
    figure = plt.figure(figsize=(4.0 * len(sections), 8.4), dpi=110)
    largest = max(np.abs(stress.to_array()).max() for _, stress, _ in pushes)
    for push in range(points.shape[0]):
        for case, ((nodes, ends, _), centre, (strains, stress, polars)) in enumerate(zip(sections, centres, pushes)):
            ax = figure.add_subplot(points.shape[0], len(sections), push * len(sections) + case + 1)
            draw_push(ax, nodes[ends], points[push], strains[push], stress[push], polars, centre, largest)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.05, wspace=0.05)
    return figure
