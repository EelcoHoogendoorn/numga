"""Drawing for unsymmetric bending: each section with its stress at the clamp, its principal axes, its
neutral axis, the load and the way the tip moves; the bent cantilevers beside the straight beam; the
Z-section's tip going around as its load turns; and each section pushed along the beam, with its
kern.

Sections are seen from beyond the tip, looking back at the clamp: across to the right and up. The
beams are drawn by their edges and a few cross-sections, lines only, in an orthographic view.
Coordinates are read out here alone: points through their weight, points at infinity as headings,
and lines by the foot of the centroid on them and their heading."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from examples.animation import capture
from examples.mechanics.bending import core

ga, mv = core.ga, core.mv
# Colours of the beam, of the straight beam, of the load, of the neutral axis, of the principal axes
# and of the kern, and the stress's colour map. The half width of the section views, the drawn length
# of the arrows in a section and at the tip, the pieces between drawn cross-sections, the view as
# (elevation, azimuth) in degrees, and the extent of the three-dimensional views along the beam, across
# and up.
BEAM, STRAIGHT, LOAD, NEUTRAL, PRINCIPAL, KERN = "#0369a1", "#cbd5e1", "#f97316", "#16a34a", "#64748b", "#a855f7"
STRESS = "RdBu_r"
STRESS_LEVELS = 257
SECTION_LIMIT = 0.75
ARROW, TIP_ARROW = 0.45, 1.4
OUTLINE_EVERY = 10
VIEW = (22, -58)
SPACE = ((-0.3, 10.3), (-3.8, 3.8), (-2.6, 2.0))


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


def unit(directions: np.ndarray) -> np.ndarray:
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def arrow(ax, start: np.ndarray, direction: np.ndarray, length: float, colour: str) -> None:
    ax.annotate("", xy=start + length * unit(direction), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2, shrinkA=0, shrinkB=0))


def line_through(ax, line: core.Strain, centre: core.Point, half: float, **style) -> None:
    """A line of the section, drawn for `half` either side of the foot of `centre` on it."""
    foot = flat((centre | line) ^ line)
    ax.plot(*(foot + np.outer([-half, half], unit(heading(line ^ mv.w)))).T, **style)


def draw_stress(ax, triangles: core.Point, stress: core.Scalar, largest: float):
    """Linear stress interpolated from the corners of the section's triangles."""
    points = flat(triangles).reshape(-1, 2)                         # [triangles * 3, 2]
    triangles = np.arange(len(points)).reshape(-1, 3)
    return ax.tricontourf(points[:, 0], points[:, 1], triangles, stress.to_array().reshape(-1),
                         levels=np.linspace(-largest, largest, STRESS_LEVELS), cmap=STRESS)


def frame_section(ax) -> None:
    ax.set_xlim(-SECTION_LIMIT, SECTION_LIMIT)
    ax.set_ylim(-SECTION_LIMIT, SECTION_LIMIT)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_meshes(sections: list[tuple[core.Point, core.Point]]) -> plt.Figure:
    """The triangulated cross-sections and their boundaries."""
    figure, axes = plt.subplots(1, len(sections), figsize=(4 * len(sections), 3.6))
    for ax, (triangles, boundary) in zip(axes, sections):
        points, outline = flat(triangles).reshape(-1, 2), flat(boundary)
        faces = np.arange(len(points)).reshape(-1, 3)
        ax.triplot(points[:, 0], points[:, 1], faces, color=STRAIGHT, linewidth=0.8)
        ax.plot(*np.concatenate([outline, outline[:1]]).T, color=BEAM, linewidth=1.5)
        frame_section(ax)
    figure.tight_layout()
    return figure


def draw_section(ax, triangles: core.Point, stress: core.Scalar, principal: core.Strain, load: core.Point,
                 tip: core.Point, strain: core.Strain, largest: float):
    """One section: the stress along the beam over each triangle, its principal axes dashed through
    the centroid, the neutral axis, the load's direction, and the direction the tip moves."""
    mesh = draw_stress(ax, triangles, stress, largest)
    centroid = core.centroid(triangles)
    for axis in principal:
        line_through(ax, axis, centroid, 2 * SECTION_LIMIT, color=PRINCIPAL, linestyle="--", linewidth=1.0)
    line_through(ax, strain, centroid, 2 * SECTION_LIMIT, color=NEUTRAL, linewidth=1.8)
    middle = flat(centroid)                                                  # [2]
    arrow(ax, middle, heading(load), ARROW, LOAD)
    arrow(ax, middle, flat(tip) - middle, ARROW, BEAM)
    ax.plot(*middle, "o", color="#0f172a", markersize=3)
    frame_section(ax)
    return mesh


def draw_beam(ax, outlines: core.Point, centres: core.Point, load: core.Point) -> None:
    """One cantilever: its boundary swept along the beam, the straight beam faintly, and the tip load."""
    bent = space(outlines)                                                   # [pieces + 1, boundary, 3]
    path = space(centres)                                                    # [pieces + 1, 3]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=-1))])
    straight = bent[0] + arc[:, None, None] * np.array([1.0, 0.0, 0.0])
    for shape, colour, width in ((straight, STRAIGHT, 0.8), (bent, BEAM, 1.1)):
        for edge in shape.transpose(1, 0, 2):
            ax.plot(*edge.T, color=colour, linewidth=width)
        for boundary in shape[::OUTLINE_EVERY]:
            ax.plot(*np.concatenate([boundary, boundary[:1]]).T, color=colour, linewidth=width)
    wall = np.array([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, -1.0, -1.0]])
    ax.plot(*wall.T, color="#475569", linewidth=1.4)
    ax.quiver(*path[-1], *(TIP_ARROW * unit(space(load))), color=LOAD, linewidth=2, arrow_length_ratio=0.25)
    ax.set_xlim(*SPACE[0])
    ax.set_ylim(*SPACE[1])
    ax.set_zlim(*SPACE[2])
    ax.set_box_aspect([high - low for low, high in SPACE], zoom=1.2)
    ax.set_proj_type("ortho")
    ax.view_init(*VIEW)
    ax.set_axis_off()


def draw_beams(sections: list[tuple[core.Point, core.Point]], load: core.Point,
               beams: list[tuple[core.Strain, core.Motor, core.Strain, core.Scalar]]) -> plt.Figure:
    """Top, each section with its stress at the clamp; below, each cantilever bent under the load."""
    figure = plt.figure(figsize=(4.4 * len(sections), 8.6), dpi=110)
    largest = max(np.abs(stress.to_array()).max() for _, _, _, stress in beams)
    for case, ((triangles, boundary), (strains, frames, principal, stress)) in enumerate(zip(sections, beams)):
        centres = frames >> core.centroid(triangles)                          # [pieces + 1] Point
        mesh = draw_section(figure.add_subplot(2, len(sections), case + 1), triangles, stress, principal, load,
                            centres[-1], strains[0], largest)
        draw_beam(figure.add_subplot(2, len(sections), len(sections) + case + 1, projection="3d"),
                  frames[..., None] >> boundary, centres, load)
    figure.subplots_adjust(left=0.01, right=0.9, bottom=0.0, top=0.97, hspace=0.0, wspace=0.05)
    figure.colorbar(mesh, cax=figure.add_axes([0.92, 0.6, 0.012, 0.28]), ticks=MaxNLocator(5))
    return figure


def draw_path(ax, tips: np.ndarray, frame: int, principal: core.Strain, load: core.Point) -> None:
    """The tip's path as the load turns, seen from beyond the tip, with the principal axes and the
    load's direction."""
    reach = 1.15 * np.abs(tips).max()
    for direction in heading(principal ^ mv.w):
        ax.plot(*np.outer([-reach, reach], unit(direction)).T, color=PRINCIPAL, linestyle="--", linewidth=1.0)
    ax.plot(*np.concatenate([tips, tips[:1]]).T, color=STRAIGHT, linewidth=1.5)
    ax.plot(*tips[:frame + 1].T, color=BEAM, linewidth=1.8)
    ax.plot(*tips[frame], "o", color=BEAM, markersize=6)
    arrow(ax, np.zeros(2), heading(load), 0.5 * reach, LOAD)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_axis_off()


def animate_turning(triangles: core.Point, boundary: core.Point, loads: core.Point,
                    beam: tuple[core.Strain, core.Motor, core.Strain, core.Scalar]) -> list[np.ndarray]:
    """Each frame: the bent beam, its section's stress at the clamp, and the tip's path so far."""
    strains, frames, principal, stress = beam
    centroid = core.centroid(triangles)                                      # [] Point
    centres = frames >> centroid                                             # [frames, pieces + 1] Point
    outlines = frames[..., None] >> boundary                                 # [frames, pieces + 1, boundary] Point
    tips = flat(centres[:, -1]) - flat(centroid)                             # [frames, 2]
    largest = np.abs(stress.to_array()).max()
    images = []
    for frame in range(loads.shape[0]):
        figure = plt.figure(figsize=(13.5, 4.6), dpi=100)
        draw_beam(figure.add_subplot(1, 3, 1, projection="3d"), outlines[frame], centres[frame], loads[frame])
        draw_section(figure.add_subplot(1, 3, 2), triangles, stress[frame], principal, loads[frame],
                     centres[frame, -1], strains[frame, 0], largest)
        draw_path(figure.add_subplot(1, 3, 3), tips, frame, principal, loads[frame])
        figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
        images.append(capture(figure))
        plt.close(figure)
    return images


def draw_push(ax, triangles: core.Point, point: core.Point, strain: core.Strain, stress: core.Scalar,
              view: core.Point, corners: core.Scalar, largest: float):
    """One section pushed through one point: the stress over each triangle, the kern, the points
    whose pull stretches every corner, and the neutral axis, the push point's polar."""
    mesh = draw_stress(ax, triangles, stress, largest)
    grid = flat(view)                                                        # [rows, columns, 2]
    # The least stretch over the corners under a pull through each point: positive on the kern.
    least = corners.to_array().min(axis=(-2, -1))                            # [rows, columns]
    ax.contourf(grid[..., 0], grid[..., 1], least, levels=[0.0, least.max()], colors=[KERN], alpha=0.25)
    ax.contour(grid[..., 0], grid[..., 1], least, levels=[0.0], colors=[KERN], linewidths=1.2)
    line_through(ax, strain, core.centroid(triangles), 4 * SECTION_LIMIT, color=NEUTRAL, linewidth=1.8)
    ax.plot(*flat(point), "X", color=LOAD, markersize=9, markeredgecolor="white")
    ax.plot(*flat(core.centroid(triangles)), "o", color="#0f172a", markersize=3)
    frame_section(ax)
    return mesh


def draw_pushes(sections: list[tuple[core.Point, core.Point]], points: core.Point, view: core.Point,
                pushes: list[tuple[core.Strain, core.Scalar, core.Scalar]]) -> plt.Figure:
    """A row per push point and a column per section: the section pushed along the beam through the point,
    with its kern."""
    figure = plt.figure(figsize=(4.0 * len(sections), 8.4), dpi=110)
    largest = max(np.abs(stress.to_array()).max() for _, stress, _ in pushes)
    for push in range(points.shape[0]):
        for case, ((triangles, _), (strains, stress, corners)) in enumerate(zip(sections, pushes)):
            ax = figure.add_subplot(points.shape[0], len(sections), push * len(sections) + case + 1)
            mesh = draw_push(ax, triangles, points[push], strains[push], stress[push], view, corners, largest)
    figure.subplots_adjust(left=0.01, right=0.9, bottom=0.01, top=0.99, hspace=0.05, wspace=0.05)
    figure.colorbar(mesh, cax=figure.add_axes([0.92, 0.3, 0.012, 0.4]), ticks=MaxNLocator(5))
    return figure
