"""Orbital paths, their spinor lifts, and integration errors."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import room, simplices
from examples.util.wgsl.camera import Camera, flat_panel
from examples.mechanics.kepler import core

EXACT_COLOUR = "#626c76"
PHYSICAL_COLOUR = "#d06c36"
REGULARIZED_COLOUR = "#287fa0"
BODY_COLOUR = "#ff00b7"
CENTRE_COLOUR = "#e5ab32"
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.4"
# The margin around a lifted path, as a share of its extent, and around the animation's paths.
LABEL_ROOM = 0.12
MARGIN = 0.05
# The room beyond the periapsis detail's box for its frame, as a share of the box's half width.
FRAME_ROOM = 0.02
# The centre of attraction's marker, twice a chart's marker across, and the white rim of the centre's and the body's
# marks, in pixels; in the animation, the width of the paths and the radii of the centre and of the body.
CENTRE_RADIUS = charts.MARKER_PX
RIM = charts.LINE_PX / 2
ANIMATION_WIDTH, ANIMATION_CENTRE, BODY_RADIUS = 1.4, 3.2, 3.8
# How much the exact orbit covers, under the two integrated ones.
EXACT_OPACITY = 0.5


# --- plumbing -------------------------------------------------------------------------
def plane_readings(points: core.Point) -> core.Scalar:
    """Points in the orbit's plane read across and up, `[2, ...]`: their pairings with the planes x = 0 and y = 0."""
    planes = stack((core.mv.x, core.mv.y))                                    # [2] Vector+
    return planes.reshape((2,) + (1,) * points.ndim) & points


def lifted_points(spinors: core.Spinor) -> core.Point:
    """Spinors drawn in the plane of directions: each acting from one side on the direction x, which carries the
    spinors' scalar part along x and their xy part along y."""
    return core.mv.x.dual() * spinors


def framed(ax: plt.Axes, path: core.Point, margin: float) -> Camera:
    """A panel framed about a path in the orbit's plane and the centre of attraction, with a margin as a share of
    their extent, widened to fill the room the figure's layout gives the panel at equal scales; its camera, reading
    by the planes x and y."""
    # The path's extent, stretched to hold the centre of attraction, the origin, which reads zero.
    readings = plane_readings(path)                                           # [2, samples] Scalar+
    low = readings.min(axis=-1).clip(-np.inf, 0).to_array()
    high = readings.max(axis=-1).clip(0, np.inf).to_array()
    spans = (1 + 2 * margin) * (high - low)
    shape = np.array(room(ax))
    # The span of a pixel that holds both spans in the room, and the room's half extent at that span.
    half = shape * (spans / shape).max() / 2
    middle = (low + high) / 2
    ax.set(xlim=(middle[0] - half[0], middle[0] + half[0]), ylim=(middle[1] - half[1], middle[1] + half[1]),
           aspect="equal")
    return flat_panel(ax, core.ga)


def centre_drawn(radius: float, camera: Camera) -> list[np.ndarray]:
    """The centre of attraction, rimmed in white."""
    origin = (0 * core.mv.x.dual())[None]                                      # [1] Point-
    return [simplices.markers(origin, np.ones(3), radius + RIM, camera),
            simplices.markers(origin, np.array(to_rgb(CENTRE_COLOUR)), radius, camera)]


def cross(ax: plt.Axes, across: str, up: str) -> None:
    """A plane without scales, the lines of its faint cross through the origin named at their ends where they are in
    view."""
    ax.text(1, 0, across, transform=ax.get_yaxis_transform(), ha="right", va="top", color=LABEL_COLOUR, clip_on=True)
    ax.text(0, 1, f" {up}", transform=ax.get_xaxis_transform(), ha="left", va="top", color=LABEL_COLOUR, clip_on=True)
    ax.set_axis_off()


def place(across: np.ndarray, up: np.ndarray) -> core.Point:
    """Points of the orbit's plane, so far along x and along y."""
    return core.mv.x.dual() * across + core.mv.y.dual() * up


def cross_drawn(box: tuple[float, float, float, float], camera: Camera) -> np.ndarray:
    """The faint cross through the origin, from edge to edge of a box, left, right, bottom and top, as thin as a
    chart's frame, which it stands in for: its picture."""
    left, right, bottom, top = box
    ends = place(np.array([left, right, 0, 0]), np.array([0, 0, bottom, top]))   # [4] Point-
    return simplices.strokes(ends, simplices.Mesh.chains([2, 2]), np.array(to_rgb(CROSS_COLOUR)), charts.FRAME_PX,
                             camera, cap=simplices.Cap.BUTT)


def frame_drawn(box: tuple[float, float, float, float], camera: Camera) -> np.ndarray:
    """A frame round a box, left, right, bottom and top, in the cross's colour and width: its picture."""
    left, right, bottom, top = box
    corners = place(np.array([left, right, right, left, left]), np.array([bottom, bottom, top, top, bottom]))   # [5] Point-
    return simplices.strokes(corners, simplices.Mesh.chains([5]), np.array(to_rgb(CROSS_COLOUR)), charts.FRAME_PX,
                             camera)


def path_drawn(path: core.Point, colour: str, width: float, camera: Camera, opacity: float = 1.0) -> np.ndarray:
    """A path `[samples]` as a stroke."""
    return simplices.strokes(path, simplices.Mesh.chains([len(path)]), np.array(to_rgb(colour)), width, camera,
                             opacity=opacity)


@charts.style()
def draw_lift(positions: core.Point, spinors: core.Spinor) -> plt.Figure:
    """A planar orbital passage and the half-turn of its position spinor."""
    # Three fifths of the column: two paths side by side, room enough for their samples' dots to stand apart.
    figure, panels = charts.subplots(480, 200, ncols=2, layout="constrained")
    for ax, path, (across, up) in zip(panels, (positions, lifted_points(spinors)), (("x", "y"), ("scalar", "xy"))):
        # Room beyond the path for the names of the cross.
        camera = framed(ax, path, LABEL_ROOM)
        cross(ax, across, up)
        samples = plt.get_cmap("viridis")(np.linspace(0, 1, len(path)))[:, :3]
        # The centre of attraction twice the size of the samples.
        simplices.overlay(ax, [cross_drawn(camera.extent, camera),
                               path_drawn(path, REGULARIZED_COLOUR, charts.LINE_PX, camera),
                               simplices.markers(path, samples, charts.MARKER_PX / 2, camera),
                               *centre_drawn(CENTRE_RADIUS, camera)])
    return figure


@charts.style()
def draw_comparison(reference: core.State, physical: core.State, regularized: core.State) -> plt.Figure:
    """One orbit at equal step counts, with a periapsis detail below each full path."""
    # The exact orbit's extent in each case, `[2, cases]` each.
    exact = plane_readings(reference.position)                                # [2, samples, cases] Scalar+
    lows, highs = exact.min(axis=1).to_array(), exact.max(axis=1).to_array()
    cases = lows.shape[1]
    # Three quarters of the column: a full path and its periapsis a case, over one legend.
    figure, panels = charts.subplots(600, 400, nrows=2, ncols=cases, squeeze=False, layout="constrained")
    # The exact orbit wide and faint, under the two integrated ones.
    styles = [(reference.position, EXACT_COLOUR, 2 * charts.LINE_PX, EXACT_OPACITY, "Exact"),
              (physical.position, PHYSICAL_COLOUR, charts.LINE_PX, 1.0, "Physical clock"),
              (regularized.position, REGULARIZED_COLOUR, charts.LINE_PX, 1.0, "Spinor clock")]
    # Each panel's box, left, right, bottom and top: the full path, and the periapsis detail.
    boxes = np.empty((2, cases, 4))
    for case in range(cases):
        low, high = lows[:, case], highs[:, case]
        span = np.max(high - low)
        centre = (low + high) / 2
        half_width = 0.12 * span
        boxes[0, case] = (centre[0] - 0.6 * span, centre[0] + 0.6 * span, centre[1] - 0.6 * span,
                          centre[1] + 0.6 * span)
        boxes[1, case] = (low[0] - half_width, low[0] + half_width, -half_width, half_width)
        for row in range(2):
            ax = panels[row, case]
            left, right, bottom, top = boxes[row, case]
            # The periapsis detail's limits a little wider than its box, room for the frame drawn round it.
            room_beyond = (FRAME_ROOM if row == 1 else 0) * (right - left) / 2
            ax.set(xlim=(left - room_beyond, right + room_beyond), ylim=(bottom - room_beyond, top + room_beyond),
                   aspect="equal")
            cross(ax, "x", "y")
    figure.get_layout_engine().set(hspace=0.08)
    handles = [Line2D([], [], color=colour, linewidth=charts.points(width), alpha=opacity, label=label)
               for _, colour, width, opacity, label in styles]
    figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    for case in range(cases):
        for row, ax in enumerate(panels[:, case]):
            camera = flat_panel(ax, core.ga)
            # The periapsis detail framed, so that it reads as a close-up rather than as the path running on.
            frame = [frame_drawn(boxes[row, case], camera)] if row == 1 else []
            simplices.overlay(ax, [cross_drawn(boxes[row, case], camera)]
                              + [path_drawn(position[:, case], colour, width, camera, opacity)
                                 for position, colour, width, opacity, _ in styles]
                              + centre_drawn(CENTRE_RADIUS, camera) + frame)
    return figure


@charts.style()
def draw_errors(counts: np.ndarray, physical: core.Scalar, regularized: core.Scalar) -> plt.Figure:
    """Physical-time-weighted RMS position errors against the exact orbit."""
    direct = physical.to_array()
    lifted = regularized.to_array()
    # Three quarters of the column for the cases' panels: a few points each on straight lines, and the height the
    # error's axis label needs above the legend.
    figure, panels = charts.subplots(600, 280, ncols=direct.shape[1], squeeze=False, layout="constrained")
    for case, ax in enumerate(panels[0]):
        ax.loglog(counts, direct[:, case], "o-", color=PHYSICAL_COLOUR, label="Physical clock")
        ax.loglog(counts, lifted[:, case], "o-", color=REGULARIZED_COLOUR, label="Spinor clock")
        ax.set_xlabel("Steps per orbit")
        ax.set_xticks(counts, labels=[str(count) for count in counts])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
    panels[0, 0].set_ylabel("RMS error / semimajor axis")
    # The legend under the panels, clear of the falling lines.
    figure.legend(*panels[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    return figure


def animate_lift(positions: core.Point, spinors: core.Spinor) -> Iterator[np.ndarray]:
    """Two orbital revolutions and one closed spinor loop, sampled in the regularized clock."""
    figure, panels = plt.subplots(1, 2, figsize=(4, 1.7), layout="constrained")
    paths = (positions, lifted_points(spinors))
    for ax in panels:
        ax.set_axis_off()
    # Lay the figure out once; only the body moves between frames.
    cameras = [framed(ax, path, MARGIN) for ax, path in zip(panels, paths)]
    figure.canvas.draw()
    figure.set_layout_engine("none")
    backdrops = [[path_drawn(path, REGULARIZED_COLOUR, ANIMATION_WIDTH, camera),
                  *centre_drawn(ANIMATION_CENTRE, camera)] for path, camera in zip(paths, cameras)]
    images = [simplices.overlay(ax, backdrop) for ax, backdrop in zip(panels, backdrops)]
    for index in range(len(positions)):
        for image, backdrop, path, camera in zip(images, backdrops, paths, cameras):
            body = path[index][None]                                           # [1] Point-
            image.set_data(simplices.composited(backdrop + [
                simplices.markers(body, np.ones(3), BODY_RADIUS + RIM, camera),
                simplices.markers(body, np.array(to_rgb(BODY_COLOUR)), BODY_RADIUS, camera)]))
        yield capture(figure)
    plt.close(figure)
