"""Planar cameras and log-log charts render Kepler orbital passages, their spinor lifts, and integration errors."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from numga.topology import Topology
from numga.render import Camera, drawing, simplices
from examples.util.panels import offset_panel, overlay, room
from examples.mechanics.kepler import core

exact_colour = "#626c76"
physical_colour = "#d06c36"
regularized_colour = "#287fa0"
body_colour = "#ff00b7"
centre_colour = "#e5ab32"
cross_colour, label_colour = "0.85", "0.4"
# Margin around a lifted path and around the animation paths, as a share of path extent.
label_room = 0.12
margin = 0.05
# Room beyond the periapsis detail box for its frame, as a share of the box half-width.
frame_room = 0.02
# Marker radii and stroke widths in pixels for the static charts and the animation.
centre_radius = charts.MARKER_PX
rim_width = charts.LINE_PX / 2
animation_width, animation_centre_radius, body_radius = 1.4, 3.2, 3.8
exact_opacity = 0.5


# --- plumbing -------------------------------------------------------------------------
def plane_readings(points: core.Point) -> core.Scalar:
    """Pairing `points` with the basis planes reads horizontal and vertical coordinates (`[2, ...] Scalar+`)."""
    planes = stack((core.mv.x, core.mv.y))                                    # [2] Plane+
    return planes.reshape((2,) + (1,) * points.ndim) & points


def spinor_points(spinors: core.Spinor) -> core.Point:
    """Right-multiplying the unit x-point by `spinors` maps scalar and xy spinor components into the orbital plane (`[...] Point-`)."""
    return core.mv.x.dual() * spinors


def panel_camera(ax: plt.Axes, path: core.Point, panel_margin: float) -> Camera:
    """Equal-scale bounds around `path` and the attracting origin frame `ax` with `panel_margin` padding and configure its planar `Camera`."""
    # Stretch the path extent to include the attracting centre at the origin, where both readings vanish.
    readings = plane_readings(path)                                           # [2, ...] Scalar+
    lower_corner = readings.min(axis=-1).clip(-np.inf, 0).to_array()
    upper_corner = readings.max(axis=-1).clip(0, np.inf).to_array()
    spans = (1 + 2 * panel_margin) * (upper_corner - lower_corner)
    panel_shape = np.array(room(ax))
    # Fit both coordinate spans into the panel's pixel room at equal horizontal and vertical scales.
    half_span = panel_shape * (spans / panel_shape).max() / 2
    box_centre = (lower_corner + upper_corner) / 2
    ax.set(xlim=(box_centre[0] - half_span[0], box_centre[0] + half_span[0]),
           ylim=(box_centre[1] - half_span[1], box_centre[1] + half_span[1]),
           aspect="equal")
    return offset_panel(ax, core.mv.x, core.mv.y)


def centre_markers(radius: float, camera: Camera) -> list[np.ndarray]:
    """Concentric disc layers draw the white-rimmed attracting centre marker at the origin."""
    origin = (0 * core.mv.x.dual())[None]                                     # [1] Point-
    return [simplices.markers(origin, np.ones(3), radius + rim_width, camera),
            simplices.markers(origin, np.array(to_rgb(centre_colour)), radius, camera)]


def origin_labels(ax: plt.Axes, horizontal_label: str, vertical_label: str) -> None:
    """Axis annotations mark the visible ends of the origin crosshairs while hiding the matplotlib frame."""
    ax.text(1, 0, horizontal_label, transform=ax.get_yaxis_transform(),
            ha="right", va="top", color=label_colour, clip_on=True)
    ax.text(0, 1, f" {vertical_label}", transform=ax.get_xaxis_transform(),
            ha="left", va="top", color=label_colour, clip_on=True)
    ax.set_axis_off()


def planar_points(x_coordinates: np.ndarray, y_coordinates: np.ndarray) -> core.Point:
    """Dualized basis combinations construct planar `Point-` extensors from `x_coordinates` and `y_coordinates` arrays."""
    return (core.mv.x * x_coordinates + core.mv.y * y_coordinates).dual()


def origin_cross(box: tuple[float, float, float, float], camera: Camera) -> np.ndarray:
    """Two orthogonal strokes render thin origin crosshairs spanning `box` in `camera`."""
    left_edge, right_edge, bottom_edge, top_edge = box
    endpoints = planar_points(
        np.array([left_edge, right_edge, 0.0, 0.0]),
        np.array([0.0, 0.0, bottom_edge, top_edge]),
    )                                                                         # [4] Point-
    return simplices.strokes(endpoints, Topology.chains([2, 2]), np.array(to_rgb(cross_colour)), charts.FRAME_PX,
                             camera, cap=simplices.Cap.BUTT)


def box_frame(box: tuple[float, float, float, float], camera: Camera) -> np.ndarray:
    """A closed four-segment loop renders a thin rectangular border around `box` in `camera`."""
    left_edge, right_edge, bottom_edge, top_edge = box
    corners = planar_points(
        np.array([left_edge, right_edge, right_edge, left_edge]),
        np.array([bottom_edge, bottom_edge, top_edge, top_edge]),
    )                                                                         # [4] Point-
    return simplices.strokes(corners, Topology.loops([4]), np.array(to_rgb(cross_colour)), charts.FRAME_PX,
                             camera)


def path_stroke(path: core.Point, colour: str, width: float, opacity: float, camera: Camera) -> np.ndarray:
    """A connected polyline stroke renders `path` (`[...] Point-`) in `camera`."""
    return simplices.strokes(path, Topology.chains([len(path)]), np.array(to_rgb(colour)), width, camera,
                             opacity=opacity)


@charts.style()
def lift_figure(positions: core.Point, spinors: core.Spinor) -> plt.Figure:
    """Side-by-side planar panels show one eccentric orbital passage alongside the half-turn of its lifted position spinor."""
    # Three fifths of the column fits both paths side by side with clear separation between sample dots.
    figure, panels = charts.subplots(480, 200, ncols=2, layout="constrained")
    axis_labels = (("x", "y"), ("scalar", "xy"))
    for ax, path, (horizontal_label, vertical_label) in zip(panels, (positions, spinor_points(spinors)), axis_labels):
        camera = panel_camera(ax, path, label_room)
        origin_labels(ax, horizontal_label, vertical_label)
        sample_colours = plt.get_cmap("viridis")(np.linspace(0, 1, len(path)))[:, :3]
        overlay(ax, [origin_cross(camera.extent, camera),
                     path_stroke(path, regularized_colour, charts.LINE_PX, 1.0, camera),
                     simplices.markers(path, sample_colours, charts.MARKER_PX / 2, camera),
                     *centre_markers(centre_radius, camera)])
    return figure


@charts.style()
def passage_figure(reference_positions: core.Point, physical_positions: core.Point,
                   regularized_positions: core.Point) -> plt.Figure:
    """Two-row orbital panels show full ellipses and magnified periapsis passages for physical-clock and spinor-clock Verlet steps."""
    n_samples, n_cases = reference_positions.shape
    exact_readings = plane_readings(reference_positions)                      # [2, n_samples, n_cases] Scalar+
    lower_bounds = exact_readings.min(axis=1).to_array()
    upper_bounds = exact_readings.max(axis=1).to_array()
    # Three quarters of the column fits a full path and its periapsis detail per case above the shared legend.
    figure, panels = charts.subplots(600, 400, nrows=2, ncols=n_cases, squeeze=False, layout="constrained")
    styles = [(reference_positions, exact_colour, 2 * charts.LINE_PX, exact_opacity, "Exact"),
              (physical_positions, physical_colour, charts.LINE_PX, 1.0, "Physical clock"),
              (regularized_positions, regularized_colour, charts.LINE_PX, 1.0, "Spinor clock")]
    boxes = np.empty((2, n_cases, 4))
    for case_index in range(n_cases):
        lower_corner, upper_corner = lower_bounds[:, case_index], upper_bounds[:, case_index]
        span = np.max(upper_corner - lower_corner)
        box_centre = (lower_corner + upper_corner) / 2
        half_width = 0.12 * span
        boxes[0, case_index] = (box_centre[0] - 0.6 * span, box_centre[0] + 0.6 * span,
                                box_centre[1] - 0.6 * span, box_centre[1] + 0.6 * span)
        boxes[1, case_index] = (lower_corner[0] - half_width, lower_corner[0] + half_width, -half_width, half_width)
        for row_index in range(2):
            ax = panels[row_index, case_index]
            left_edge, right_edge, bottom_edge, top_edge = boxes[row_index, case_index]
            # Widen the periapsis detail slightly beyond its box so its border frame remains visible.
            extra_room = (frame_room if row_index == 1 else 0.0) * (right_edge - left_edge) / 2
            ax.set(xlim=(left_edge - extra_room, right_edge + extra_room),
                   ylim=(bottom_edge - extra_room, top_edge + extra_room),
                   aspect="equal")
            origin_labels(ax, "x", "y")
    figure.get_layout_engine().set(hspace=0.08)
    handles = [Line2D([], [], color=colour, linewidth=charts.points(width), alpha=opacity, label=label)
               for _, colour, width, opacity, label in styles]
    figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    for case_index in range(n_cases):
        for row_index, ax in enumerate(panels[:, case_index]):
            camera = offset_panel(ax, core.mv.x, core.mv.y)
            border = [box_frame(boxes[row_index, case_index], camera)] if row_index == 1 else []
            overlay(ax, [origin_cross(boxes[row_index, case_index], camera)]
                        + [path_stroke(positions[:, case_index], colour, width, opacity, camera)
                           for positions, colour, width, opacity, _ in styles]
                        + centre_markers(centre_radius, camera) + border)
    return figure


@charts.style()
def convergence_figure(step_counts: np.ndarray, physical_errors: core.Scalar,
                       regularized_errors: core.Scalar) -> plt.Figure:
    """Log-log panels plot physical-time-weighted position errors against step count for each eccentricity."""
    physical_values = physical_errors.to_array()
    regularized_values = regularized_errors.to_array()
    figure, panels = charts.subplots(600, 280, ncols=physical_values.shape[1], squeeze=False, layout="constrained")
    for case_index, ax in enumerate(panels[0]):
        ax.loglog(step_counts, physical_values[:, case_index], "o-", color=physical_colour, label="Physical clock")
        ax.loglog(step_counts, regularized_values[:, case_index], "o-", color=regularized_colour, label="Spinor clock")
        ax.set_xlabel("Steps per orbit")
        ax.set_xticks(step_counts, labels=[str(count) for count in step_counts])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
    panels[0, 0].set_ylabel("RMS error / semimajor axis")
    figure.legend(*panels[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    return figure


def lift_frames(states: Iterable[tuple[core.Point, core.Spinor]]) -> Iterator[np.ndarray]:
    """Yields RGB frames showing two physical revolutions alongside one closed spinor cycle in the regularized clock."""
    state_list = list(states)
    n_samples = len(state_list)
    positions = stack([position for position, _ in state_list])               # [n_samples] Point-
    spinors = stack([spinor for _, spinor in state_list])                     # [n_samples] Spinor+
    figure, panels = plt.subplots(1, 2, figsize=(4, 1.7), layout="constrained")
    paths = (positions, spinor_points(spinors))
    for ax in panels:
        ax.set_axis_off()
    # Lock the layout once so only the body markers move across frames.
    cameras = [panel_camera(ax, path, margin) for ax, path in zip(panels, paths)]
    figure.canvas.draw()
    figure.set_layout_engine("none")
    backdrops = [[path_stroke(path, regularized_colour, animation_width, 1.0, camera),
                  *centre_markers(animation_centre_radius, camera)] for path, camera in zip(paths, cameras)]
    images = [overlay(ax, backdrop) for ax, backdrop in zip(panels, backdrops)]
    for frame_index in range(n_samples):
        for image, backdrop, path, camera in zip(images, backdrops, paths, cameras):
            body_point = path[frame_index][None]                              # [1] Point-
            image.set_data(drawing.composited(backdrop + [
                simplices.markers(body_point, np.ones(3), body_radius + rim_width, camera),
                simplices.markers(body_point, np.array(to_rgb(body_colour)), body_radius, camera)]))
        yield capture(figure)
    plt.close(figure)
