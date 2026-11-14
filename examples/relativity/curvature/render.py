"""Geometric views of plane-wave spacetime curvature and its tidal readout on a ring of beads.

Coordinates leave the algebra at the plotting boundary through pairings with the coordinate slices.
Two 3D spacetime diagrams show how curvature maps an observer's ribbon onto a null plane, while 2D
panels display magnified weak-wave bead displacements and fixed-scale tidal acceleration arrows.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA2D
from numga.render import drawing, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology

from examples.relativity.curvature.core import Antivector, Curvature, Scalar, mv
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import labels, offset_panel, overlay, show

ink_colour, blue_colour, orange_colour = "#263b51", "#267fa9", "#d17c24"
light_colour = "#c3ccd4"
tracking_colours = ("#d15a67", "#b48616", "#168676", "#8759a6")
polarization_names = ("Plus", "Cross", "Circular")
animated_polarization = polarization_names.index("Circular")
animation_px = 330
detector_px = (780, 260)
tracked_scale = 1.6
frames_per_unit_time = 25

elevation, azimuth = np.radians([15.0, 22.0])
view_margin = 0.08
n_map_pixels = 250
panel_width = 0.44
blue_rgb, orange_rgb, light_rgb, patch_rgb = (
    np.array(to_rgb(colour)) for colour in (blue_colour, orange_colour, light_colour, blue_colour)
)
patch_opacity = 0.15
axis_length, label_distance = 0.88, 0.97
axis_width, outline_width = 1.5, 1.8
chevron_scale = 4.0

cross_label_colour = "#7b8794"
cross_rgb = np.array(to_rgb("#edf0f3"))
n_rest_samples = 199
rest_dots = (1.5, 2.5)
arrow_reach, arrow_width = 0.31, 2.0

page = NumpyContext(VGA2D).multivector
map_arrow_width = 1.5


# --- coordinate readout ---------------------------------------------------------------
def plain_axes(ax: plt.Axes, x_label: str, y_label: str) -> None:
    """Horizontal and vertical axis labels sit on numberless spines with the horizontal spine at zero."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set(xticks=[], yticks=[], xlabel=x_label, ylabel=y_label)
    ax.xaxis.set_label_coords(0.5, -0.03)
    ax.xaxis.label.set_va("top")


# --- the curvature map at one event ---------------------------------------------------
def diagram_points(directions: Antivector) -> solids.Point:
    """Spacetime directions map x, z, and ct onto the three Euclidean axes of the 3D diagram."""
    local_mv = directions.context.multivector
    readings = stack((local_mv.x, local_mv.z, local_mv.t)) & directions[..., None]   # [..., 3] Scalar+
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)   # [...] Point-


def area_corners(first: Antivector, second: Antivector) -> Antivector:
    """Four corners around the origin outline the spacetime patch spanned by `first` and `second` (`[4] Antivector-`)."""
    return (first * (0.5 * np.array([-1, 1, 1, -1]))
            + second * (0.6 * np.array([-1, -1, 1, 1])))                             # [4] Antivector-


def diagram_arrows(ends: solids.Point, colour: np.ndarray, width: float, view: solids.View) -> tuple[Draw, float]:
    """Chevron-tipped 3D tubes connect the origin to `ends` in `colour` at `width` pixels (`[n_arrows]`)."""
    n_arrows = ends.shape[0]
    shafts = stack((solids.origin.broadcast_to(ends.shape), ends), axis=-1).reshape(-1)   # [n_arrows * 2] Point-
    return (simplices.tubes(shafts, Topology.chains([2] * n_arrows), colour, width, view, gloss=solids.gloss,
                            chevrons=np.ones(n_arrows), chevron=chevron_scale * width), 1.0)


def spacetime_panel(
    ax: plt.Axes,
    corners: Antivector,
    arrow_groups: list[tuple[Antivector, np.ndarray, float]],
    extent: float,
    view: solids.View,
) -> None:
    """One 3D spacetime diagram renders the coordinate axes, a translucent four-corner patch, and direction arrows."""
    places = diagram_points(corners)                                                 # [4] Point-
    outline = (simplices.tubes(places, Topology.loops([4]), patch_rgb, outline_width, view, gloss=solids.gloss), 1.0)
    axis_tips = solids.origin + solids.axes.dual() * (extent * axis_length)          # [3] Point-
    drawn_tubes = ([outline, diagram_arrows(axis_tips, light_rgb, axis_width, view)]
                   + [diagram_arrows(diagram_points(tips), colour, width, view) for tips, colour, width in arrow_groups])
    area_plate = [(solids.plates(places[None], view.pixel), patch_rgb, patch_opacity)]
    show(ax, solids.draw([], view, np.ones(3), inks=[], translucent=area_plate, meshes=drawn_tubes), view)
    label_points = solids.origin + solids.axes.dual() * (extent * label_distance)    # [3] Point-
    labels(ax, view, label_points, ["x", "z", "ct"], ha="center", va="center", fontsize=12, color=ink_colour)


def curvature_map_figure(
    curvature: Curvature, observer: Antivector, wave: Antivector, edge: Antivector,
) -> plt.Figure:
    """Two linked spacetime diagrams depict an observer's ribbon and its null image under `curvature`."""
    readout = -(curvature(observer & edge) & observer).dual()                        # [] Antivector-
    unit_acceleration = readout / readout.antinorm()                                 # [] Antivector-
    first_patch = area_corners(observer, edge)                                       # [4] Antivector-
    second_patch = area_corners(wave, unit_acceleration)                             # [4] Antivector-
    drawn_directions = concatenate((first_patch, second_patch, observer[None], wave[None], readout[None]))   # [11] Antivector-
    widths = (solids.axes & diagram_points(drawn_directions)[..., None]).abs().reshape(-1)
    extent = widths.max().clip(1, np.inf) * 1.22
    # Both diagrams share one view framing the axis labels and every drawn direction.
    label_points = solids.origin + solids.axes.dual() * (extent * label_distance)    # [3] Point-
    view = solids.view([concatenate((diagram_points(drawn_directions), label_points))],
                       0.0, elevation, azimuth, view_margin, n_map_pixels)
    rows, columns = view.shape
    fig = plt.figure(figsize=(9.5, 9.5 * panel_width * rows / columns), dpi=60, facecolor="white")
    before_axes = fig.add_axes((0, 0, panel_width, 1))
    after_axes = fig.add_axes((1 - panel_width, 0, panel_width, 1))
    spacetime_panel(before_axes, first_patch, [(observer[None], orange_rgb, 2.0)], extent, view)
    spacetime_panel(after_axes, second_patch, [(wave[None], blue_rgb, 2.2), (observer[None], orange_rgb, 2.0),
                                               (readout[None], orange_rgb, 2.5)], extent, view)
    width, height = fig.bbox.width, fig.bbox.height
    connector_axes = fig.add_axes((0, 0, 1, 1))
    connector_axes.set(xlim=(0, width), ylim=(0, height))
    connector_axes.set_axis_off()
    ends = (page.x.dual() * (width * np.array([panel_width - 0.01, 1 - panel_width + 0.01]))
            + page.y.dual() * (height / 2))                                          # [2] Antivector-
    overlay(connector_axes, [simplices.strokes(ends, Topology.chains([2]), np.array(to_rgb(ink_colour)),
                                               map_arrow_width, offset_panel(connector_axes, page.x, page.y),
                                               chevrons=np.ones(1), chevron=chevron_scale * map_arrow_width,
                                               cap=simplices.Cap.BUTT)])
    return fig


# --- the strain packet and the Doppler check ------------------------------------------
@charts.style()
def packet_figure(time: np.ndarray, strains: np.ndarray, curvatures: np.ndarray) -> plt.Figure:
    """Two stacked time-series panels plot the plus and cross strain profiles above their curvature weights."""
    fig, (top, bottom) = charts.subplots(440, 260, nrows=2, sharex=True, facecolor="white", layout="constrained")
    for ax, values, y_label, x_label in ((top, strains, "strain", ""), (bottom, curvatures, "curvature", "time")):
        ax.plot(time, values[:, 0], color=blue_colour)
        ax.plot(time, values[:, 1], color=orange_colour)
        plain_axes(ax, x_label, y_label)
    crests = strains.argmax(axis=0)                                                  # [2]
    for crest, values, name, colour, side in ((crests[0], strains[:, 0], "plus", blue_colour, -1),
                                              (crests[1], strains[:, 1], "cross", orange_colour, 1)):
        top.annotate(name, (time[crest], values[crest]), xytext=(side * 10, 0), textcoords="offset points",
                     ha="right" if side < 0 else "left", va="center", color=colour)
    return fig


@charts.style()
def doppler_figure(rapidities: np.ndarray, amplitudes: Scalar) -> plt.Figure:
    """Measured tidal singular values across boosted observers sit on the exponential Doppler curve."""
    n_observers = len(rapidities)
    measured = amplitudes.to_array()                                                 # [n_observers]
    fine_rapidities = np.linspace(rapidities.min(), rapidities.max(), 200)
    fig, ax = charts.subplots(380, 240, facecolor="white", layout="constrained")
    ax.plot(fine_rapidities, np.exp(-2 * fine_rapidities), color=ink_colour)
    ax.scatter(rapidities, measured, color=blue_colour, zorder=5)
    ax.axvline(0, color=light_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set(ylim=(0, None))
    plain_axes(ax, "rapidity", "tidal amplitude")
    return fig


# --- the detector ring ----------------------------------------------------------------
def detector_layout(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector, amplification: float,
) -> tuple[np.ndarray, Antivector, Antivector, Antivector, float, int]:
    """Normalized bead positions, fixed-scale tidal arrow tips, view half-width, and peak frame index."""
    n_times, n_polarizations, n_beads = displacement.shape
    radius = reference.antinorm().mean()                                             # [] Scalar+
    unit_reference = reference / radius                                              # [n_beads] Antivector-
    positions = (reference + amplification * displacement) / radius                  # [n_times, n_polarizations, n_beads] Antivector-
    accelerations = amplification * acceleration / radius                            # [n_times, n_polarizations, n_beads] Antivector-
    largest = accelerations.antinorm().max()                                         # [] Scalar+
    farthest = (stack((mv.x, mv.y)) & positions[..., None]).abs().max()              # [] Scalar+
    tips = positions + accelerations * (arrow_reach / largest)                       # [n_times, n_polarizations, n_beads] Antivector-
    limit = max(1.35, float(farthest.to_array()) + arrow_reach)
    most_deformed = int(displacement.squared().sum(axis=(1, 2)).argmax())
    return np.asarray(time), unit_reference, positions, tips, limit, most_deformed


def ring_axes(fig: plt.Figure, limit: float) -> list[plt.Axes]:
    """Three equal-aspect axes frame the plus, cross, and circular bead rings side by side."""
    axes = []
    for column in range(len(polarization_names)):
        ax = fig.add_subplot(1, 3, column + 1)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
        axes.append(ax)
    return axes


def ring_updater(
    ax: plt.Axes,
    time: np.ndarray,
    reference: Antivector,
    positions: Antivector,
    tips: Antivector,
    polarization: int,
    show_full_trails: bool,
):
    """Per-frame compositor draws the dotted rest circle, tracked bead trails, beads, and tidal acceleration arrows."""
    n_times = len(time)
    n_beads = reference.shape[0]
    selected = np.arange(0, n_beads, max(1, n_beads // 4))[:4]
    n_tracked = len(selected)
    arrows_at = np.arange(0, n_beads, max(1, n_beads // 8))
    n_arrows = len(arrows_at)
    trail_colours = np.array([to_rgb(colour) for colour in tracking_colours[:n_tracked]])
    camera = offset_panel(ax, mv.x, mv.y)
    half_span = 0.9 * ax.get_xlim()[1]
    cross_ends = half_span * stack((-mv.x.dual(), mv.x.dual(), -mv.y.dual(), mv.y.dual()))   # [4] Antivector-
    ax.annotate("x", (half_span, 0), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                color=cross_label_colour)
    ax.annotate("y", (0, half_span), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                color=cross_label_colour)
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rest_samples, endpoint=False) / 2)).exp()   # [n_rest_samples] Rotor+
    rest = turns >> mv.x.dual()                                                      # [n_rest_samples] Antivector-
    beneath = [
        simplices.strokes(cross_ends, Topology.chains([2, 2]), cross_rgb, charts.FRAME_PX, camera,
                          cap=simplices.Cap.BUTT),
        simplices.strokes(rest, Topology.loops([n_rest_samples]), light_rgb, charts.LINE_PX, camera,
                          dashes=rest_dots)]
    if show_full_trails and n_times > 1:
        tracks = positions[:, polarization, selected].reshape(-1)                    # [n_times * n_tracked] Antivector-
        beneath.append(simplices.strokes(
            tracks, Topology(len(tracks), np.stack((np.arange(len(tracks) - n_tracked),
                                                    np.arange(n_tracked, len(tracks))), axis=-1)),
            np.tile(trail_colours, (n_times - 1, 1)), charts.FRAME_PX, camera, opacity=0.22))
    centre = 0 * mv.x.dual()                                                         # [] Antivector-
    above = [simplices.markers(centre[None], orange_rgb, charts.MARKER_PX / 2, camera)]
    shown = overlay(ax, beneath + above)
    n_trail_steps = max(2, n_times // 7)

    def update(frame: int) -> None:
        frame_positions = positions[frame, polarization]                             # [n_beads] Antivector-
        first = max(0, frame - n_trail_steps)
        n_window = frame + 1 - first
        trails = positions[np.arange(first, frame + 1)[None, :], polarization, selected[:, None]]   # [n_tracked, n_window] Antivector-
        arrow_segments = stack((frame_positions[arrows_at], tips[frame, polarization, arrows_at]), axis=-1)   # [n_arrows, 2] Antivector-
        layers = [
            simplices.strokes(frame_positions, Topology.loops([n_beads]), blue_rgb, charts.LINE_PX, camera,
                              opacity=0.42),
        ]
        if n_window > 1:
            layers.append(simplices.strokes(
                trails.reshape(-1), Topology.chains([n_window] * n_tracked),
                np.repeat(trail_colours, n_window - 1, axis=0), charts.LINE_PX, camera, opacity=0.60))
        layers.extend([
            simplices.markers(frame_positions, np.ones(3), charts.MARKER_PX / 2 + charts.FRAME_PX, camera),
            simplices.markers(frame_positions, blue_rgb, charts.MARKER_PX / 2, camera),
            simplices.strokes(arrow_segments.reshape(-1), Topology.chains([2] * n_arrows), blue_rgb,
                              arrow_width, camera, chevrons=np.ones(n_arrows), chevron=chevron_scale * arrow_width,
                              opacity=0.85, cap=simplices.Cap.BUTT),
            simplices.markers(frame_positions[selected], np.ones(3), tracked_scale * charts.MARKER_PX / 2 + charts.FRAME_PX,
                              camera),
            simplices.markers(frame_positions[selected], trail_colours, tracked_scale * charts.MARKER_PX / 2, camera),
        ])
        shown.set_data(drawing.composited(beneath + layers + above))

    return update


@charts.style()
def detector_figure(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector,
    amplification: float,
) -> plt.Figure:
    """Three side-by-side panels show the plus, cross, and circular bead rings at peak deformation."""
    sample_times, unit_reference, positions, tips, limit, most_deformed = detector_layout(
        time, reference, displacement, acceleration, amplification)
    fig = plt.figure(figsize=(detector_px[0] / charts.DPI, detector_px[1] / charts.DPI), facecolor="white")
    axes = ring_axes(fig, limit)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    for polarization, ax in enumerate(axes):
        ring_updater(ax, sample_times, unit_reference, positions, tips, polarization, show_full_trails=True)(most_deformed)
    return fig


def detector_animation(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector, amplification: float,
) -> Iterator[np.ndarray]:
    """Yields RGB frames of the bead ring responding to the circularly polarized wave packet."""
    sample_times, unit_reference, positions, tips, limit, _ = detector_layout(
        time, reference, displacement, acceleration, amplification)
    with charts.style():
        fig = plt.figure(figsize=(animation_px / charts.DPI, animation_px / charts.DPI), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
        update = ring_updater(ax, sample_times, unit_reference, positions, tips, animated_polarization,
                              show_full_trails=False)
        n_frames = min(len(sample_times), max(2, round((sample_times[-1] - sample_times[0]) * frames_per_unit_time) + 1))
        for index in np.linspace(0, len(sample_times) - 1, n_frames, dtype=int):
            update(index)
            yield capture(fig)
        plt.close(fig)
