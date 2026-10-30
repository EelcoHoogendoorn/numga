"""Geometric views of a plane wave's curvature and its tidal readout: drawing only.

Coordinates leave the algebra here, at the plotting boundary, as pairings with the coordinate slices.
The map panels depict the curvature applied at one event. The detector panels show weak-wave
displacements magnified for display; acceleration arrows share one fixed scale throughout an
animation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack

from examples.util import charts
from examples.util.animation import capture
from examples.relativity.curvature import core
from examples.relativity.curvature.core import Antivector, Curvature, Scalar, mv
from examples.util.wgsl import labels, show, solids

INK, BLUE, ORANGE = "#263b51", "#267fa9", "#d17c24"
light_colour = "#c3ccd4"
tracking_colours = ("#d15a67", "#b48616", "#168676", "#8759a6")
polarization_names = ("Plus", "Cross", "Circular")
# The animated ring: the circular packet, alone, in a square of so many pixels a side; the still's three rings, in so
# many pixels across and up.
animated_polarization = polarization_names.index("Circular")
ANIMATION_PX = 330
DETECTOR_PX = (780, 260)
# The tracked beads, half again as large as the others.
TRACKED_SCALE = 1.6
# Frames of the animated ring per unit time: at 40 ms a frame, a unit of time takes a second.
frames_per_unit_time = 25
# The spacetime diagrams, drawn on the GPU with t up: the view's elevation and azimuth in radians, azimuth zero
# looking along +z, the margin as a share of the larger side, and the pixels across; the share of the figure's
# width each diagram takes.
ELEVATION, AZIMUTH = np.radians([15.0, 22.0])
MARGIN = 0.08
N_MAP_PIXELS = 250
PANEL_WIDTH = 0.44
BLUE_RGB, ORANGE_RGB, LIGHT, PATCH = (np.array(to_rgb(colour)) for colour in (BLUE, ORANGE, light_colour, BLUE))
# How much each of a patch's two faces hides.
PATCH_OPACITY = 0.15
# The axes' arrows and their names, in extents of the diagram; widths in pixels, of the axes' arrows and of the
# patches' outlines, and an arrowhead's length and width in widths of its shaft.
AXIS_LENGTH, LABEL_DISTANCE = 0.88, 0.97
AXIS_WIDTH, OUTLINE_WIDTH = 1.5, 1.8
HEAD_LENGTH, HEAD_WIDTH = 4.5, 3.0
# The colour of the names of the lines of the faint cross through a ring's centre.
CROSS_LABEL_COLOUR = "#7b8794"


# --- coordinate readout ---------------------------------------------------------------
def plain_axes(ax: plt.Axes, across: str, up: str) -> None:
    """Axes named by what runs along them, the quantity's zero as the horizontal one, without numbers."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set(xticks=[], yticks=[], xlabel=across, ylabel=up)
    # The horizontal axis's name below the plot, wherever its zero lies.
    ax.xaxis.set_label_coords(0.5, -0.03)
    ax.xaxis.label.set_va("top")


def transverse(value: Antivector) -> np.ndarray:
    """(..., 2) x and y of separations in the plane transverse to the wave: their pairings with the
    slices x and y."""
    return (stack((mv.x, mv.y)) & value[..., None]).to_array()


# --- the curvature map at one event ---------------------------------------------------
def diagram_points(directions: Antivector) -> solids.Point:
    """Draw x, z and time along the scene's x, y and z: an explicit diagram, with Euclidean picture distances."""
    mv = directions.context.multivector
    readings = stack((mv.x, mv.z, mv.t)) & directions[..., None]
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)



def patch(first: Antivector, second: Antivector) -> Antivector:
    """The corners of the area element spanned by two directions, around it."""
    return (first * (0.5 * np.array([-1, 1, 1, -1]))
            + second * (0.6 * np.array([-1, -1, 1, 1])))                           # [corners] Antivector-


def arrows(ends: solids.Point, colour: np.ndarray, width: float, view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Arrows from the origin to the diagram's points `[arrows]`, drawn with t up, `width` pixels across: their
    shafts and heads, in one colour."""
    shafts, heads = solids.arrows(solids.origin.broadcast_to(ends.shape), ends, width / 2 * view.pixel,
                                  HEAD_LENGTH * width * view.pixel, HEAD_WIDTH * width / 2 * view.pixel)
    return [(shafts, colour), (heads, colour)]


def panel(ax: plt.Axes, corners: Antivector, ends: list[tuple[Antivector, np.ndarray, float]], extent: float,
          view: solids.View) -> None:
    """One spacetime diagram in its axes: horizontal x, depth z and vertical t as light arrows named at their tips,
    the area element `[4]` as a translucent plate a pixel thick with its outline, rods joined round at the
    corners, and arrows to directions `[arrows]` each with its colour and width in pixels."""
    places = diagram_points(corners)                                           # [corners] Point-
    rims, joints = solids.edges(places, places[np.roll(np.arange(4), -1)], OUTLINE_WIDTH / 2 * view.pixel)
    parts = (arrows(solids.origin + solids.axes.dual() * (extent * AXIS_LENGTH), LIGHT, AXIS_WIDTH, view)
             + [(rims, PATCH), (joints, PATCH)]
             + [part for tips, colour, width in ends for part in arrows(diagram_points(tips), colour, width, view)])
    area = [(solids.plates(places[None], view.pixel), PATCH, PATCH_OPACITY)]
    show(ax, solids.draw(parts, view, np.ones(3), inks=[], translucent=area, meshes=[]), view)
    labels(ax, view, solids.origin + solids.axes.dual() * (extent * LABEL_DISTANCE), ["x", "z", "ct"], ha="center",
           va="center", fontsize=12, color=INK)


def draw_curvature_map(
    curvature: Curvature, observer: Antivector, wave: Antivector, edge: Antivector,
) -> plt.Figure:
    """A spacetime ribbon and its image under curvature.

    The ribbon is the observer's velocity joined with a unit separation. Its image is a null plane
    containing the wave's direction; the observer reads it as a tidal force, which the metric turns
    into an acceleration. The image is drawn as the plane of the wave's direction and that acceleration.
    """
    readout = -(observer & curvature(observer & edge)).dual()           # [] Antivector-
    # The acceleration at unit length: a direction within the observer's slice squares to its squared length.
    across = readout / readout.squared().square_root()                   # [] Antivector-
    patches = (patch(observer, edge), patch(wave, across))               # [corners] Antivector-, Antivector-
    drawn = concatenate((*patches, observer[None], wave[None], readout[None]))   # [drawn] Antivector-
    widths = (solids.axes & diagram_points(drawn)[..., None]).abs().reshape(-1)
    extent = widths.max().clip(1, np.inf) * 1.22
    # One view for both diagrams, framing the axes' names and everything drawn on them.
    names = solids.origin + solids.axes.dual() * (extent * LABEL_DISTANCE)   # [3] Point-
    view = solids.view([concatenate((diagram_points(drawn), names))], 0.0, ELEVATION, AZIMUTH, MARGIN, N_MAP_PIXELS)
    rows, columns = view.shape
    fig = plt.figure(figsize=(9.5, 9.5 * PANEL_WIDTH * rows / columns), dpi=60, facecolor="white")
    before, after = fig.add_axes((0, 0, PANEL_WIDTH, 1)), fig.add_axes((1 - PANEL_WIDTH, 0, PANEL_WIDTH, 1))
    panel(before, patches[0], [(observer[None], ORANGE_RGB, 2.0)], extent, view)
    panel(after, patches[1], [(wave[None], BLUE_RGB, 2.2), (observer[None], ORANGE_RGB, 2.0),
                              (readout[None], ORANGE_RGB, 2.5)], extent, view)
    # The map from the ribbon to its image, an arrow across the gap between the diagrams.
    before.annotate("", xy=(1 - PANEL_WIDTH + 0.01, 0.5), xytext=(PANEL_WIDTH - 0.01, 0.5),
                    xycoords=fig.transFigure, textcoords=fig.transFigure,
                    arrowprops={"arrowstyle": "->", "lw": 1.8, "color": INK})
    return fig


# --- the strain packet and the Doppler check ------------------------------------------
@charts.style()
def draw_packet(time: np.ndarray, strains: np.ndarray, curvatures: np.ndarray) -> plt.Figure:
    """The plus and cross strain profiles and the curvature's weights `[times, 2]`, plus in blue and cross in orange,
    each named beside its highest crest."""
    # About half the column: two short strips of two smooth packets each.
    fig, (top, bottom) = charts.subplots(440, 260, nrows=2, sharex=True, facecolor="white", layout="constrained")
    for ax, values, name, across in ((top, strains, "strain", ""), (bottom, curvatures, "curvature", "time")):
        ax.plot(time, values[:, 0], color=BLUE)
        ax.plot(time, values[:, 1], color=ORANGE)
        plain_axes(ax, across, name)
    crests = strains.argmax(axis=0)                                            # [2]
    for crest, values, name, colour, side in ((crests[0], strains[:, 0], "plus", BLUE, -1),
                                              (crests[1], strains[:, 1], "cross", ORANGE, 1)):
        top.annotate(name, (time[crest], values[crest]), xytext=(side * 10, 0), textcoords="offset points",
                     ha="right" if side < 0 else "left", va="center", color=colour)
    return fig


@charts.style()
def draw_doppler(rapidities: np.ndarray, amplitudes: Scalar) -> plt.Figure:
    """Tidal amplitude measured by observers boosted along the wave, against `np.exp(-2 * rapidities)`."""
    measured = amplitudes.to_array()                                     # [n]
    fine = np.linspace(rapidities.min(), rapidities.max(), 200)
    # Under half the column: one falling curve and its dots.
    fig, ax = charts.subplots(380, 240, facecolor="white", layout="constrained")
    # The Doppler factor squared as a line, the tidal map's largest singular value as dots.
    ax.plot(fine, np.exp(-2 * fine), color=INK)
    ax.scatter(rapidities, measured, color=BLUE, zorder=5)
    # The observer at rest, faint.
    ax.axvline(0, color=light_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set(ylim=(0, None))
    plain_axes(ax, "rapidity", "tidal amplitude")
    return fig


# --- the detector ring ----------------------------------------------------------------
@dataclass(frozen=True)
class _Detector:
    time: np.ndarray            # [n_time]
    reference: np.ndarray       # [n_beads, 2] in initial-radius units
    rest: np.ndarray            # [n_rest_samples, 2] the ring at rest, of unit radius
    positions: np.ndarray       # [n_polarizations, n_time, n_beads, 2]
    accelerations: np.ndarray   # [n_polarizations, n_time, n_beads, 2]
    largest_acceleration: float
    most_deformed: int          # the time of largest deformation, over every polarization


def _detector(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector, amplification: float,
) -> _Detector:
    """The beads in initial-radius units, with displacements magnified for display, read out."""
    # The ring's radius: a separation within the slice of space squares to its squared length.
    radius = reference.squared().square_root().mean()                    # [] Scalar+
    positions = (reference + amplification * displacement) / radius     # [n_time, n_polarizations, n_beads] Antivector-
    accelerations = amplification * acceleration / radius                # [n_time, n_polarizations, n_beads] Antivector-
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, 200) / 2)).exp()        # [n_rest_samples] Rotor+
    return _Detector(
        time=np.asarray(time),
        reference=transverse(reference / radius),
        rest=transverse(turns >> mv.x.dual()),
        positions=np.swapaxes(transverse(positions), 0, 1),
        accelerations=np.swapaxes(transverse(accelerations), 0, 1),
        largest_acceleration=float(accelerations.squared().square_root().to_array().max()),
        most_deformed=int(displacement.squared().sum(axis=(1, 2)).argmax()),
    )


def _ring_axes(fig: plt.Figure, data: _Detector) -> list[plt.Axes]:
    limit = max(1.35, float(np.abs(data.positions).max()) + 0.31)
    axes = []
    for column in range(len(polarization_names)):
        ax = fig.add_subplot(1, 3, column + 1)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
        axes.append(ax)
    return axes


def _ring(ax: plt.Axes, data: _Detector, polarization: int, static: bool):
    reference = data.reference
    selected = np.arange(0, len(reference), max(1, len(reference) // 4))[:4]
    arrows_at = np.arange(0, len(reference), max(1, len(reference) // 8))
    ax.plot(*data.rest.T, color=light_colour, ls=":")
    ax.scatter([0], [0], color=ORANGE, zorder=8)
    # The cross through the centre, across the whole view but for room for its names at its ends.
    half_span = 0.9 * ax.get_xlim()[1]
    ax.plot([-half_span, half_span], [0, 0], color="#edf0f3", lw=charts.FRAME_WIDTH, zorder=0)
    ax.plot([0, 0], [-half_span, half_span], color="#edf0f3", lw=charts.FRAME_WIDTH, zorder=0)
    ax.annotate("x", (half_span, 0), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                color=CROSS_LABEL_COLOUR)
    ax.annotate("y", (0, half_span), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                color=CROSS_LABEL_COLOUR)
    outline, = ax.plot([], [], color=BLUE, alpha=0.42, zorder=2)
    beads = ax.scatter([], [], color=BLUE, edgecolors="white", linewidths=charts.FRAME_WIDTH, zorder=5)
    markers = ax.scatter(*reference[selected].T, s=charts.points(TRACKED_SCALE * charts.MARKER_PX) ** 2,
                         c=tracking_colours[:len(selected)], edgecolors="white", linewidths=charts.FRAME_WIDTH, zorder=7)
    trails = [ax.plot([], [], color=colour, alpha=0.60, zorder=4)[0]
              for colour in tracking_colours[:len(selected)]]
    quiver = ax.quiver(*reference[arrows_at].T, np.zeros(len(arrows_at)), np.zeros(len(arrows_at)),
                       color=BLUE, alpha=0.85, angles="xy", scale_units="xy",
                       scale=max(data.largest_acceleration, 1e-15) / 0.31, width=0.006, headwidth=3.5, headlength=4, zorder=6)
    if static:
        for index in selected:
            ax.plot(*data.positions[polarization, :, index].T,
                    color=tracking_colours[list(selected).index(index)], lw=charts.FRAME_WIDTH, alpha=0.22)

    def update(frame: int):
        positions = data.positions[polarization, frame]
        outline.set_data(*np.vstack((positions, positions[0])).T)
        beads.set_offsets(positions)
        markers.set_offsets(positions[selected])
        quiver.set_offsets(positions[arrows_at])
        quiver.set_UVC(*data.accelerations[polarization, frame, arrows_at].T)
        first = max(0, frame - max(2, len(data.time) // 7))
        for index, trail in zip(selected, trails):
            trail.set_data(*data.positions[polarization, first:frame + 1, index].T)
        return [outline, beads, markers, quiver, *trails]

    return update


@charts.style()
def draw_detector(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector,
    amplification: float,
) -> plt.Figure:
    """The three polarizations at one shared instant of largest deformation, with faint full trails."""
    data = _detector(time, reference, displacement, acceleration, amplification)
    fig = plt.figure(figsize=(DETECTOR_PX[0] / charts.DPI, DETECTOR_PX[1] / charts.DPI), facecolor="white")
    axes = _ring_axes(fig, data)
    # The rings filling the figure side by side, each a square of its height.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    for polarization, ax in enumerate(axes):
        _ring(ax, data, polarization, static=True)(data.most_deformed)
    return fig


def animate_detector(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector, amplification: float,
) -> Iterator[np.ndarray]:
    """The bead ring through the circular packet, as frames at `frames_per_unit_time`."""
    data = _detector(time, reference, displacement, acceleration, amplification)
    with charts.style():
        fig = plt.figure(figsize=(ANIMATION_PX / charts.DPI, ANIMATION_PX / charts.DPI), facecolor="white")
        limit = max(1.35, float(np.abs(data.positions).max()) + 0.31)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
        update = _ring(ax, data, animated_polarization, static=False)
        count = min(len(data.time), max(2, round((data.time[-1] - data.time[0]) * frames_per_unit_time) + 1))
        for index in np.linspace(0, len(data.time) - 1, count, dtype=int):
            update(index)
            yield capture(fig)
        plt.close(fig)
