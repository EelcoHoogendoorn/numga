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

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA2D

from examples.util import charts
from examples.util.animation import capture
from examples.relativity.curvature import core
from examples.relativity.curvature.core import Antivector, Curvature, Scalar, mv
from examples.util.wgsl import labels, show, simplices, solids
from examples.util.wgsl.camera import flat_panel
from examples.util.wgsl.meshes import Samples

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
# patches' outlines; and the length of the strokes of the chevron at an arrow's tip, in widths of the arrow, in the
# diagrams, on the rings and between the diagrams alike.
AXIS_LENGTH, LABEL_DISTANCE = 0.88, 0.97
AXIS_WIDTH, OUTLINE_WIDTH = 1.5, 1.8
CHEVRON = 4.0
# The colour of the names of the lines of the faint cross through a ring's centre, and of the cross.
CROSS_LABEL_COLOUR = "#7b8794"
CROSS = np.array(to_rgb("#edf0f3"))
# The ring at rest, its samples and its dots, a dash and a gap in pixels; the accelerations' arrows: the largest's
# length in radii of the ring, and their width in pixels.
n_rest_samples = 200
REST_DOTS = (1.5, 2.5)
ARROW_REACH, ARROW_WIDTH = 0.31, 2.0
# The page the map's arrow between the two diagrams is drawn on, in the figure's pixels, its points antivectors; the
# arrow's width in pixels.
page = NumpyContext(VGA2D).multivector
MAP_ARROW_WIDTH = 1.5


# --- coordinate readout ---------------------------------------------------------------
def plain_axes(ax: plt.Axes, across: str, up: str) -> None:
    """Axes named by what runs along them, the quantity's zero as the horizontal one, without numbers."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set(xticks=[], yticks=[], xlabel=across, ylabel=up)
    # The horizontal axis's name below the plot, wherever its zero lies.
    ax.xaxis.set_label_coords(0.5, -0.03)
    ax.xaxis.label.set_va("top")


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


def arrows(ends: solids.Point, colour: np.ndarray, width: float, view: solids.View) -> tuple[Samples, float]:
    """Arrows from the origin to the diagram's points `[arrows]`, drawn with t up, tubes `width` pixels across, each
    ending in a chevron, in one colour: a mesh for `solids.draw` to compose among the solids, and its opacity."""
    n_arrows = ends.shape[0]
    shafts = stack((solids.origin.broadcast_to(ends.shape), ends), axis=-1).reshape(-1)   # [arrows * 2] Point-
    return (simplices.tubes(shafts, simplices.Topology.chains([2] * n_arrows), colour, width, view, gloss=solids.gloss,
                            chevrons=np.ones(n_arrows), chevron=CHEVRON * width), 1.0)


def panel(ax: plt.Axes, corners: Antivector, ends: list[tuple[Antivector, np.ndarray, float]], extent: float,
          view: solids.View) -> None:
    """One spacetime diagram in its axes: horizontal x, depth z and vertical t as light arrows named at their tips,
    the area element `[4]` as a translucent plate a pixel thick with its outline, rods joined round at the
    corners, and arrows to directions `[arrows]` each with its colour and width in pixels."""
    places = diagram_points(corners)                                           # [corners] Point-
    tails, heads = (simplices.Topology.loops([4]).incidence[1, 0] * places.field()).batch()   # [corners] Point-, Point-
    rims, joints = solids.edges(tails, heads, OUTLINE_WIDTH / 2 * view.pixel)
    shown = ([arrows(solids.origin + solids.axes.dual() * (extent * AXIS_LENGTH), LIGHT, AXIS_WIDTH, view)]
             + [arrows(diagram_points(tips), colour, width, view) for tips, colour, width in ends])
    area = [(solids.plates(places[None], view.pixel), PATCH, PATCH_OPACITY)]
    show(ax, solids.draw([(rims, PATCH), (joints, PATCH)], view, np.ones(3), inks=[], translucent=area, meshes=shown),
         view)
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
    across = readout / readout.squared().square_root()                  # [] Antivector-
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
    # The map from the ribbon to its image, an arrow across the gap between the diagrams, on a page spanning the
    # figure, in its pixels.
    width, height = fig.bbox.width, fig.bbox.height
    across = fig.add_axes((0, 0, 1, 1))
    across.set(xlim=(0, width), ylim=(0, height))
    across.set_axis_off()
    ends = (page.x.dual() * (width * np.array([PANEL_WIDTH - 0.01, 1 - PANEL_WIDTH + 0.01]))
            + page.y.dual() * (height / 2))                                    # [2] Antivector-
    simplices.overlay(across, [simplices.strokes(ends, simplices.Topology.chains([2]), np.array(to_rgb(INK)),
                                                 MAP_ARROW_WIDTH, flat_panel(across, VGA2D), chevrons=np.ones(1),
                                                 chevron=CHEVRON * MAP_ARROW_WIDTH, cap=simplices.Cap.BUTT)])
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
    reference: Antivector       # [n_beads] in initial-radius units
    positions: Antivector       # [n_time, n_polarizations, n_beads] in initial-radius units
    tips: Antivector            # [n_time, n_polarizations, n_beads] the accelerations' arrows' tips, at one fixed scale
    limit: float                # the view's half width, past the farthest bead and its arrow
    most_deformed: int          # the time of largest deformation, over every polarization


def _detector(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector, amplification: float,
) -> _Detector:
    """The beads in initial-radius units, with displacements magnified for display, and the arrows of their
    accelerations, the largest a share `ARROW_REACH` of the radius long."""
    # The ring's radius: a separation within the slice of space squares to its squared length.
    radius = reference.squared().square_root().mean()                    # [] Scalar+
    positions = (reference + amplification * displacement) / radius     # [n_time, n_polarizations, n_beads] Antivector-
    accelerations = amplification * acceleration / radius               # [n_time, n_polarizations, n_beads] Antivector-
    largest = accelerations.squared().square_root().max()                # [] Scalar+
    farthest = (stack((mv.x, mv.y)) & positions[..., None]).abs().max()  # [] Scalar+
    return _Detector(
        time=np.asarray(time),
        reference=reference / radius,
        positions=positions,
        tips=positions + accelerations * (ARROW_REACH / largest),
        limit=max(1.35, float(farthest.to_array()) + ARROW_REACH),
        most_deformed=int(displacement.squared().sum(axis=(1, 2)).argmax()),
    )


def _ring_axes(fig: plt.Figure, data: _Detector) -> list[plt.Axes]:
    axes = []
    for column in range(len(polarization_names)):
        ax = fig.add_subplot(1, 3, column + 1)
        ax.set(xlim=(-data.limit, data.limit), ylim=(-data.limit, data.limit), aspect="equal")
        ax.set_axis_off()
        axes.append(ax)
    return axes


def _ring(ax: plt.Axes, data: _Detector, polarization: int, static: bool):
    """The ring of beads in a panel at each frame: the ring at rest dotted over a faint cross, the beads' outline,
    trails of four tracked beads, the beads, arrows of eight beads' accelerations, the tracked beads and the centre;
    with the tracked beads' full trails, faint, for a still."""
    n_beads = data.reference.shape[0]
    selected = np.arange(0, n_beads, max(1, n_beads // 4))[:4]
    arrows_at = np.arange(0, n_beads, max(1, n_beads // 8))
    trail_colours = np.array([to_rgb(colour) for colour in tracking_colours[:len(selected)]])
    camera = flat_panel(ax, mv.x.algebra)
    # The cross through the centre, across the whole view but for room for its names at its ends.
    half_span = 0.9 * ax.get_xlim()[1]
    cross_ends = half_span * stack((-mv.x.dual(), mv.x.dual(), -mv.y.dual(), mv.y.dual()))   # [4] Antivector-
    ax.annotate("x", (half_span, 0), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                color=CROSS_LABEL_COLOUR)
    ax.annotate("y", (0, half_span), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                color=CROSS_LABEL_COLOUR)
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rest_samples) / 2)).exp()   # [n_rest_samples] Rotor+
    rest = turns >> mv.x.dual()                                          # [n_rest_samples] Antivector-
    beneath = [
        simplices.strokes(cross_ends, simplices.Topology.chains([2, 2]), CROSS, charts.FRAME_PX, camera,
                          cap=simplices.Cap.BUTT),
        simplices.strokes(rest, simplices.Topology.chains([n_rest_samples]), LIGHT, charts.LINE_PX, camera,
                          dashes=REST_DOTS)]
    if static:
        tracks = data.positions[:, polarization, selected].reshape(-1)     # [n_time * tracked] Antivector-
        n_time = len(data.time)
        beneath.append(simplices.strokes(
            tracks, simplices.Topology(len(tracks), np.stack((np.arange(len(tracks) - len(selected)),
                                                           np.arange(len(selected), len(tracks))), axis=-1)),
            np.tile(trail_colours, (n_time - 1, 1)), charts.FRAME_PX, camera, opacity=0.22))
    centre = 0 * mv.x.dual()                                            # [] Antivector-
    above = [simplices.markers(centre[None], ORANGE_RGB, charts.MARKER_PX / 2, camera)]
    shown = simplices.overlay(ax, beneath + above)
    n_trail = max(2, len(data.time) // 7)

    def update(frame: int) -> None:
        positions = data.positions[frame, polarization]                  # [n_beads] Antivector-
        first = max(0, frame - n_trail)
        trails = data.positions[np.arange(first, frame + 1)[None, :], polarization, selected[:, None]]   # [tracked, n_trail] Antivector-
        arrows = stack((positions[arrows_at], data.tips[frame, polarization, arrows_at]), axis=-1)   # [arrows, 2] Antivector-
        shown.set_data(simplices.composited(beneath + [
            simplices.strokes(positions, simplices.Topology.loops([n_beads]), BLUE_RGB, charts.LINE_PX, camera,
                              opacity=0.42),
            simplices.strokes(trails.reshape(-1), simplices.Topology.chains([frame + 1 - first] * len(selected)),
                              np.repeat(trail_colours, frame - first, axis=0), charts.LINE_PX, camera, opacity=0.60),
            # Each bead ringed in white.
            simplices.markers(positions, np.ones(3), charts.MARKER_PX / 2 + charts.FRAME_PX, camera),
            simplices.markers(positions, BLUE_RGB, charts.MARKER_PX / 2, camera),
            simplices.strokes(arrows.reshape(-1), simplices.Topology.chains([2] * len(arrows_at)), BLUE_RGB,
                              ARROW_WIDTH, camera, chevrons=np.ones(len(arrows_at)), chevron=CHEVRON * ARROW_WIDTH,
                              opacity=0.85, cap=simplices.Cap.BUTT),
            simplices.markers(positions[selected], np.ones(3), TRACKED_SCALE * charts.MARKER_PX / 2 + charts.FRAME_PX,
                              camera),
            simplices.markers(positions[selected], trail_colours, TRACKED_SCALE * charts.MARKER_PX / 2, camera),
        ] + above))

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
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(-data.limit, data.limit), ylim=(-data.limit, data.limit), aspect="equal")
        ax.set_axis_off()
        update = _ring(ax, data, animated_polarization, static=False)
        count = min(len(data.time), max(2, round((data.time[-1] - data.time[0]) * frames_per_unit_time) + 1))
        for index in np.linspace(0, len(data.time) - 1, count, dtype=int):
            update(index)
            yield capture(fig)
        plt.close(fig)
