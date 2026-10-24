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

from numga import stack

from examples.util.animation import capture
from examples.relativity.curvature import core
from examples.relativity.curvature.core import Antivector, Curvature, Scalar, mv
from examples.util.wgsl import labels, show, solids

INK, BLUE, ORANGE = "#263b51", "#267fa9", "#d17c24"
light_colour = "#c3ccd4"
tracking_colours = ("#d15a67", "#b48616", "#168676", "#8759a6")
polarization_names = ("Plus", "Cross", "Circular")
# The animated ring: the circular packet, alone, at the dots per inch its detail needs.
animated_polarization = polarization_names.index("Circular")
animation_dpi = 64
# The spacetime diagrams, drawn on the GPU with t up: the view's elevation and azimuth in radians, azimuth zero
# looking along +z, the margin as a share of the larger side, and the pixels across; the share of the figure's
# width each diagram takes.
ELEVATION, AZIMUTH = np.radians([15.0, 22.0])
MARGIN = 0.08
N_MAP_PIXELS = 500
PANEL_WIDTH = 0.44
BLUE_RGB, ORANGE_RGB, LIGHT, PATCH = (np.array(to_rgb(colour)) for colour in (BLUE, ORANGE, light_colour, BLUE))
# How much each of a patch's two faces hides.
PATCH_OPACITY = 0.15
# The axes' arrows and their names, in extents of the diagram; widths in pixels, of the axes' arrows and of the
# patches' outlines, and an arrowhead's length and width in widths of its shaft.
AXIS_LENGTH, LABEL_DISTANCE = 0.88, 0.97
AXIS_WIDTH, OUTLINE_WIDTH = 3.0, 3.5
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


def spacetime(value: Antivector) -> np.ndarray:
    """(..., 3) x, z and t of directions for the spacetime panels."""
    return (stack((mv.x, mv.z, mv.t)) & value[..., None]).to_array()


# --- the curvature map at one event ---------------------------------------------------
def patch(first: Antivector, second: Antivector) -> np.ndarray:
    """(4, 3) corners of the area element spanned by two directions, in x, z, t."""
    first, second = spacetime(first), spacetime(second)
    return np.stack([sign_first * 0.5 * first + sign_second * 0.6 * second
                     for sign_first, sign_second in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def arrows(ends: np.ndarray, colour: np.ndarray, width: float, view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Arrows from the origin to ends `[arrows, 3]` in x, z, t, drawn with t up, `width` pixels across: their
    shafts and heads, in one colour."""
    shafts, heads = solids.arrows(solids.mv.point(np.zeros_like(ends)), solids.mv.point(ends), width / 2 * view.pixel,
                                  HEAD_LENGTH * width * view.pixel, HEAD_WIDTH * width / 2 * view.pixel)
    return [(shafts, colour), (heads, colour)]


def panel(ax: plt.Axes, corners: np.ndarray, ends: list[tuple[np.ndarray, np.ndarray, float]], extent: float,
          view: solids.View) -> None:
    """One spacetime diagram in its axes: horizontal x, depth z and vertical t as light arrows named at their tips,
    the area element `[4, 3]` as a translucent plate a pixel thick with its outline, rods joined round at the
    corners, and arrows `[arrows, 3]` each with its colour and width in pixels."""
    axis_ends = np.eye(3) * extent * AXIS_LENGTH
    places = solids.mv.point(corners)                                          # [4] Point
    rims, joints = solids.edges(places, places[np.roll(np.arange(4), -1)], OUTLINE_WIDTH / 2 * view.pixel)
    parts = (arrows(axis_ends, LIGHT, AXIS_WIDTH, view) + [(rims, PATCH), (joints, PATCH)]
             + [part for points, colour, width in ends for part in arrows(points, colour, width, view)])
    area = [(solids.plates(places[None], view.pixel), PATCH, PATCH_OPACITY)]
    show(ax, solids.draw(parts, view, np.ones(3), translucent=area), view)
    labels(ax, view, solids.mv.point(np.eye(3) * extent * LABEL_DISTANCE), ["x", "z", "ct"], ha="center", va="center",
           fontsize=12, color=INK)


def draw_curvature_map(
    curvature: Curvature, observer: Antivector, wave: Antivector, edge: Antivector,
) -> plt.Figure:
    """A spacetime ribbon and its image under curvature.

    The ribbon is the observer's velocity joined with a unit separation. Its image is a null plane
    containing the wave's direction; the observer reads it as a tidal force, which the metric turns
    into an acceleration. The image is drawn as the plane of the wave's direction and that acceleration.
    """
    readout = -(observer & curvature(observer & edge)).dual()           # [] Antivector
    across = readout / np.sqrt(readout.squared().to_array())             # [] Antivector
    patches = (patch(observer, edge), patch(wave, across))
    observer_end, wave_end, readout_end = spacetime(observer)[None], spacetime(wave)[None], spacetime(readout)[None]
    extent = max(1.0, float(np.abs(np.concatenate((*patches, observer_end, wave_end, readout_end))).max())) * 1.22
    # One view for both diagrams, framing the axes and everything drawn on them.
    view = solids.view([solids.mv.point(np.concatenate((*patches, observer_end, wave_end, readout_end,
                                                        np.eye(3) * extent * LABEL_DISTANCE)))], 0.0,
                       ELEVATION, AZIMUTH, MARGIN, N_MAP_PIXELS)
    rows, columns = view.shape
    fig = plt.figure(figsize=(9.5, 9.5 * PANEL_WIDTH * rows / columns), dpi=120, facecolor="white")
    before, after = fig.add_axes((0, 0, PANEL_WIDTH, 1)), fig.add_axes((1 - PANEL_WIDTH, 0, PANEL_WIDTH, 1))
    panel(before, patches[0], [(observer_end, ORANGE_RGB, 4.0)], extent, view)
    panel(after, patches[1], [(wave_end, BLUE_RGB, 4.5), (observer_end, ORANGE_RGB, 4.0), (readout_end, ORANGE_RGB, 5.0)],
          extent, view)
    # The map from the ribbon to its image, an arrow across the gap between the diagrams.
    before.annotate("", xy=(1 - PANEL_WIDTH + 0.01, 0.5), xytext=(PANEL_WIDTH - 0.01, 0.5),
                    xycoords=fig.transFigure, textcoords=fig.transFigure,
                    arrowprops={"arrowstyle": "->", "lw": 1.8, "color": INK})
    return fig


# --- the strain packet and the Doppler check ------------------------------------------
def draw_packet(time: np.ndarray, strains: np.ndarray, curvatures: np.ndarray) -> plt.Figure:
    """The plus and cross strain profiles and the curvature's weights `[times, 2]`, plus in blue and cross in orange,
    each named beside its highest crest."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 4.4), dpi=120, sharex=True, facecolor="white",
                                      layout="constrained")
    for ax, values, name, across in ((top, strains, "strain", ""), (bottom, curvatures, "curvature", "time")):
        ax.plot(time, values[:, 0], color=BLUE, lw=1.6)
        ax.plot(time, values[:, 1], color=ORANGE, lw=1.6)
        plain_axes(ax, across, name)
    crests = strains.argmax(axis=0)                                            # [2]
    for crest, values, name, colour, side in ((crests[0], strains[:, 0], "plus", BLUE, -1),
                                              (crests[1], strains[:, 1], "cross", ORANGE, 1)):
        top.annotate(name, (time[crest], values[crest]), xytext=(side * 10, 0), textcoords="offset points",
                     ha="right" if side < 0 else "left", va="center", color=colour, fontsize=10)
    return fig


def draw_doppler(rapidities: np.ndarray, amplitudes: Scalar) -> plt.Figure:
    """Tidal amplitude measured by observers boosted along the wave, against `np.exp(-2 * rapidities)`."""
    measured = amplitudes.kernel[..., 0]                                 # [n] float
    fine = np.linspace(rapidities.min(), rapidities.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120, facecolor="white", layout="constrained")
    # The Doppler factor squared as a line, the tidal map's largest singular value as dots.
    ax.plot(fine, np.exp(-2 * fine), color=INK, lw=1.5)
    ax.scatter(rapidities, measured, color=BLUE, s=36, zorder=5)
    # The observer at rest, faint.
    ax.axvline(0, color=light_colour, lw=0.8, zorder=0)
    ax.set(ylim=(0, None))
    plain_axes(ax, "rapidity", "tidal amplitude")
    return fig


# --- the detector ring ----------------------------------------------------------------
@dataclass(frozen=True)
class _Detector:
    time: np.ndarray            # [n_time]
    reference: np.ndarray       # [n_beads, 2] in initial-radius units
    positions: np.ndarray       # [n_polarizations, n_time, n_beads, 2]
    accelerations: np.ndarray   # [n_polarizations, n_time, n_beads, 2]
    amplification: float


def _detector(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector, amplification: float,
) -> _Detector:
    """Read the beads out in initial-radius units, with displacements magnified for display."""
    positions = reference + amplification * displacement                # [n_time, n_polarizations, n_beads] Antivector
    radius = float(np.linalg.norm(transverse(reference), axis=-1).mean())
    return _Detector(
        time=np.asarray(time),
        reference=transverse(reference) / radius,
        positions=np.swapaxes(transverse(positions), 0, 1) / radius,
        accelerations=np.swapaxes(transverse(acceleration) * amplification, 0, 1) / radius,
        amplification=amplification,
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
    angles = np.linspace(0, 2 * np.pi, 200)
    radius = np.linalg.norm(reference, axis=-1).mean()
    ax.plot(radius * np.cos(angles), radius * np.sin(angles), color=light_colour, ls=":", lw=1.5)
    ax.scatter([0], [0], s=22, color=ORANGE, zorder=8)
    # The cross through the centre, across the whole view but for room for its names at its ends.
    reach = 0.9 * ax.get_xlim()[1]
    ax.plot([-reach, reach], [0, 0], color="#edf0f3", lw=0.8, zorder=0)
    ax.plot([0, 0], [-reach, reach], color="#edf0f3", lw=0.8, zorder=0)
    ax.annotate("x", (reach, 0), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                color=CROSS_LABEL_COLOUR)
    ax.annotate("y", (0, reach), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                color=CROSS_LABEL_COLOUR)
    outline, = ax.plot([], [], color=BLUE, lw=1.1, alpha=0.42, zorder=2)
    beads = ax.scatter([], [], s=19, color=BLUE, edgecolors="white", linewidths=0.4, zorder=5)
    markers = ax.scatter(*reference[selected].T, s=47, c=tracking_colours[:len(selected)],
                         edgecolors="white", linewidths=0.8, zorder=7)
    trails = [ax.plot([], [], color=colour, lw=1.6, alpha=0.60, zorder=4)[0]
              for colour in tracking_colours[:len(selected)]]
    maximum = float(np.linalg.norm(data.accelerations, axis=-1).max())
    quiver = ax.quiver(*reference[arrows_at].T, np.zeros(len(arrows_at)), np.zeros(len(arrows_at)),
                       color=BLUE, alpha=0.85, angles="xy", scale_units="xy",
                       scale=max(maximum, 1e-15) / 0.31, width=0.006, headwidth=3.5, headlength=4, zorder=6)
    if static:
        for index in selected:
            ax.plot(*data.positions[polarization, :, index].T,
                    color=tracking_colours[list(selected).index(index)], lw=1.0, alpha=0.22)

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


def draw_detector(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector,
    amplification: float,
) -> plt.Figure:
    """The three polarizations at one shared instant of largest deformation, with faint full trails."""
    data = _detector(time, reference, displacement, acceleration, amplification)
    fig = plt.figure(figsize=(14, 5.6), dpi=120, facecolor="white")
    axes = _ring_axes(fig, data)
    offsets = data.positions - data.reference
    frame = int(np.argmax(np.sum(offsets * offsets, axis=(0, 2, 3))))
    for polarization, ax in enumerate(axes):
        _ring(ax, data, polarization, static=True)(frame)
    return fig


def animate_detector(
    time: np.ndarray, reference: Antivector, displacement: Antivector, acceleration: Antivector, amplification: float,
) -> Iterator[np.ndarray]:
    """The bead ring through the circular packet, as frames at 20 per unit time."""
    data = _detector(time, reference, displacement, acceleration, amplification)
    fig = plt.figure(figsize=(5.2, 5.2), dpi=animation_dpi, facecolor="white")
    limit = max(1.35, float(np.abs(data.positions).max()) + 0.31)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.set_axis_off()
    update = _ring(ax, data, animated_polarization, static=False)
    count = min(len(data.time), max(2, round((data.time[-1] - data.time[0]) * 20) + 1))
    for index in np.linspace(0, len(data.time) - 1, count, dtype=int):
        update(index)
        yield capture(fig)
    plt.close(fig)
