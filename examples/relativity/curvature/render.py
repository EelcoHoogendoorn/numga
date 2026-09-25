"""Geometric views of a plane wave's curvature and its tidal readout: drawing only.

Coordinates leave the algebra here, at the plotting boundary. The map panels depict
one local bivector map applied at one event, not successive steps in time. The
detector panels show weak-wave displacements magnified for display; acceleration
arrows share one fixed scale throughout an animation.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from examples.animation import capture
from examples.relativity.curvature import core
from examples.relativity.curvature.core import STA, Bivector, Curvature, Scalar, Vector, mv

INK, BLUE, ORANGE = "#263b51", "#267fa9", "#d17c24"
LIGHT = "#c3ccd4"
TRACKING = ("#d15a67", "#b48616", "#168676", "#8759a6")
POLARIZATIONS = ("Plus", "Cross", "Circular")


# --- coordinate readout ---------------------------------------------------------------
def transverse(value: Vector) -> np.ndarray:
    """(..., 2) x and y coefficients: separations in the plane transverse to the wave."""
    return value.select_subspace(STA.subspace("x y")).kernel


def spacetime(value: Vector) -> np.ndarray:
    """(..., 3) x, z and t coefficients for the spacetime panels."""
    return value.select_subspace(STA.subspace("x z t")).kernel


def arrow(value: Vector) -> np.ndarray:
    """(2, 3) segment from the origin to the tip of a vector, in x, z, t."""
    return spacetime(mv.scalar([[0], [1]]) * value)


# --- the curvature map at one event ---------------------------------------------------
def _arrow(ax, points: np.ndarray, colour: str, width: float = 2.0) -> None:
    start, end = points
    ax.quiver(*start, *(end - start), color=colour, linewidth=width,
              arrow_length_ratio=0.15, normalize=False)


def _spacetime_axes(ax, extent: float) -> None:
    """One consistent view: horizontal x, depth z, and vertical ct."""
    ax.set_proj_type("ortho")
    ax.view_init(elev=15, azim=-68)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent),
           zlim=(-extent, extent), box_aspect=(1, 1, 1))
    ax.set_axis_off()
    for end in np.eye(3) * extent * 0.88:
        _arrow(ax, np.array([np.zeros(3), end]), LIGHT, 0.9)


def _patch(ax, vertices: np.ndarray, colour: str) -> None:
    ax.add_collection3d(Poly3DCollection(
        [vertices], facecolors=colour, edgecolors=colour, linewidths=1.7, alpha=0.28))
    closed = np.vstack((vertices, vertices[0]))
    ax.plot(*closed.T, color=colour, lw=1.8)


def draw_curvature_map(
    curvature: Curvature, observer: Vector, wave: Vector, edge: Vector,
) -> plt.Figure:
    """A spacetime ribbon, its image under curvature, and that image's image.

    The ribbon is the observer's velocity wedged with a unit separation. Its image is a null
    plane containing the wave direction; contracting that plane with the observer gives the
    tidal readout. Applying the curvature once more gives zero.
    """
    incoming: Bivector = observer.wedge(edge)                            # [] Bivector
    outgoing: Bivector = curvature(incoming)                             # [] Bivector
    readout: Vector = outgoing.commutator(observer)                      # [] Vector
    patches = (spacetime(core.plane_patch(incoming, edge)), spacetime(core.plane_patch(outgoing, edge)))
    observer_arrow, wave_arrow, readout_arrow = arrow(observer), arrow(wave), arrow(readout)
    extent = max(1.0, float(np.abs(np.concatenate(
        (*patches, observer_arrow, wave_arrow, readout_arrow))).max())) * 1.22

    fig = plt.figure(figsize=(14, 4.8), dpi=120, facecolor="white")
    axes = []
    for column in range(3):
        ax = fig.add_subplot(1, 3, column + 1, projection="3d")
        _spacetime_axes(ax, extent)
        axes.append(ax)
    _patch(axes[0], patches[0], BLUE)
    _arrow(axes[0], observer_arrow, ORANGE)
    _patch(axes[1], patches[1], BLUE)
    _arrow(axes[1], wave_arrow, BLUE, 2.3)
    _arrow(axes[1], observer_arrow, ORANGE)
    _arrow(axes[1], readout_arrow, ORANGE, 2.7)
    axes[2].scatter([0], [0], [0], s=32, color=INK, depthshade=False)
    for left_ax, right_ax in zip(axes[:-1], axes[1:]):
        left, right = left_ax.get_position().x1 - 0.01, right_ax.get_position().x0 + 0.01
        axes[0].annotate("", xy=(right, 0.5), xytext=(left, 0.5),
                         xycoords=fig.transFigure, textcoords=fig.transFigure,
                         arrowprops={"arrowstyle": "->", "lw": 1.8, "color": INK})
    return fig


# --- the strain packet and the Doppler check ------------------------------------------
def draw_packet(time: np.ndarray, strain: Scalar, second: Scalar) -> plt.Figure:
    """Strain profiles and their second derivatives, the weights of the curvature batch."""
    h, h2 = strain.kernel[..., 0], second.kernel[..., 0]                 # [n_time, n_phases] float
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 4.4), dpi=120, sharex=True, facecolor="white")
    for ax, values in ((top, h), (bottom, h2)):
        # Plus in blue, cross in orange.
        ax.plot(time, values[:, 0], color=BLUE, lw=1.6)
        ax.plot(time, values[:, 1], color=ORANGE, lw=1.6)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def draw_doppler(rapidities: np.ndarray, amplitudes: Scalar) -> plt.Figure:
    """Tidal amplitude measured by observers boosted along the wave, against `np.exp(-2 * rapidities)`."""
    measured = amplitudes.kernel[..., 0]                                 # [n] float
    fine = np.linspace(rapidities.min(), rapidities.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120, facecolor="white")
    # The Doppler factor squared as a line, the tidal map's largest singular value as dots.
    ax.plot(fine, np.exp(-2 * fine), color=INK, lw=1.5)
    ax.scatter(rapidities, measured, color=BLUE, s=36, zorder=5)
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
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
    time: np.ndarray, reference: Vector, displacement: Vector, acceleration: Vector, amplification: float,
) -> _Detector:
    """Read the beads out in initial-radius units, with displacements magnified for display."""
    positions: Vector = reference + amplification * displacement         # [n_time, n_polarizations, n_beads] Vector
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
    for column, title in enumerate(POLARIZATIONS):
        ax = fig.add_subplot(1, 3, column + 1)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, color=INK, fontweight="bold", pad=10)
        axes.append(ax)
    return axes


def _ring(ax: plt.Axes, data: _Detector, polarization: int, static: bool):
    reference = data.reference
    selected = np.arange(0, len(reference), max(1, len(reference) // 4))[:4]
    arrows_at = np.arange(0, len(reference), max(1, len(reference) // 8))
    angles = np.linspace(0, 2 * np.pi, 200)
    radius = np.linalg.norm(reference, axis=-1).mean()
    ax.plot(radius * np.cos(angles), radius * np.sin(angles), color=LIGHT, ls=":", lw=1.5)
    ax.scatter([0], [0], s=22, color=ORANGE, zorder=8)
    ax.plot([-radius * 1.08, radius * 1.08], [0, 0], color="#edf0f3", lw=0.8, zorder=0)
    ax.plot([0, 0], [-radius * 1.08, radius * 1.08], color="#edf0f3", lw=0.8, zorder=0)
    outline, = ax.plot([], [], color=BLUE, lw=1.1, alpha=0.42, zorder=2)
    beads = ax.scatter([], [], s=19, color=BLUE, edgecolors="white", linewidths=0.4, zorder=5)
    markers = ax.scatter(*reference[selected].T, s=47, c=TRACKING[:len(selected)],
                         edgecolors="white", linewidths=0.8, zorder=7)
    trails = [ax.plot([], [], color=colour, lw=1.6, alpha=0.60, zorder=4)[0]
              for colour in TRACKING[:len(selected)]]
    maximum = float(np.linalg.norm(data.accelerations, axis=-1).max())
    quiver = ax.quiver(*reference[arrows_at].T, np.zeros(len(arrows_at)), np.zeros(len(arrows_at)),
                       color=BLUE, alpha=0.85, angles="xy", scale_units="xy",
                       scale=max(maximum, 1e-15) / 0.31, width=0.006, headwidth=3.5, headlength=4, zorder=6)
    if static:
        for index in selected:
            ax.plot(*data.positions[polarization, :, index].T,
                    color=TRACKING[list(selected).index(index)], lw=1.0, alpha=0.22)

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
    time: np.ndarray, reference: Vector, displacement: Vector, acceleration: Vector,
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
    time: np.ndarray, reference: Vector, displacement: Vector, acceleration: Vector, amplification: float,
) -> list[np.ndarray]:
    """The three bead rings through the packet, as frames at 20 per unit time."""
    data = _detector(time, reference, displacement, acceleration, amplification)
    fig = plt.figure(figsize=(14, 5.2), dpi=80, facecolor="white")
    axes = _ring_axes(fig, data)
    updates = [_ring(ax, data, polarization, static=False) for polarization, ax in enumerate(axes)]
    count = min(len(data.time), max(2, round((data.time[-1] - data.time[0]) * 20) + 1))
    frames = []
    for index in np.linspace(0, len(data.time) - 1, count, dtype=int):
        for update in updates:
            update(index)
        frames.append(capture(fig))
    plt.close(fig)
    return frames
