"""Geometric views of a plane wave's curvature and its tidal readout.

The upper panels depict local bivector maps, not successive steps in time.
The lower panels show weak-wave particle displacements magnified for display;
acceleration arrows share one fixed scale throughout the animation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.integrate import cumulative_simpson

from numga import Algebra, Extensor, NumpyContext, stack

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: R_{1,3}, t+ x- y- z-)
# ---------------------------------------------------------------------------
STA = Algebra("t+x-y-z-")
ctx = NumpyContext(STA)
mv = ctx.multivector
t, x, y, z = mv.vector(np.eye(4))

Scalar = STA.gatype.scalar()
Vector = STA.gatype.vector()
Bivector = STA.gatype.bivector()
Rotor = STA.gatype.rotor()



INK, BLUE, ORANGE = "#263b51", "#267fa9", "#d17c24"
MUTED, LIGHT = "#73808c", "#c3ccd4"
TRACKING = ("#d15a67", "#b48616", "#168676", "#8759a6")


@dataclass
class CurvaturePlot:
    time: np.ndarray
    reference: np.ndarray
    positions: np.ndarray
    accelerations: np.ndarray
    input_patch: np.ndarray
    output_patch: np.ndarray
    observer_arrow: np.ndarray
    readout_arrow: np.ndarray
    wave_arrow: np.ndarray
    amplification: float


def _arrow(ax, points: np.ndarray, colour: str, width: float = 2.0) -> None:
    start, end = points
    ax.quiver(*start, *(end - start), color=colour, linewidth=width,
              arrow_length_ratio=0.15, normalize=False)


def _spacetime(ax, extent: float) -> None:
    """One consistent view: horizontal x, depth z, and vertical ct."""
    ax.set_proj_type("ortho")
    ax.view_init(elev=15, azim=-68)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent),
           zlim=(-extent, extent), box_aspect=(1, 1, 1))
    ax.set_axis_off()
    for end, label in zip(np.eye(3) * extent * 0.88, ("x", "z", "ct")):
        _arrow(ax, np.array([np.zeros(3), end]), LIGHT, 0.9)
        ax.text(*(end * 1.12), label, color=MUTED, fontsize=10)


def _patch(ax, vertices: np.ndarray, colour: str) -> None:
    ax.add_collection3d(Poly3DCollection(
        [vertices], facecolors=colour, edgecolors=colour,
        linewidths=1.7, alpha=0.28))
    closed = np.vstack((vertices, vertices[0]))
    ax.plot(*closed.T, color=colour, lw=1.8)


def _map_panels(fig: plt.Figure, data: CurvaturePlot) -> None:
    extent = max(1.0, np.max(np.abs(np.concatenate((
        data.input_patch, data.output_patch, data.observer_arrow,
        data.readout_arrow, data.wave_arrow))))) * 1.22
    headings = (
        ("A small spacetime ribbon", "Separation swept forward in time"),
        ("A nonzero null plane", "The output is a Lorentz generator"),
        ("Apply curvature again: zero", "Every output lies in the kernel"),
    )
    axes = []
    for column, (title, subtitle) in enumerate(headings):
        centre = 0.177 + 0.323 * column
        fig.text(centre, 0.850, title, ha="center", fontsize=13,
                 fontweight="bold", color=INK)
        fig.text(centre, 0.824, subtitle, ha="center", fontsize=10, color=MUTED)
        ax = fig.add_axes([0.042 + 0.323 * column, 0.565, 0.27, 0.265],
                          projection="3d")
        _spacetime(ax, extent)
        axes.append(ax)
    _patch(axes[0], data.input_patch, BLUE)
    _patch(axes[1], data.output_patch, BLUE)
    _arrow(axes[0], data.observer_arrow, ORANGE)
    _arrow(axes[1], data.wave_arrow, BLUE, 2.3)
    _arrow(axes[1], data.observer_arrow, ORANGE)
    _arrow(axes[1], data.readout_arrow, ORANGE, 2.7)
    axes[1].text(*(data.readout_arrow[1] * 1.10), "tidal\nreadout",
                 color=ORANGE, fontsize=9, ha="center")
    axes[1].text2D(0.74, 0.76, "Lightlike\ndirection", transform=axes[1].transAxes,
                   color=BLUE, fontsize=9, ha="center")
    axes[2].scatter([0], [0], [0], s=32, color=INK, depthshade=False)
    axes[2].text2D(0.5, 0.06, "No oriented area remains", transform=axes[2].transAxes,
                   color=MUTED, fontsize=10, ha="center")
    for left, right in ((0.303, 0.365), (0.628, 0.689)):
        axes[0].annotate("", xy=(right, 0.711), xytext=(left, 0.711),
                         xycoords=fig.transFigure, textcoords=fig.transFigure,
                         arrowprops={"arrowstyle": "->", "lw": 1.8, "color": INK})
        fig.text((left + right) / 2, 0.733, "curvature", ha="center",
                 fontsize=9, color=INK)
    fig.text(0.5, 0.572, "One unit-strength plus snapshot; the same local map applied twice at one event.",
             ha="center", fontsize=10, color=MUTED)


def _new_figure(data: CurvaturePlot):
    fig = plt.figure(figsize=(15, 10), dpi=150, facecolor="white")
    fig.suptitle("Zero eigenvalues. Nonzero tidal motion.", y=0.966,
                 fontsize=23, color=INK, fontweight="medium")
    fig.text(0.5, 0.921, "A vacuum plane gravitational wave makes a nonzero curvature map "
             "whose square vanishes.", ha="center", fontsize=12, color=INK)
    fig.text(0.057, 0.882, "THE FULL CURVATURE MAP", fontsize=10,
             fontweight="bold", color=BLUE)
    _map_panels(fig, data)

    # An observer-bound readout is a different map, not a second application.
    fig.text(0.5, 0.528, "Choose the orange time direction: curvature becomes "
             "a map from separation to relative acceleration.", ha="center",
             fontsize=12, color=INK,
             bbox={"facecolor": "#fbf2e7", "edgecolor": "none", "pad": 9})
    fig.text(0.057, 0.478, "WHAT A FREELY FALLING OBSERVER MEASURES", fontsize=10,
             fontweight="bold", color=BLUE)
    fig.text(0.943, 0.478, "Wave travels out of the page  ⊙", ha="right",
             fontsize=10, color=MUTED)

    limit = max(1.35, float(np.abs(data.positions).max()) + 0.31)
    axes = []
    for column, (title, detail) in enumerate((
            ("Plus", "Stretch and squeeze exchange"),
            ("Cross", "The same pattern, turned by 45°"),
            ("Circular", "Turning shape; orbiting beads"))):
        ax = fig.add_axes([0.047 + 0.323 * column, 0.166, 0.258, 0.270])
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, color=INK, fontweight="bold", pad=12)
        ax.text(0.5, -0.02, detail, transform=ax.transAxes, ha="center",
                fontsize=10, color=MUTED)
        axes.append(ax)
    clock = fig.text(0.5, 0.105, "", ha="center", fontsize=11, color=INK)
    handles = [Line2D([], [], color=LIGHT, ls=":", label="Reference circle"),
               Line2D([], [], color=TRACKING[0], lw=2, label="Tracked bead and trail"),
               Line2D([], [], color=BLUE, marker=">", markersize=6,
                      label="Relative acceleration")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.054),
               ncol=3, frameon=False, fontsize=10, columnspacing=3)
    fig.text(0.5, 0.035, "Freely floating beads; outlines only guide the eye. "
             f"Small detector, weak wave; displacements magnified ×{data.amplification:g}.",
             ha="center", fontsize=10, color=MUTED)
    fig.text(0.5, 0.014, "A finite pulse: the beads begin and end at rest "
             "on the reference circle, to first order.",
             ha="center", fontsize=9, color=MUTED)
    return fig, axes, clock


def _detector(ax, data: CurvaturePlot, polarization: int, static: bool):
    reference = data.reference
    selected = np.arange(0, len(reference), max(1, len(reference) // 4))[:4]
    arrows_at = np.arange(0, len(reference), max(1, len(reference) // 8))
    angles = np.linspace(0, 2 * np.pi, 200)
    radius = np.linalg.norm(reference, axis=-1).mean()
    ax.plot(radius * np.cos(angles), radius * np.sin(angles),
            color=LIGHT, ls=":", lw=1.5)
    ax.scatter([0], [0], s=22, color=ORANGE, zorder=8)
    ax.text(0.06, 0.06, "observer", fontsize=8, color=ORANGE)
    ax.plot([-radius * 1.08, radius * 1.08], [0, 0], color="#edf0f3", lw=0.8, zorder=0)
    ax.plot([0, 0], [-radius * 1.08, radius * 1.08], color="#edf0f3", lw=0.8, zorder=0)
    outline, = ax.plot([], [], color=BLUE, lw=1.1, alpha=0.42, zorder=2)
    beads = ax.scatter([], [], s=19, color=BLUE, edgecolors="white", linewidths=0.4, zorder=5)
    markers = ax.scatter(*reference[selected].T, s=47, c=TRACKING[:len(selected)],
                         edgecolors="white", linewidths=0.8, zorder=7)
    trails = [ax.plot([], [], color=colour, lw=1.6, alpha=0.60, zorder=4)[0]
              for colour in TRACKING[:len(selected)]]
    maximum = float(np.linalg.norm(data.accelerations, axis=-1).max())
    quiver = ax.quiver(*reference[arrows_at].T, np.zeros(len(arrows_at)),
                       np.zeros(len(arrows_at)), color=BLUE, alpha=0.85,
                       angles="xy", scale_units="xy", scale=max(maximum, 1e-15) / 0.31,
                       width=0.006, headwidth=3.5, headlength=4, zorder=6)
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


def draw_curvature(data: CurvaturePlot, plot_path: str) -> plt.Figure:
    """Save the local map composition and one shared snapshot of its readouts."""
    fig, axes, clock = _new_figure(data)
    displacement = data.positions - data.reference
    frame = int(np.argmax(np.sum(displacement * displacement, axis=(0, 2, 3))))
    for polarization, ax in enumerate(axes):
        _detector(ax, data, polarization, static=True)(frame)
    clock.set_text(f"One shared instant: {data.time[frame]:.2f} s into the pulse  ·  "
                   "Faint trails show the full motion")
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, facecolor="white")
    return fig


def save_animation(data: CurvaturePlot, animation_path: str) -> None:
    """Animate the three observer readouts while keeping the local map fixed."""
    fig, axes, clock = _new_figure(data)
    updates = [_detector(ax, data, polarization, static=False)
               for polarization, ax in enumerate(axes)]

    def frame(index: int):
        clock.set_text(f"Shared time: {data.time[index]:.2f} s  ·  "
                       "The local map diagram above stays fixed")
        artists = [clock]
        for update in updates:
            artists.extend(update(index))
        return artists

    count = min(len(data.time), max(2, round((data.time[-1] - data.time[0]) * 20) + 1))
    frames = np.linspace(0, len(data.time) - 1, count, dtype=int)
    animation = FuncAnimation(fig, frame, frames=frames, interval=50, blit=False)
    Path(animation_path).parent.mkdir(parents=True, exist_ok=True)
    animation.save(animation_path, writer=PillowWriter(fps=20), dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Coefficient work: the waveform, the integration and the plot read-out
# ---------------------------------------------------------------------------
def wave_packet(
    time: np.ndarray, duration: float = 6.0, cycles: int = 3, amplitude: float = 1e-4,
) -> tuple[Scalar, Scalar]:
    """Cosine/sine strain profiles and second derivatives, as (time, phase) batches.

    A sin^4 envelope has zero value and first two derivatives at its endpoints.
    Both profiles and their derivatives are zero outside the packet.
    """
    time = np.asarray(time, dtype=float)
    s = np.clip(time / duration, 0.0, 1.0)
    a, b = np.pi / duration, 2 * np.pi * cycles / duration
    sine, cosine = np.sin(np.pi * s), np.cos(np.pi * s)
    envelope = amplitude * sine**4
    first = amplitude * 4 * a * sine**3 * cosine
    second = amplitude * 4 * a**2 * (3 * sine**2 * cosine**2 - sine**4)
    phase = b * (time - duration / 2)
    carrier = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    derivative = b * np.stack((-np.sin(phase), np.cos(phase)), axis=-1)
    profile = envelope[:, None] * carrier
    acceleration = (second - b**2 * envelope)[:, None] * carrier + 2 * first[:, None] * derivative
    inside = ((time > 0) & (time < duration))[:, None]
    return mv.scalar((profile * inside)[..., None]), mv.scalar((acceleration * inside)[..., None])


def integrate_acceleration(time: np.ndarray, acceleration: Vector) -> Vector:
    """Numerical boundary: integrate along the first batch axis, starting at rest."""
    velocity = cumulative_simpson(acceleration.kernel, x=time, axis=0, initial=0)
    displacement = cumulative_simpson(velocity, x=time, axis=0, initial=0)
    return mv.vector(displacement)


def plane_patch(area: Bivector, edge: Vector) -> Vector:
    """Draw a simple area using a unit spacelike edge contained in its plane."""
    other: Vector = edge.commutator(area) * .5
    edge = edge * .6
    return stack((-edge - other, -edge + other, edge + other, edge - other))


def detector_ring(count: int = 24) -> Vector:
    """Unit reference separations in the plane transverse to the wave."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    coordinates = np.zeros((count, 4))
    coordinates[:, 1:3] = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    return mv.vector(coordinates)


def plot_data(
    time: np.ndarray, reference: Vector, displacement: Vector,
    acceleration: Vector, curvature: Extensor, amplification: float,
):
    """Coefficient boundary: read detector coordinates and geometric map patches."""
    def xy(value: Vector) -> np.ndarray:
        return value.select_subspace(STA.subspace("x y")).kernel

    def xzt(value: Vector) -> np.ndarray:
        return value.select_subspace(STA.subspace("x z t")).kernel

    def arrow(value: Vector) -> np.ndarray:
        return xzt(mv.scalar([[0], [1]]) * value)

    incoming: Bivector = t.wedge(x)
    outgoing: Bivector = curvature(incoming)
    positions = reference + amplification * displacement
    # Draw in initial-radius units, independently of the physical detector size.
    radius = np.linalg.norm(xy(reference), axis=-1).mean()
    return CurvaturePlot(
        time=time,
        reference=xy(reference) / radius,
        positions=np.swapaxes(xy(positions), 0, 1) / radius,
        accelerations=np.swapaxes(xy(acceleration) * amplification, 0, 1) / radius,
        input_patch=xzt(plane_patch(incoming, x)),
        output_patch=xzt(plane_patch(outgoing, x)),
        observer_arrow=arrow(t),
        readout_arrow=arrow(outgoing.commutator(t)),
        wave_arrow=arrow(t + z),
        amplification=amplification,
    )
