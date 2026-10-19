"""Drawing the waves media carry: the fields along their way, the residual against trial speeds, the
speeds by heading, and the drag of moving glass."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.electromagnetism.constitutive import core

Modes = tuple[np.ndarray, core.Bivector]


def _no_ticks(ax: plt.Axes) -> None:
    """No tick marks or tick labels."""
    ax.set_xticks([])
    ax.set_yticks([])

def draw_wave_propagation(
    ax: plt.Axes,
    modes: Modes,
    tau: float = 0.0,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
) -> None:
    """Draw 3D spatial snapshot of traveling E and B field vectors in medium along propagation axis z."""
    samples, stations = 200, 21
    distance = np.linspace(0, z_max, samples)
    speeds, fields = modes
    fields = fields / (-(fields | core.t).squared()).square_root()
    phase = omega * (distance[:, None] / speeds - tau)
    wave = (fields * np.cos(phase)).sum(axis=-1)

    # Observer arrows are read out only when drawing the field.
    electric = (wave | core.t).cast(core.STA.subspace("x y z")).kernel
    magnetic = (wave.dual() | core.t).cast(core.STA.subspace("x y z")).kernel
    electric_amplitudes = (fields | core.t).cast(core.STA.subspace("x y z")).kernel
    magnetic_amplitudes = (fields.dual() | core.t).cast(core.STA.subspace("x y z")).kernel
    amplitudes = np.maximum(np.abs(electric_amplitudes).sum(axis=0), np.abs(magnetic_amplitudes).sum(axis=0))
    transverse_extent = 1.15 * amplitudes[:2].max()
    longitudinal_extent = 1.15 * amplitudes[2]

    ax.plot([0, 0], [0, 0], [0, z_max], color="gray", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.plot(electric[:, 0], electric[:, 1], distance + electric[:, 2], color="crimson", linewidth=2.0, zorder=5)
    ax.plot(magnetic[:, 0], magnetic[:, 1], distance + magnetic[:, 2], color="dodgerblue", linewidth=1.6, linestyle="--", zorder=4)

    station_indices = np.linspace(0, samples - 1, stations, dtype=int)
    for arrows, color in ((electric, "crimson"), (magnetic, "dodgerblue")):
        ax.quiver(0, 0, distance[station_indices], *arrows[station_indices].T,
                  color=color, alpha=0.7, arrow_length_ratio=0.18, linewidth=1.2)

    ax.set_xlim([-transverse_extent, transverse_extent])
    ax.set_ylim([-transverse_extent, transverse_extent])
    ax.set_zlim([-longitudinal_extent, z_max + longitudinal_extent])
    ax.view_init(elev=20, azim=-60)
    _no_ticks(ax)
    ax.set_zticks([])


def draw_wave_comparison_3d(
    fig: plt.Figure,
    glass_modes: Modes,
    crystal_modes: Modes,
    z_max: float = 4.0 * np.pi,
) -> tuple[plt.Axes, plt.Axes]:
    """Render side-by-side 3D views comparing isotropic vs birefringent medium."""
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # Isotropic glass on the left, birefringent crystal on the right.
    draw_wave_propagation(ax1, glass_modes, z_max=z_max)
    draw_wave_propagation(ax2, crystal_modes, z_max=z_max)
    return ax1, ax2


def draw_wave_comparison_figure(
    glass_modes: Modes,
    crystal_modes: Modes,
    z_max: float = 4.0 * np.pi,
) -> plt.Figure:
    """Render standalone figure with side-by-side 3D views comparing isotropic vs birefringent medium."""
    fig = plt.figure(figsize=(14, 6), dpi=120)
    draw_wave_comparison_3d(fig, glass_modes, crystal_modes, z_max=z_max)
    plt.tight_layout()
    return fig


def animate_wave_comparison(
    glass_modes: Modes,
    crystal_modes: Modes,
    n_frames: int,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
) -> list[np.ndarray]:
    """Frames of the travelling E and B field vectors through one period, in isotropic glass beside
    the birefringent crystal."""
    fig = plt.figure(figsize=(14, 6), dpi=80)
    glass_axes = fig.add_subplot(1, 2, 1, projection="3d")
    crystal_axes = fig.add_subplot(1, 2, 2, projection="3d")
    # One layout for every frame, so the axes stay put while the waves move.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0)

    frames = []
    for tau in np.linspace(0, 2.0 * np.pi / omega, n_frames, endpoint=False):
        for ax, modes in ((glass_axes, glass_modes), (crystal_axes, crystal_modes)):
            ax.cla()
            draw_wave_propagation(ax, modes, tau=float(tau), z_max=z_max, omega=omega)
        frames.append(capture(fig))

    plt.close(fig)
    return frames


def draw_dispersion(ax: plt.Axes, speeds: np.ndarray, residual: core.Scalar, expected: np.ndarray) -> None:
    """Each medium's residual against the trial speeds on a log scale `[speeds, media]`, its expected
    speeds `[media, 2]` marked in its colour."""
    lines = ax.semilogy(speeds, residual.to_array(), linewidth=1.5)
    for line, marks in zip(lines, expected):
        for speed in marks:
            ax.axvline(speed, color=line.get_color(), linestyle=":", linewidth=1.2, alpha=0.8)

    _no_ticks(ax)


def draw_polarizations(
    ax: plt.Axes,
    modes: Modes,
) -> None:
    """Draw 2D transverse polarization arrows in the xy plane for the slow and fast birefringent modes."""
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    _, fields = modes
    electric = fields | core.t
    electric = electric / (-electric.squared()).square_root()
    arrows = electric.cast(core.STA.subspace("x y")).kernel
    colors = ("crimson", "dodgerblue")
    ax.quiver(np.zeros(len(arrows)), np.zeros(len(arrows)), *arrows.T,
              angles="xy", scale_units="xy", scale=1, color=colors, width=0.015)
    for (horizontal, vertical), color in zip(arrows, colors):
        ax.plot([-horizontal, horizontal], [-vertical, vertical], color=color, linestyle=":", alpha=0.6)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_aspect("equal")
    _no_ticks(ax)


def draw_fresnel_surface_polar(
    ax: plt.Axes,
    angles: np.ndarray,
    residual: core.Scalar,
    speeds: np.ndarray,
    threshold: float,
) -> None:
    """The speeds of the waves against their heading in the xz plane, one colour per medium: the dips of
    each medium's residual along speed `[speeds, headings, media]`, the n-th dip at each heading on the
    n-th sheet."""
    for dips, color in zip(np.moveaxis(core.dips(speeds, residual, threshold), -1, 0), plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        # Each dip's place among the dips at its heading, counting from the slowest.
        sheets = np.cumsum(dips, axis=0) * dips                                # [speeds - 2, headings]
        for sheet in range(1, sheets.max() + 1):
            heading_idx, speed_idx = np.nonzero((sheets == sheet).T)
            ax.plot(angles[heading_idx], speeds[1:-1][speed_idx], color=color, linewidth=1.6)

    # Zero radians along +z (North), angles clockwise: +x along East.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    _no_ticks(ax)


def draw_fresnel_drag_curves(
    ax: plt.Axes,
    fractions: np.ndarray,
    found: np.ndarray,
    curve: np.ndarray,
    relativistic: np.ndarray,
    first_order: np.ndarray,
) -> None:
    """The speeds of light found in the moving glass, with it in red and against it in blue `[2, drags]`,
    over the relativistic sum, solid, and Fresnel's first-order drag, dashed `[2, curve]`."""
    for speeds, exact, approximate, color, marker in zip(found, relativistic, first_order, "rb", "os"):
        ax.plot(fractions, speeds, color=color, marker=marker, linestyle="none", markersize=5)
        ax.plot(curve, exact, color=color, linewidth=1.5)
        ax.plot(curve, approximate, color=color, linestyle="--", linewidth=1.1, alpha=0.7)

    _no_ticks(ax)


def draw_dispersion_figure(speeds: np.ndarray, residual: core.Scalar, expected: np.ndarray) -> plt.Figure:
    """Each medium's residual against the trial speeds."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    draw_dispersion(ax, speeds, residual, expected)
    plt.tight_layout()
    return fig


def draw_polarizations_figure(modes: Modes) -> plt.Figure:
    """Render standalone figure for 2D transverse polarization eigenmode quivers."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    draw_polarizations(ax, modes)
    plt.tight_layout()
    return fig


def draw_fresnel_surface_figure(angles: np.ndarray, residual: core.Scalar, speeds: np.ndarray,
                                threshold: float) -> plt.Figure:
    """The speeds of the waves against their heading, on polar axes."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, subplot_kw={"projection": "polar"})
    draw_fresnel_surface_polar(ax, angles, residual, speeds, threshold)
    plt.tight_layout()
    return fig


def draw_fresnel_drag_figure(
    fractions: np.ndarray,
    found: np.ndarray,
    curve: np.ndarray,
    relativistic: np.ndarray,
    first_order: np.ndarray,
) -> plt.Figure:
    """The speeds of light in the moving glass, against the speed of the glass."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    draw_fresnel_drag_curves(ax, fractions, found, curve, relativistic, first_order)
    plt.tight_layout()
    return fig
