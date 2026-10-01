"""Rendering and plotting functions for electromagnetic constitutive maps.

Visualizations:
1. 3D Traveling wave propagation & polarization precession.
2. 1D Dispersion resonance spectrum (singular value dips).
3. 2D Transverse polarization mode quivers.
4. 2D Polar Fresnel wave surfaces (ordinary/extraordinary sheets and Doppler shift).
5. Relativistic Fresnel drag curves vs. Einstein velocity addition.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor

from examples.animation import capture
from examples.electromagnetism.constitutive import core



def _no_ticks(ax: plt.Axes) -> None:
    """No tick marks or tick labels."""
    ax.set_xticks([])
    ax.set_yticks([])

def draw_wave_propagation(
    ax: plt.Axes,
    modes: core.Modes,
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
    glass_modes: core.Modes,
    crystal_modes: core.Modes,
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
    glass_modes: core.Modes,
    crystal_modes: core.Modes,
    z_max: float = 4.0 * np.pi,
) -> plt.Figure:
    """Render standalone figure with side-by-side 3D views comparing isotropic vs birefringent medium."""
    fig = plt.figure(figsize=(14, 6), dpi=120)
    draw_wave_comparison_3d(fig, glass_modes, crystal_modes, z_max=z_max)
    plt.tight_layout()
    return fig


def animate_wave_propagation(
    modes: core.Modes,
    n_frames: int = 24,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
) -> list[np.ndarray]:
    """Frames of the traveling E and B field vectors through one period."""
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    frames = []
    for tau in np.linspace(0, 2.0 * np.pi / omega, n_frames, endpoint=False):
        ax.cla()
        draw_wave_propagation(ax, modes, tau=float(tau), z_max=z_max, omega=omega)
        frames.append(capture(fig))

    plt.close(fig)
    return frames


def animate_wave_comparison(
    glass_modes: core.Modes,
    crystal_modes: core.Modes,
    n_frames: int = 24,
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


def draw_dispersion(
    ax: plt.Axes,
    speeds: np.ndarray,
    curves: dict[str, np.ndarray],
    expected: dict[str, list[float]],
) -> None:
    """Plot log smallest singular values vs. trial phase speeds, marking expected roots."""
    for name, curve in curves.items():
        is_axion = "axion" in name
        line, = ax.semilogy(
            speeds,
            curve.to_array(),
            linestyle="--" if is_axion else "-",
            linewidth=1.8 if is_axion else 1.5,
        )
        for v in expected.get(name, []):
            ax.axvline(v, color=line.get_color(), linestyle=":", linewidth=1.2, alpha=0.8)

    _no_ticks(ax)


def draw_polarizations(
    ax: plt.Axes,
    modes: core.Modes,
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
    surfaces: dict[str, Extensor],
    speeds: np.ndarray,
) -> None:
    """Plot 2D polar Fresnel wave normal surfaces, phase speed against angle, in the xz propagation plane.

    Each surface is the smallest singular value of the wave map over (speed, angle); its
    local minima along speed are the sheets.
    """
    for name, data in surfaces.items():
        svals = data.to_array()
        left, mid, right = svals[:-2], svals[1:-1], svals[2:]
        interior = (mid <= left) & (mid <= right) & (mid < 4e-3)
        sheet_speeds = []
        for j in range(len(angles)):
            idx = np.where(interior[:, j])[0]
            sheet_speeds.append(speeds[1:-1][idx].tolist())

        max_branches = max((len(s) for s in sheet_speeds), default=0)
        for b in range(max_branches):
            branch_r = []
            branch_theta = []
            for theta, spds in zip(angles, sheet_speeds):
                if b < len(spds):
                    branch_r.append(spds[b])
                    branch_theta.append(theta)
            if branch_r:
                ax.plot(branch_theta, branch_r, linewidth=1.6)

    # Zero radians along +z (North), angles clockwise: +x along East.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    _no_ticks(ax)


def draw_fresnel_drag_curves(
    ax: plt.Axes,
    betas: np.ndarray,
    v_down: np.ndarray,
    v_up: np.ndarray,
    eps: float,
    mu: float,
) -> None:
    """Plot downstream/upstream phase speed vs medium velocity beta."""
    refractive_index = np.sqrt(eps * mu)
    beta_fine = np.linspace(betas[0], betas[-1], 200)

    # Einstein relativistic velocity addition:
    einstein_down = (1.0 / refractive_index + beta_fine) / (1.0 + beta_fine / refractive_index)
    einstein_up = (1.0 / refractive_index - beta_fine) / (1.0 - beta_fine / refractive_index)

    # Classical 1st order Fresnel drag:
    fresnel_drag_coeff = 1.0 - 1.0 / (refractive_index * refractive_index)
    fresnel_down = 1.0 / refractive_index + beta_fine * fresnel_drag_coeff
    fresnel_up = 1.0 / refractive_index - beta_fine * fresnel_drag_coeff

    # Eigensolve points from boosted constitutive extensor:
    ax.plot(betas, v_down, "ro", markersize=5)
    ax.plot(betas, v_up, "bs", markersize=5)

    # Analytical theory curves:
    ax.plot(beta_fine, einstein_down, "r-", linewidth=1.5)
    ax.plot(beta_fine, einstein_up, "b-", linewidth=1.5)
    ax.plot(beta_fine, fresnel_down, "r--", linewidth=1.1, alpha=0.7)
    ax.plot(beta_fine, fresnel_up, "b--", linewidth=1.1, alpha=0.7)

    _no_ticks(ax)


def draw_dispersion_figure(
    speeds: np.ndarray,
    curves: dict[str, np.ndarray],
    expected: dict[str, list[float]],
) -> plt.Figure:
    """Render standalone figure for 1D dispersion resonance notches."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    draw_dispersion(ax, speeds, curves, expected)
    plt.tight_layout()
    return fig


def draw_polarizations_figure(modes: core.Modes) -> plt.Figure:
    """Render standalone figure for 2D transverse polarization eigenmode quivers."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    draw_polarizations(ax, modes)
    plt.tight_layout()
    return fig


def draw_fresnel_surface_figure(angles: np.ndarray, surfaces: dict[str, Extensor], speeds: np.ndarray) -> plt.Figure:
    """Render standalone polar figure for 2D Fresnel wave normal surfaces."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, subplot_kw={"projection": "polar"})
    draw_fresnel_surface_polar(ax, angles, surfaces, speeds)
    plt.tight_layout()
    return fig


def draw_fresnel_drag_figure(
    betas: np.ndarray,
    v_down: np.ndarray,
    v_up: np.ndarray,
    eps: float,
    mu: float,
) -> plt.Figure:
    """Render standalone figure for relativistic Fresnel drag curves."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    draw_fresnel_drag_curves(ax, betas, v_down, v_up, eps=eps, mu=mu)
    plt.tight_layout()
    return fig
