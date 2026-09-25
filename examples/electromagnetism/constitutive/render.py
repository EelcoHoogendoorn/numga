"""Rendering and plotting functions for electromagnetic constitutive maps.

Visualizations:
1. 3D Traveling wave propagation & polarization precession.
2. 1D Dispersion resonance spectrum (singular value dips).
3. 2D Transverse polarization mode quivers.
4. 2D Polar Fresnel wave surfaces (ordinary/extraordinary sheets and Doppler shift).
5. Relativistic Fresnel drag curves vs. Einstein velocity addition.
"""

from __future__ import annotations

from typing import Sequence

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
    modes: Sequence[tuple[float, Extensor]],
    tau: float = 0.0,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
) -> None:
    """Draw 3D spatial snapshot of traveling E and B field vectors in medium along propagation axis z."""
    z = np.linspace(0, z_max, 200)

    # Accumulate field trajectories over modes:
    ex_total = np.zeros_like(z)
    ey_total = np.zeros_like(z)
    bx_total = np.zeros_like(z)
    by_total = np.zeros_like(z)

    for speed, pol in modes:
        xy = pol.cast(core.STA.subspace("x y")).kernel
        px, py = float(xy[0]), float(xy[1])
        norm = np.hypot(px, py)
        if norm > 1e-6:
            px, py = px / norm, py / norm
        k = omega / speed
        phase = k * z - omega * tau
        # Electric field along pol:
        ex_total += px * np.cos(phase)
        ey_total += py * np.cos(phase)
        # Magnetic field: orthogonal to E and z, scaled by 1/speed:
        bx_total += (-py * np.cos(phase)) * (0.5 / speed)
        by_total += (px * np.cos(phase)) * (0.5 / speed)

    # Propagation centerline:
    ax.plot([0, 0], [0, 0], [0, z_max], color="gray", linestyle="--", linewidth=1.2, alpha=0.5)

    # Continuous wave envelopes:
    ax.plot(ex_total, ey_total, z, color="crimson", linewidth=2.0, zorder=5)
    ax.plot(bx_total, by_total, z, color="dodgerblue", linewidth=1.6, linestyle="--", zorder=4)

    # Quiver arrows at discrete stations:
    n_stations = 21
    z_s = np.linspace(0, z_max, n_stations)
    for zi in z_s:
        ex_i, ey_i, bx_i, by_i = 0.0, 0.0, 0.0, 0.0
        for speed, pol in modes:
            xy = pol.cast(core.STA.subspace("x y")).kernel
            px, py = float(xy[0]), float(xy[1])
            norm = np.hypot(px, py)
            if norm > 1e-6:
                px, py = px / norm, py / norm
            k = omega / speed
            phase_i = k * zi - omega * tau
            ex_i += px * np.cos(phase_i)
            ey_i += py * np.cos(phase_i)
            bx_i += (-py * np.cos(phase_i)) * (0.5 / speed)
            by_i += (px * np.cos(phase_i)) * (0.5 / speed)
        ax.quiver(0, 0, zi, ex_i, ey_i, 0, color="crimson", alpha=0.75, arrow_length_ratio=0.18, linewidth=1.4)
        ax.quiver(0, 0, zi, bx_i, by_i, 0, color="dodgerblue", alpha=0.6, arrow_length_ratio=0.18, linewidth=1.1)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([0, z_max])
    ax.view_init(elev=20, azim=-60)
    _no_ticks(ax)
    ax.set_zticks([])


def draw_wave_comparison_3d(
    fig: plt.Figure,
    glass_modes: Sequence[tuple[float, Extensor]],
    crystal_modes: Sequence[tuple[float, Extensor]],
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
    glass_modes: Sequence[tuple[float, Extensor]],
    crystal_modes: Sequence[tuple[float, Extensor]],
    z_max: float = 4.0 * np.pi,
) -> plt.Figure:
    """Render standalone figure with side-by-side 3D views comparing isotropic vs birefringent medium."""
    fig = plt.figure(figsize=(14, 6), dpi=120)
    draw_wave_comparison_3d(fig, glass_modes, crystal_modes, z_max=z_max)
    plt.tight_layout()
    return fig


def animate_wave_propagation(
    modes: Sequence[tuple[float, Extensor]],
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
    modes: list[tuple[float, Extensor]],
) -> None:
    """Draw 2D transverse polarization arrows in the xy plane for the slow and fast birefringent modes."""
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    # The slow wave in crimson, the fast wave in blue.
    for (speed, pol), color in zip(modes, ("crimson", "dodgerblue")):
        # JIT coordinate readout at visualization boundary:
        xy = pol.cast(core.STA.subspace("x y")).kernel
        vx, vy = float(xy[0]), float(xy[1])
        norm = np.hypot(vx, vy)
        if norm > 1e-6:
            vx, vy = vx / norm, vy / norm
        ax.quiver(
            0, 0, vx, vy,
            angles="xy", scale_units="xy", scale=1,
            color=color, width=0.015,
        )
        # Bidirectional polarization oscillation line:
        ax.plot([-vx, vx], [-vy, vy], color=color, linestyle=":", alpha=0.6)

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


def draw_polarizations_figure(modes: list[tuple[float, Extensor]]) -> plt.Figure:
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
