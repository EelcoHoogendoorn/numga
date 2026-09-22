"""Rendering and plotting functions for electromagnetic constitutive maps.

Visualizations:
1. 3D Traveling wave propagation & polarization precession.
2. 1D Dispersion resonance spectrum (singular value dips).
3. 2D Transverse polarization mode quivers.
4. 2D Polar Fresnel wave surfaces (ordinary/extraordinary sheets and Doppler shift).
5. Relativistic Fresnel drag curves vs. Einstein velocity addition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from examples.electromagnetism.constitutive import core


def draw_wave_propagation(
    ax: plt.Axes,
    modes: Sequence[tuple[float, Any]],
    tau: float = 0.0,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
    title: str = r"Traveling $\mathbf{E}$ and $\mathbf{B}$ Field Vectors",
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
    ax.plot(ex_total, ey_total, z, color="crimson", linewidth=2.0, label="E", zorder=5)
    ax.plot(bx_total, by_total, z, color="dodgerblue", linewidth=1.6, linestyle="--", label="B", zorder=4)

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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=10, pad=6)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.view_init(elev=20, azim=-60)


def draw_wave_comparison_3d(
    fig: plt.Figure,
    glass_modes: Sequence[tuple[float, Any]],
    crystal_modes: Sequence[tuple[float, Any]],
    z_max: float = 4.0 * np.pi,
) -> tuple[plt.Axes, plt.Axes]:
    """Render side-by-side 3D views comparing isotropic vs birefringent medium."""
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_wave_propagation(
        ax1, glass_modes, z_max=z_max,
        title="isotropic glass",
    )
    draw_wave_propagation(
        ax2, crystal_modes, z_max=z_max,
        title="birefringent crystal",
    )
    return ax1, ax2


def draw_wave_comparison_figure(
    glass_modes: Sequence[tuple[float, Any]],
    crystal_modes: Sequence[tuple[float, Any]],
    z_max: float = 4.0 * np.pi,
    plot_path: Path | str | None = None,
) -> plt.Figure:
    """Render standalone figure with side-by-side 3D views comparing isotropic vs birefringent medium."""
    fig = plt.figure(figsize=(14, 6), dpi=120)
    draw_wave_comparison_3d(fig, glass_modes, crystal_modes, z_max=z_max)
    plt.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), bbox_inches="tight")
    return fig


def draw_wave_propagation_figure(
    modes: Sequence[tuple[float, Any]],
    tau: float = 0.0,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
    title: str = r"Traveling $\mathbf{E}$ and $\mathbf{B}$ Field Vectors",
    plot_path: Path | str | None = None,
) -> plt.Figure:
    """Render standalone 3D figure of traveling E and B field vectors through medium."""
    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    draw_wave_propagation(ax, modes, tau=tau, z_max=z_max, omega=omega, title=title)
    plt.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), bbox_inches="tight")
    return fig


def animate_wave_propagation(
    modes: Sequence[tuple[float, Any]],
    plot_path: Path | str | None = None,
    n_frames: int = 24,
    duration_ms: int = 50,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
) -> Path:
    """Render animated GIF of traveling E and B field vectors through medium."""
    from examples.animation import capture, save_gif

    if plot_path is None:
        from examples import PLOT_DIR
        plot_path = PLOT_DIR / "constitutive_wave.gif"
    plot_path = Path(plot_path)

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    frames = []
    for tau in np.linspace(0, 2.0 * np.pi / omega, n_frames, endpoint=False):
        ax.cla()
        draw_wave_propagation(ax, modes, tau=float(tau), z_max=z_max, omega=omega)
        frames.append(capture(fig))

    plt.close(fig)
    save_gif(frames, str(plot_path), duration_ms=duration_ms)
    return plot_path


def draw_dispersion(
    ax: plt.Axes,
    speeds: np.ndarray,
    curves: dict[str, np.ndarray],
    expected: dict[str, list[float]],
) -> None:
    """Plot log smallest singular values vs. trial phase speeds, marking expected roots."""
    for name, curve in curves.items():
        is_axion = "axion" in name
        data = curve.to_array() if hasattr(curve, "to_array") else np.asarray(curve)
        line, = ax.semilogy(
            speeds,
            data,
            label=name,
            linestyle="--" if is_axion else "-",
            linewidth=1.8 if is_axion else 1.5,
        )
        for v in expected.get(name, []):
            ax.axvline(v, color=line.get_color(), linestyle=":", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("phase speed v")
    ax.set_ylabel(r"smallest $\sigma$")
    ax.set_title("dispersion scan", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)


def draw_polarizations(
    ax: plt.Axes,
    modes: list[tuple[str, float, Any, str]],
) -> None:
    """Draw 2D transverse polarization arrows in the xy plane for the birefringent modes."""
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    for label, speed, pol, color in modes:
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
            label=f"{label} (v={speed:.2f})",
        )
        # Bidirectional polarization oscillation line:
        ax.plot([-vx, vx], [-vy, vy], color=color, linestyle=":", alpha=0.6)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_aspect("equal")
    ax.set_xlabel("Ex")
    ax.set_ylabel("Ey")
    ax.set_title("polarization modes", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def draw_fresnel_surface_polar(
    ax: plt.Axes,
    angles: np.ndarray,
    surfaces: dict[str, Any],
    speeds: np.ndarray | None = None,
) -> None:
    """Plot 2D polar Fresnel wave normal surfaces v(theta) in the xz propagation plane."""
    for name, data in surfaces.items():
        data_arr = data.to_array() if hasattr(data, "to_array") else data
        if isinstance(data_arr, np.ndarray) and speeds is not None:
            # Vectorized 2D singular value map [n_speeds, n_angles]: extract local minimum sheets
            svals = data_arr
            left, mid, right = svals[:-2], svals[1:-1], svals[2:]
            interior = (mid <= left) & (mid <= right) & (mid < 4e-3)
            sheet_speeds = []
            for j in range(len(angles)):
                idx = np.where(interior[:, j])[0]
                sheet_speeds.append(speeds[1:-1][idx].tolist())
        else:
            sheet_speeds = data_arr

        max_branches = max((len(s) for s in sheet_speeds), default=0)
        for b in range(max_branches):
            branch_r = []
            branch_theta = []
            for theta, spds in zip(angles, sheet_speeds):
                if b < len(spds):
                    branch_r.append(spds[b])
                    branch_theta.append(theta)
            if branch_r:
                label = name if b == 0 else None
                ax.plot(branch_theta, branch_r, label=label, linewidth=1.6)

    ax.set_theta_zero_location("N")  # 0 radians along +z (North)
    ax.set_theta_direction(-1)       # Clockwise: +x along East
    ax.set_title("wave surfaces", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.0), fontsize=8.5)


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
    ax.plot(betas, v_down, "ro", markersize=5, label="downstream")
    ax.plot(betas, v_up, "bs", markersize=5, label="upstream")

    # Analytical theory curves:
    ax.plot(beta_fine, einstein_down, "r-", linewidth=1.5, label="Einstein")
    ax.plot(beta_fine, einstein_up, "b-", linewidth=1.5)
    ax.plot(beta_fine, fresnel_down, "r--", linewidth=1.1, alpha=0.7, label="Fresnel 1st-order")
    ax.plot(beta_fine, fresnel_up, "b--", linewidth=1.1, alpha=0.7)

    ax.set_xlabel(r"boost $\beta$")
    ax.set_ylabel("phase velocity v")
    ax.set_title("Fresnel drag", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.9)


def draw_dispersion_figure(
    speeds: np.ndarray,
    curves: dict[str, np.ndarray],
    expected: dict[str, list[float]],
    plot_path: Path | str | None = None,
) -> plt.Figure:
    """Render standalone figure for 1D dispersion resonance notches."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    draw_dispersion(ax, speeds, curves, expected)
    plt.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), bbox_inches="tight")
    return fig


def draw_polarizations_figure(
    modes: list[tuple[str, float, Any, str]],
    plot_path: Path | str | None = None,
) -> plt.Figure:
    """Render standalone figure for 2D transverse polarization eigenmode quivers."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    draw_polarizations(ax, modes)
    plt.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), bbox_inches="tight")
    return fig


def draw_fresnel_surface_figure(
    angles: np.ndarray,
    surfaces: dict[str, Any],
    speeds: np.ndarray | None = None,
    plot_path: Path | str | None = None,
) -> plt.Figure:
    """Render standalone polar figure for 2D Fresnel wave normal surfaces."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, subplot_kw={"projection": "polar"})
    draw_fresnel_surface_polar(ax, angles, surfaces, speeds=speeds)
    plt.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), bbox_inches="tight")
    return fig


def draw_fresnel_drag_figure(
    betas: np.ndarray,
    v_down: np.ndarray,
    v_up: np.ndarray,
    eps: float,
    mu: float,
    plot_path: Path | str | None = None,
) -> plt.Figure:
    """Render standalone figure for relativistic Fresnel drag curves."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    draw_fresnel_drag_curves(ax, betas, v_down, v_up, eps=eps, mu=mu)
    plt.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), bbox_inches="tight")
    return fig


def draw_canonical_figure(
    speeds: np.ndarray,
    dispersion_curves: dict[str, np.ndarray],
    dispersion_expected: dict[str, list[float]],
    polar_angles: np.ndarray,
    polar_surfaces: dict[str, list[list[float]]],
    betas: np.ndarray,
    v_down: np.ndarray,
    v_up: np.ndarray,
    eps: float,
    mu: float,
    plot_path: Path | str | None = None,
) -> plt.Figure:
    """Render 3-panel figure or individual standalone figures."""
    fig = plt.figure(figsize=(15, 5), dpi=120)

    ax1 = fig.add_subplot(1, 3, 1)
    draw_dispersion(ax1, speeds, dispersion_curves, dispersion_expected)

    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    draw_fresnel_surface_polar(ax2, polar_angles, polar_surfaces)

    ax3 = fig.add_subplot(1, 3, 3)
    draw_fresnel_drag_curves(ax3, betas, v_down, v_up, eps, mu)

    plt.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), bbox_inches="tight")
        print(f"Canonical figure saved to {plot_path}")
    return fig
