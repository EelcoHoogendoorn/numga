"""Scenarios, rendering, and CLI runner for Spherical Quadric Physics in Cl(3)."""

from __future__ import annotations

import argparse
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.animation import FuncAnimation
import numpy as np

from examples import PLOT_DIR
from examples.animation import save_gif
from examples.quadrics.spherical_quadric_physics import (
    mv,
    Point,
    Plane,
    Bivector,
    Motor,
    SphericalBody,
    pointcloud_inertia,
    make_spherical_quadric,
    simulate,
)


# ---------------------------------------------------------------------------
# 1. Scenarios (Initial Conditions)
# ---------------------------------------------------------------------------
def make_cap_mesh(th_x: float, th_y: float, mass: float = 1.0, n_phi: int = 1536, n_r: int = 10) -> tuple[Point, np.ndarray]:
    """Generate parametric grid of antivector points spanning the cap on S²."""
    tx, ty = np.tan(th_x), np.tan(th_y)
    R, Phi = np.meshgrid(np.linspace(0.0, 1.0, n_r + 1)[1:], np.linspace(0.0, 2.0 * np.pi, n_phi + 1))
    x = tx * R * np.cos(Phi)
    y = ty * R * np.sin(Phi)
    coords = np.stack([x, y, np.ones_like(x)], axis=-1)
    # Sphere area element of the gnomonic parametrization, times trapezoid quadrature weights,
    # so the point masses approximate a uniform mass distribution over the cap.
    area = tx * ty * R / (1.0 + x**2 + y**2) ** 1.5
    w_r = np.ones(n_r); w_r[-1] = 0.5  # the omitted R = 0 node has zero area element
    w_phi = np.ones(n_phi + 1); w_phi[[0, -1]] = 0.5
    point_mass = area * w_phi[:, None] * w_r[None, :]
    point_mass *= mass / point_mass.sum()
    return mv.point(coords).normalized(), point_mass


def create_spherical_quad(
    name: str,
    th_x: float,
    th_y: float,
    mass: float,
    motor: Motor,
    rate: Bivector,
    color: str = "#38bdf8",
    n_phi: int = 384,
) -> SphericalBody:
    """Construct a spherical quadric rigid body with pointcloud inertia on S²."""
    cap, masses = make_cap_mesh(th_x, th_y, mass, n_phi=n_phi)
    I = pointcloud_inertia(cap, masses)
    return SphericalBody(
        name=name,
        color=color,
        mass=mass,
        motor=motor,
        momentum=I(rate),
        Q=make_spherical_quadric(th_x, th_y),
        cap_pts=cap,
        I_inv=I.inverse(),
    )


# Canonical camera rotor folded directly into initial body orientations
CAMERA_MOTOR: Motor = mv.bivector([
    0.2714904022294391,
    0.6097774271693677,
    1.0148400517968847,
]).exp()


def motor_from_spherical(theta: float, phi: float) -> Motor:
    """Construct rotation motor mapping north pole P0 to spherical direction (theta, phi)."""
    axis_x = -np.sin(phi)
    axis_y = np.cos(phi)
    B = mv.yz * axis_x + (-mv.xz) * axis_y
    return (B * (theta / 2.0)).exp()


def setup_crowded_scene() -> list[SphericalBody]:
    """Scenario 'crowded': 7 extreme-shape bodies distributed on S²."""
    specs = [
        # (name, th_x_deg, th_y_deg, mass, theta, phi, rate, color)
        ("Cyan Baton", 30.0, 6.0, 1.0, 0.35, 0.2, [0.8, 2.6, 0.5], "#38bdf8"),
        ("Rose Disc", 17.0, 17.0, 1.2, 1.05, 0.7, [1.8, -1.2, 0.6], "#f43f5e"),
        ("Amber Needle", 28.0, 6.5, 0.9, 1.15, 2.1, [-1.7, 1.5, -0.6], "#fbbf24"),
        ("Emerald Sliver", 24.0, 5.5, 0.8, 1.10, 3.6, [2.0, 1.0, -0.5], "#34d399"),
        ("Purple Dart", 15.0, 4.0, 0.5, 1.25, 4.9, [-2.2, -1.8, 0.7], "#a855f7"),
        ("Orange Oval", 26.0, 9.0, 1.1, 0.90, -0.67, [1.6, 1.4, 0.7], "#fb923c"),
        ("Pink Puck", 9.0, 9.0, 0.4, 1.55, 2.90, [-2.4, 0.8, -0.4], "#ec4899"),
    ]
    return [
        create_spherical_quad(
            name,
            np.radians(th_x),
            np.radians(th_y),
            mass,
            CAMERA_MOTOR * motor_from_spherical(theta, phi),
            mv.bivector(rate),
            color,
        )
        for name, th_x, th_y, mass, theta, phi, rate, color in specs
    ]


def setup_tumbling_scene(
    th_x_deg: float = 40.0,
    th_y_deg: float = 8.0,
    w_intermediate: float = 14.0,
    w_perturb_y: float = 0.1,
    w_perturb_z: float = 0.05,
) -> SphericalBody:
    """Scenario 'tumbling': single asymmetric oval spinning near intermediate principal axis."""
    motor = CAMERA_MOTOR * (
        (mv.xz * (np.radians(15.0) / 2.0)).exp() * (mv.yz * (np.radians(10.0) / 2.0)).exp()
    )
    rate = mv.bivector([w_intermediate, -w_perturb_y, w_perturb_z])
    return create_spherical_quad(
        name="Tumbling Oval",
        th_x=np.radians(th_x_deg),
        th_y=np.radians(th_y_deg),
        mass=1.0,
        motor=motor,
        rate=rate,
        color="#38bdf8",
    )


def setup_hyperbolic_scene() -> list[SphericalBody]:
    """Scenario 'hyperbolic': giant dominant quadric oval (75° x 48°) and agile channel swarm."""
    specs = [
        # (name, th_x_deg, th_y_deg, mass, theta, phi, rate, color, n_phi)
        ("Giant Oval", 75.0, 48.0, 6.0, 0.0, 0.0, [0.3, 0.2, 0.4], "#38bdf8", 1536),
        ("Ruby Needle", 18.0, 4.5, 0.6, 1.57, 0.35, [2.5, 1.5, -0.8], "#f43f5e", 384),
        ("Amber Puck", 10.0, 10.0, 0.7, 1.57, 1.10, [-2.2, 1.2, 0.6], "#fbbf24", 384),
        ("Emerald Dart", 15.0, 4.0, 0.5, 1.57, 1.85, [1.8, -2.0, 0.7], "#34d399", 384),
        ("Purple Sliver", 16.0, 5.0, 0.5, 1.57, 2.65, [-1.6, -1.8, -0.5], "#a855f7", 384),
    ]
    return [
        create_spherical_quad(
            name,
            np.radians(th_x),
            np.radians(th_y),
            mass,
            CAMERA_MOTOR * motor_from_spherical(theta, phi),
            mv.bivector(rate),
            color,
            n_phi=n_phi,
        )
        for name, th_x, th_y, mass, theta, phi, rate, color, n_phi in specs
    ]


SCENARIOS = {
    "crowded": setup_crowded_scene,
    "tumbling": lambda: [setup_tumbling_scene()],
    "hyperbolic": setup_hyperbolic_scene,
}


# ---------------------------------------------------------------------------
# 2. Rendering & Diagnostic Visualization
# ---------------------------------------------------------------------------
BACKGROUND, DISK, RIM = "#0a0f1d", "#111827", "#334155"


def render_spherical_frame(
    bodies: list[SphericalBody],
    motors: list[Motor],
    resolution: int = 240,
    supersample: int = 4,
) -> np.ndarray:
    """Rasterise the front hemisphere by testing every body's quadric at every pixel.

    A pixel inside the disk is a point on S²; it belongs to a body when the primal quadric,
    the inverse of the body's world-frame dual quadric, is negative there. The spherical
    quadric is a cone through the origin, so the front hemisphere already shows both halves.
    """
    n = resolution * supersample
    u, v = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(1.0, -1.0, n))
    r2 = u**2 + v**2
    inside = r2 <= 1.0
    pixels = mv.point(np.stack([u, v, np.sqrt(np.clip(1.0 - r2, 0.0, None))], axis=-1)[inside])

    disk = np.where((r2[inside] > 0.985)[:, None], to_rgb(RIM), to_rgb(DISK))
    for body, motor in zip(bodies, motors):
        quadric = (motor >> body.Q(motor << Plane)).inverse()
        covered = (pixels & quadric(pixels)).kernel[:, 0] < 0.0
        disk[covered] = to_rgb(body.color)

    image = np.empty((n, n, 3))
    image[:] = to_rgb(BACKGROUND)
    image[inside] = disk
    image = image.reshape(resolution, supersample, resolution, supersample, 3).mean(axis=(1, 3))
    return (image * 255.0).round().astype(np.uint8)


def render_spherical_scene(
    ax,
    bodies: list[SphericalBody],
    motors: list[Motor],
) -> None:
    """Draw the rasterised hemisphere on a static orthonormal circular disk."""
    ax.imshow(render_spherical_frame(bodies, motors), extent=(-1.0, 1.0, -1.0, 1.0), interpolation="none")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def render_diagnostic_plot(
    bodies: list[SphericalBody],
    keyframe_snapshots: dict[int, list[Motor]],
    diagnostic_frame_indices: list[int],
    energy_history: list[float],
    momentum_history: list[float],
    dt: float,
    plot_path: str,
    omega_history: list = (),
) -> None:
    """Generate 4-panel diagnostic figure tailored to single-body tumbling or multi-body dynamics."""
    fig_diag = plt.figure(figsize=(14, 11), dpi=110)
    fig_diag.patch.set_facecolor("#0a0f1d")

    is_single = (len(bodies) == 1)

    if is_single:
        snap_titles = [
            f"Frame {diagnostic_frame_indices[0]}: Initial State (Intermediate Spin)",
            f"Frame {diagnostic_frame_indices[1]}: First Inversion (180° Dzhanibekov Flip)",
            f"Frame {diagnostic_frame_indices[2]}: Second Inversion (Return Flip)",
        ]
        snap_keys = [diagnostic_frame_indices[0], diagnostic_frame_indices[1], diagnostic_frame_indices[2]]
    else:
        k3 = diagnostic_frame_indices[3] if len(diagnostic_frame_indices) > 3 else diagnostic_frame_indices[-1]
        snap_titles = [
            f"Frame {diagnostic_frame_indices[0]}: Initial State",
            f"Frame {diagnostic_frame_indices[1]}: Mid-Orbit Trajectories",
            f"Frame {k3}: Post-Impact Ricochets / Rebounds",
        ]
        snap_keys = [diagnostic_frame_indices[0], diagnostic_frame_indices[1], k3]

    for plot_idx, (f_idx, title) in enumerate(zip(snap_keys, snap_titles)):
        ax_s = fig_diag.add_subplot(2, 2, plot_idx + 1)
        ax_s.set_facecolor("#0a0f1d")
        render_spherical_scene(ax_s, bodies, keyframe_snapshots[f_idx])
        ax_s.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)

    # Panel 4: Invariants / Angular Velocity
    ax_m = fig_diag.add_subplot(2, 2, 4)
    ax_m.set_facecolor("#111827")
    time_arr = np.arange(len(energy_history)) * dt
    e_arr = np.array(energy_history)
    m_arr = np.array(momentum_history)
    e_norm = e_arr / e_arr[0]
    m_norm = m_arr / m_arr[0]

    if is_single and len(omega_history) > 0:
        w_arr = np.array(omega_history)
        ax_m.plot(time_arr, w_arr[:, 0], color="#38bdf8", linewidth=2.0, label="ω_yz (Intermediate Axis)")
        ax_m.plot(time_arr, w_arr[:, 1], color="#f43f5e", linewidth=1.5, linestyle="--", label="ω_zx (Major Axis)")
        ax_m.plot(time_arr, w_arr[:, 2], color="#fbbf24", linewidth=1.5, linestyle=":", label="ω_xy (Polar Axis)")
        ax_m.plot(time_arr, e_norm, color="#34d399", linewidth=2.0, alpha=0.8, label="Energy (norm)")
        ax_m.plot(time_arr, m_norm, color="#a855f7", linewidth=2.0, linestyle="-.", alpha=0.8, label="Momentum (norm)")
        ax_m.set_title("Periodic Dzhanibekov Flips (4 Inversions) & Conservation", color="white", fontsize=12, fontweight="bold")
        ax_m.set_ylabel("Body Angular Velocity & Normalized Invariants", color="#cbd5e1", fontsize=9)
    else:
        ax_m.plot(time_arr, e_norm, color="#38bdf8", linewidth=2.0, label="Total Kinetic Energy (normalized)")
        ax_m.plot(time_arr, m_norm, color="#34d399", linewidth=2.0, linestyle="--", label="Total Angular Momentum (normalized)")
        ax_m.set_title("Physical Invariants & Conservation on S²", color="white", fontsize=12, fontweight="bold")
        ax_m.set_ylabel("Normalized Invariant Value", color="#cbd5e1", fontsize=9)
        ax_m.set_ylim(0.95, 1.05)

    ax_m.set_xlabel("Simulation Time (seconds)", color="#cbd5e1", fontsize=9)
    ax_m.tick_params(colors="#94a3b8")
    for spine in ax_m.spines.values():
        spine.set_color("#334155")
    ax_m.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
    ax_m.legend(loc="upper right" if is_single else "lower left", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig_diag)
    print(f"Summary diagnostic plot exported to {plot_path}")


def create_spherical_animation(
    bodies: list[SphericalBody],
    frame_motors: list[list[Motor]],
    dt: float = 0.015,
    repeat: bool = True,
    interval: int = 15,
) -> tuple[plt.Figure, FuncAnimation]:
    """Construct an interactive FuncAnimation animated plot on S²."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.patch.set_facecolor("#0a0f1d")
    ax.set_facecolor("#0a0f1d")

    def update(frame_idx: int):
        ax.cla()
        ax.set_facecolor("#0a0f1d")
        render_spherical_scene(ax, bodies, frame_motors[frame_idx])
        return ax

    anim = FuncAnimation(
        fig, update, frames=len(frame_motors), interval=interval, repeat=repeat
    )
    return fig, anim


def export_simulation_gifs(
    frame_motors: list[list[Motor]],
    bodies: list[SphericalBody],
    gif_path: str,
    dt: float = 0.015,
    plot_path: str = "",
    down_dim: int = 120,
    artifact_dir: str = os.environ.get("ANTIGRAVITY_ARTIFACT_DIR", ""),
) -> tuple[str, str]:
    """Export standard GIF and average-filtered downsampled small GIF, and sync artifacts."""
    print(f"Exporting animated GIF to {gif_path} ({len(frame_motors)} frames)...")
    frames = [render_spherical_frame(bodies, motors) for motors in frame_motors]

    save_gif(frames, gif_path, duration_ms=int(dt * 1000))
    small_gif_path = gif_path.replace(".gif", "_small.gif")
    save_gif(frames, small_gif_path, duration_ms=int(dt * 1000), scale=down_dim / frames[0].shape[0], colors=40)
    orig_sz = os.path.getsize(gif_path)
    small_sz = os.path.getsize(small_gif_path)
    print(f"Small GIF {small_gif_path}: {small_sz / 1024:.1f} KB ({orig_sz / small_sz:.2f}x smaller)")

    if artifact_dir and os.path.isdir(artifact_dir):
        shutil.copy2(gif_path, os.path.join(artifact_dir, os.path.basename(gif_path)))
        shutil.copy2(small_gif_path, os.path.join(artifact_dir, os.path.basename(small_gif_path)))
        if plot_path and os.path.isfile(plot_path):
            shutil.copy2(plot_path, os.path.join(artifact_dir, os.path.basename(plot_path)))
        print(f"Artifacts synced to {artifact_dir}")

    return gif_path, small_gif_path


SCENARIO_CONFIGS = {
    "crowded": {
        "setup": setup_crowded_scene,
        "default_frames": 160,
        "substeps": 6,
        "gif": "spherical_quadric_physics.gif",
        "plot": "spherical_quadric_physics.png",
    },
    "tumbling": {
        "setup": lambda: [setup_tumbling_scene()],
        "default_frames": 240,
        "substeps": 6,
        "gif": "spherical_tumbling_oval.gif",
        "plot": "spherical_tumbling_oval.png",
    },
    "hyperbolic": {
        "setup": setup_hyperbolic_scene,
        "default_frames": 160,
        "substeps": 6,
        "gif": "spherical_hyperbolic_arena.gif",
        "plot": "spherical_hyperbolic_arena.png",
    },
}


def run_scenario(
    scenario_name: str = "crowded",
    num_frames: int = 0,
    dt: float = 0.015,
    gif_path: str = "",
    plot_path: str = "",
    show_window: bool = False,
    export_files: bool = True,
):
    """Execute simulation for a chosen scenario and export all diagnostic figures and GIFs."""
    if scenario_name not in SCENARIO_CONFIGS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIO_CONFIGS.keys())}")

    cfg = SCENARIO_CONFIGS[scenario_name]
    frames = num_frames or cfg["default_frames"]
    gif = str(PLOT_DIR / (gif_path or cfg["gif"]))
    plot = str(PLOT_DIR / (plot_path or cfg["plot"]))

    print(f"\n=== Running Scenario: '{scenario_name}' ({frames} frames, dt = {dt:.3f}s) ===")
    bodies = cfg["setup"]()

    bodies, frame_motors, energy_history, momentum_history, omega_history, keyframe_snapshots, diag_indices = (
        simulate(bodies, frames, dt, cfg["substeps"])
    )

    if export_files:
        render_diagnostic_plot(
            bodies,
            keyframe_snapshots,
            diag_indices,
            energy_history,
            momentum_history,
            dt,
            plot,
            omega_history=omega_history,
        )
        export_simulation_gifs(frame_motors, bodies, gif, dt, plot_path=plot)

    if show_window:
        fig, anim = create_spherical_animation(bodies, frame_motors, dt=dt)
        print("Opening interactive animation window...")
        plt.show()
        return fig, anim



# ---------------------------------------------------------------------------
# 4. CLI Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spherical Quadric Physics & Scenarios in Cl(3)")
    parser.add_argument(
        "--scenario", "-s",
        choices=["crowded", "tumbling", "hyperbolic", "all"],
        default="crowded",
        help="Scenario to run (crowded, tumbling, hyperbolic, or all)",
    )
    parser.add_argument("--window", "-w", action="store_true", help="Open interactive running animation window")
    parser.add_argument("--frames", "-n", type=int, default=0, help="Number of simulation frames (defaults by scenario)")
    parser.add_argument("--no-save", action="store_true", help="Skip exporting GIF and diagnostic plot")
    args = parser.parse_args()

    scenarios_to_run = list(SCENARIO_CONFIGS.keys()) if args.scenario == "all" else [args.scenario]
    for sc in scenarios_to_run:
        run_scenario(
            scenario_name=sc,
            num_frames=args.frames,
            show_window=args.window,
            export_files=not args.no_save,
        )
