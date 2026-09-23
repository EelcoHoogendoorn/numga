"""Drawing for quadric physics on S² and S³."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga.algebras import Spherical3D
from examples import instantiate
from examples.quadrics.s3_raytracer import core as raytracer
from examples.quadrics.s3_raytracer.render import render as trace

S2 = instantiate("examples.quadrics.elliptic_physics.core", Spherical3D)
S3 = instantiate("examples.quadrics.elliptic_physics.core", raytracer.ga)

BACKGROUND, DISK, RIM = "#0a0f1d", "#111827", "#334155"


# --- the front hemisphere of S² -------------------------------------------------------
def hemisphere(size: int):
    """The front hemisphere seen along z on a size x size grid: coordinates (m, 3) of the pixels
    inside the disk, the disk mask, and which of the disk's pixels form its rim."""
    u, v = np.meshgrid(np.linspace(-1.0, 1.0, size), np.linspace(1.0, -1.0, size))
    r2 = u**2 + v**2
    inside = r2 <= 1.0
    coordinates = np.stack([u, v, np.sqrt(np.clip(1.0 - r2, 0.0, None))], axis=-1)[inside]
    return coordinates, inside, r2[inside] > 0.985


def paint(coverage: np.ndarray, colors: np.ndarray, inside: np.ndarray, rim: np.ndarray, supersample: int) -> np.ndarray:
    """An RGB image of the disk: each pixel takes the colour of the last body covering it.
    coverage is (bodies, pixels) boolean; the image is box-filtered by the supersampling."""
    covered = coverage.any(axis=0)
    last = coverage.shape[0] - 1 - coverage[::-1].argmax(axis=0)
    disk = np.where(covered[:, None], colors[last], np.where(rim[:, None], to_rgb(RIM), to_rgb(DISK)))
    image = np.empty(inside.shape + (3,))
    image[:] = to_rgb(BACKGROUND)
    image[inside] = disk
    size = inside.shape[0] // supersample
    image = image.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))
    return (image * 255.0).round().astype(np.uint8)


def rgb(colors: list[str]) -> np.ndarray:
    """RGB rows (bodies, 3) of matplotlib colour names."""
    return np.array([to_rgb(color) for color in colors])


def hues(fractions: np.ndarray) -> np.ndarray:
    """RGB rows (bodies, 3) around the hue circle."""
    return plt.get_cmap("hsv")(fractions)[:, :3]


# --- S² -------------------------------------------------------------------------------
def hemisphere_frames(surfaces: S2.Quadric, colors: np.ndarray, resolution: int, supersample: int) -> list[np.ndarray]:
    """One frame per row of world forms: a pixel of the front hemisphere is a point of S², and it
    belongs to a body where the body's primal form is negative. The spherical quadric is a cone
    through the origin, so the front hemisphere already shows both halves."""
    coordinates, inside, rim = hemisphere(resolution * supersample)
    pixels = S2.mv.yz * coordinates[:, 0] + S2.mv.zx * coordinates[:, 1] + S2.mv.xy * coordinates[:, 2]
    return [paint((pixels & bodies[:, None](pixels)) < 0.0, colors, inside, rim, supersample) for bodies in surfaces]


def show_hemisphere(ax, frame: np.ndarray, title: str) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.imshow(frame, extent=(-1.0, 1.0, -1.0, 1.0), interpolation="none")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)


def style_invariants(ax, title: str, ylabel: str, legend: str) -> None:
    ax.set_facecolor(DISK)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=9)
    ax.set_xlabel("Simulation Time (seconds)", color="#cbd5e1", fontsize=9)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color(RIM)
    ax.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
    ax.legend(loc=legend, facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=8.5)


def normalized_invariants(trajectory: S2.Trajectory) -> tuple[np.ndarray, np.ndarray]:
    energy, momentum = trajectory.energy.to_array(), trajectory.momentum.norm().to_array()
    return energy / energy[0], momentum / momentum[0]


def draw_tumbling(trajectory: S2.Trajectory, colors: np.ndarray, dt: float) -> plt.Figure:
    """The start and the first two flips of the tumbling oval, and its body rates with the invariants."""
    rate = trajectory.rate[:, 0].cast(S2.ga.subspace("yz zx xy")).kernel
    flips = np.nonzero(np.diff(np.sign(rate[:, 0])))[0]
    keys = [0, int(flips[0]), int(flips[1])]
    titles = [f"Frame {keys[0]}: Initial State (Intermediate Spin)",
              f"Frame {keys[1]}: First Inversion (180° Dzhanibekov Flip)",
              f"Frame {keys[2]}: Second Inversion (Return Flip)"]
    fig = plt.figure(figsize=(14, 11), dpi=110)
    fig.patch.set_facecolor(BACKGROUND)
    for index, (frame, title) in enumerate(zip(hemisphere_frames(trajectory.surfaces[keys], colors, 240, 4), titles)):
        show_hemisphere(fig.add_subplot(2, 2, index + 1), frame, title)

    ax = fig.add_subplot(2, 2, 4)
    time = np.arange(rate.shape[0]) * dt
    energy, momentum = normalized_invariants(trajectory)
    ax.plot(time, rate[:, 0], color="#38bdf8", linewidth=2.0, label="ω_yz (Intermediate Axis)")
    ax.plot(time, rate[:, 1], color="#f43f5e", linewidth=1.5, linestyle="--", label="ω_zx (Major Axis)")
    ax.plot(time, rate[:, 2], color="#fbbf24", linewidth=1.5, linestyle=":", label="ω_xy (Polar Axis)")
    ax.plot(time, energy, color="#34d399", linewidth=2.0, alpha=0.8, label="Energy (norm)")
    ax.plot(time, momentum, color="#a855f7", linewidth=2.0, linestyle="-.", alpha=0.8, label="Momentum (norm)")
    style_invariants(ax, "Periodic Dzhanibekov Flips & Conservation", "Body Angular Velocity & Normalized Invariants", "upper right")
    fig.tight_layout()
    return fig


def draw_collisions(trajectory: S2.Trajectory, colors: np.ndarray, dt: float) -> plt.Figure:
    """Three moments of a colliding population, and its conserved energy and momentum."""
    frames = trajectory.energy.shape[0]
    keys = [0, frames // 3, frames - 1]
    titles = [f"Frame {keys[0]}: Initial State", f"Frame {keys[1]}: Mid-Orbit Trajectories",
              f"Frame {keys[2]}: Post-Impact Ricochets / Rebounds"]
    fig = plt.figure(figsize=(14, 11), dpi=110)
    fig.patch.set_facecolor(BACKGROUND)
    for index, (frame, title) in enumerate(zip(hemisphere_frames(trajectory.surfaces[keys], colors, 240, 4), titles)):
        show_hemisphere(fig.add_subplot(2, 2, index + 1), frame, title)

    ax = fig.add_subplot(2, 2, 4)
    time = np.arange(frames) * dt
    energy, momentum = normalized_invariants(trajectory)
    ax.plot(time, energy, color="#38bdf8", linewidth=2.0, label="Total Kinetic Energy (normalized)")
    ax.plot(time, momentum, color="#34d399", linewidth=2.0, linestyle="--", label="Total Angular Momentum (normalized)")
    ax.set_ylim(0.95, 1.05)
    style_invariants(ax, "Physical Invariants & Conservation on S²", "Normalized Invariant Value", "lower left")
    fig.tight_layout()
    return fig


# --- S³ -------------------------------------------------------------------------------
FOV = np.radians(120.0)                            # a wide pinhole view from the eye


def s3_frames(trajectory: S3.Trajectory, colors: np.ndarray, eye: S3.Motor, light: S3.Point,
              shape: tuple[int, int], supersample: int) -> list[np.ndarray]:
    """Every frame traced from the eye: each ellipsoid projected onto the image sphere, lit by the light."""
    chart = raytracer.pixel_chart(FOV, (shape[0] * supersample, shape[1] * supersample))
    return [(trace(eye, surfaces, colors, light, chart, shape, supersample) * 255).astype(np.uint8)
            for surfaces in trajectory.surfaces]


def draw_last_frame(frames: list[np.ndarray], bodies: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.imshow(frames[-1], interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"{bodies} ellipsoids on the 3-sphere after {len(frames)} frames")
    return fig
