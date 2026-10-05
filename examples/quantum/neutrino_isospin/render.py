"""Rendering for neutrino flavor isospin and MSW matter effect.

All array extraction (`.kernel`, `.to_array()`) happens strictly JIT inside this module
at the visualization boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.algebras import VGA3D as ga
from examples.animation import capture
from examples.quantum.neutrino_isospin.core import Bivector, Vector


# --- plumbing -------------------------------------------------------------------------
def euclidean(v: Vector) -> np.ndarray:
    """Extract coordinates from a batch of vectors in standard Euclidean basis (x, y, z)."""
    return v.cast(ga.subspace("x y z")).kernel


def plane_coordinates(plane: Bivector) -> np.ndarray:
    """Extract coordinates from a batch of bivectors in the basis (xy, xz, yz)."""
    return plane.cast(ga.subspace("xy xz yz")).kernel


def plane_circle(b_coords: np.ndarray, num_points: int = 80) -> np.ndarray:
    """Great circle on the unit sphere corresponding to the bivector plane."""
    normal = np.array([b_coords[2], -b_coords[1], b_coords[0]])
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.zeros((num_points, 3))
    n = normal / norm
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    phi = np.linspace(0.0, 2.0 * np.pi, num_points)
    return np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v


def draw_isospin_sphere(
    ax,
    trajectory: Vector,
    plane: Bivector,
    title: str = "Flavor Isospin Sphere",
) -> None:
    """Render the 3D flavor isospin sphere with rotation trajectory and Hamiltonian plane."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="#cbd5e1", linewidth=0.4, alpha=0.35)

    # Coordinate cross axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color="#94a3b8", linestyle=":", linewidth=0.8)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color="#94a3b8", linestyle=":", linewidth=0.8)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color="#94a3b8", linestyle=":", linewidth=0.8)

    # Flavor pole markers
    ax.scatter([0], [0], [1.0], color="#0284c7", s=30, zorder=5)
    ax.scatter([0], [0], [-1.0], color="#e11d48", s=30, zorder=5)
    ax.text(0, 0, 1.25, r"$\nu_e$", ha="center", va="center", fontsize=11, fontweight="bold", color="#0284c7")
    ax.text(0, 0, -1.25, r"$\nu_\mu$", ha="center", va="center", fontsize=11, fontweight="bold", color="#e11d48")

    # Equator circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), color="#94a3b8", linestyle="--", linewidth=0.6, alpha=0.5)

    # JIT coordinate readout with explicit blade layout casting
    coords = euclidean(trajectory)
    if len(coords) > 1:
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="#6366f1", linewidth=1.8)
        ax.scatter([coords[0, 0]], [coords[0, 1]], [coords[0, 2]], color="#10b981", s=40, zorder=6)
        ax.scatter([coords[-1, 0]], [coords[-1, 1]], [coords[-1, 2]], color="#f59e0b", s=40, zorder=6)

    # Hamiltonian plane great circle
    b_coords = plane_coordinates(plane)
    if len(b_coords.shape) == 1:
        circle = plane_circle(b_coords)
        ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="#1e293b", linewidth=1.8, linestyle="-")
    elif len(b_coords) > 1:
        c_start = plane_circle(b_coords[0])
        c_end = plane_circle(b_coords[-1])
        ax.plot(c_start[:, 0], c_start[:, 1], c_start[:, 2], color="#1e293b", linewidth=1.4, linestyle=":")
        ax.plot(c_end[:, 0], c_end[:, 1], c_end[:, 2], color="#1e293b", linewidth=1.8, linestyle="-")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=10)


def draw_probabilities(
    ax,
    distances: np.ndarray,
    pe: np.ndarray,
    pmu: np.ndarray,
    title: str = "Flavor Probabilities",
) -> None:
    """Render spatial flavor transition curves P(nu_e) and P(nu_mu)."""
    ax.plot(distances, pe, color="#0284c7", linewidth=2.0, label=r"$P(\nu_e)$")
    ax.plot(distances, pmu, color="#e11d48", linewidth=1.8, linestyle="--", label=r"$P(\nu_\mu)$")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(distances[0], distances[-1])

    ax.axhline(0.0, color="#cbd5e1", linestyle="-", linewidth=0.8)
    ax.axhline(0.5, color="#cbd5e1", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(1.0, color="#cbd5e1", linestyle="-", linewidth=0.8)

    ax.text(distances[0], -0.02, "0", ha="left", va="top", fontsize=9, color="#64748b")
    ax.text(distances[-1], -0.02, "Distance", ha="right", va="top", fontsize=9, color="#64748b")
    ax.text(distances[0], 0.5, "0.5", ha="right", va="center", fontsize=8.5, color="#94a3b8")
    ax.text(distances[0], 1.0, "1.0", ha="right", va="center", fontsize=8.5, color="#94a3b8")

    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(fontsize=8, loc="upper right", frameon=False)


# --- drawing --------------------------------------------------------------------------
def draw_precession(
    title: str,
    states: Iterable[Vector],
    plane: Bivector,
    distances: np.ndarray,
) -> plt.Figure:
    """Consume yielded states and render the 3D sphere and flavor oscillation curves."""
    trajectory = stack(list(states), axis=0)
    z_coords = trajectory.cast(ga.subspace("z")).kernel.squeeze()
    pe = (1.0 + z_coords) * 0.5
    pmu = (1.0 - z_coords) * 0.5

    fig = plt.figure(figsize=(10, 4.5), dpi=120)
    ax_sphere = fig.add_subplot(1, 2, 1, projection="3d")
    ax_prob = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.90, wspace=0.15)

    draw_isospin_sphere(ax_sphere, trajectory, plane, title=title)
    draw_probabilities(ax_prob, distances, pe, pmu, title="Flavor Probabilities")

    fig.tight_layout()
    return fig


def draw_decoherence(
    title: str,
    states: Iterable[Vector],
    plane: Bivector,
    distances: np.ndarray,
    asymptotic_pe: float,
) -> plt.Figure:
    """Consume yielded states and render spiraling decoherence trajectory and damped oscillations."""
    trajectory = stack(list(states), axis=0)
    z_coords = trajectory.cast(ga.subspace("z")).kernel.squeeze()
    pe = (1.0 + z_coords) * 0.5
    pmu = (1.0 - z_coords) * 0.5

    fig = plt.figure(figsize=(10, 4.5), dpi=120)
    ax_sphere = fig.add_subplot(1, 2, 1, projection="3d")
    ax_prob = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.90, wspace=0.15)

    draw_isospin_sphere(ax_sphere, trajectory, plane, title=title)
    draw_probabilities(ax_prob, distances, pe, pmu, title="Wavepacket Decoherence")

    ax_prob.axhline(asymptotic_pe, color="#64748b", linestyle="--", linewidth=1.2, label=f"Asymptotic {asymptotic_pe:.3f}")
    ax_prob.legend(fontsize=8, loc="upper right", frameon=False)

    fig.tight_layout()
    return fig


def draw_adiabatic(
    title: str,
    steps: Iterable[tuple[Vector, Bivector]],
    distances: np.ndarray,
    v_profile: np.ndarray,
) -> plt.Figure:
    """Consume yielded (state, plane) steps and render the 3D sphere and MSW conversion curves."""
    items = list(steps)
    trajectory = stack([s for s, _ in items], axis=0)
    planes = stack([p for _, p in items], axis=0)

    z_coords = trajectory.cast(ga.subspace("z")).kernel.squeeze()
    pe = (1.0 + z_coords) * 0.5
    pmu = (1.0 - z_coords) * 0.5

    fig = plt.figure(figsize=(10, 4.5), dpi=120)
    ax_sphere = fig.add_subplot(1, 2, 1, projection="3d")
    ax_prob = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.90, wspace=0.15)

    draw_isospin_sphere(ax_sphere, trajectory, planes, title=title)
    draw_probabilities(ax_prob, distances, pe, pmu, title="Flavor Conversion")
    ax_prob.plot(distances, v_profile / v_profile.max(), color="#94a3b8", linestyle=":", linewidth=1.2, label=r"$N_e(r)$ (norm)")
    ax_prob.legend(fontsize=8, loc="upper right", frameon=False)

    fig.tight_layout()
    return fig


def animate_precession(
    title: str,
    states: Iterable[Vector],
    plane: Bivector,
    distances: np.ndarray,
    steps: int = 50,
) -> list[np.ndarray]:
    """Capture animation frames of isospin vector orbiting on the sphere."""
    trajectory = stack(list(states), axis=0)
    z_coords = trajectory.cast(ga.subspace("z")).kernel.squeeze()
    pe = (1.0 + z_coords) * 0.5
    pmu = (1.0 - z_coords) * 0.5

    fig = plt.figure(figsize=(9.5, 4.5), dpi=100)
    ax_sphere = fig.add_subplot(1, 2, 1, projection="3d")
    ax_prob = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.90, wspace=0.15)

    n_total = len(distances)
    indices = np.linspace(2, n_total, steps, dtype=int)
    frames = []

    for idx in indices:
        ax_sphere.cla()
        ax_prob.cla()

        sub_traj = trajectory[:idx]
        sub_dist = distances[:idx]
        sub_pe = pe[:idx]
        sub_pmu = pmu[:idx]

        draw_isospin_sphere(ax_sphere, sub_traj, plane, title=title)
        draw_probabilities(ax_prob, sub_dist, sub_pe, sub_pmu, title="Flavor Probabilities")
        ax_prob.set_xlim(distances[0], distances[-1])

        frames.append(capture(fig))

    plt.close(fig)
    return frames
