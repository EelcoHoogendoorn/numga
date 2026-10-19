"""Drawing for neutrino flavor oscillations: the isospin vector's path on the flavor sphere with the plane it
turns in, beside the flavor probabilities along the way."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.animation import capture
from examples.quantum.neutrino_isospin import core

# The points along each great circle drawn, and the half width of the sphere's view.
CIRCLE_POINTS = 80
LIMIT = 1.3


# --- plumbing -------------------------------------------------------------------------
def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x, y and z coordinates of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def great_circle(plane: core.Bivector) -> np.ndarray:
    """The great circle of the unit sphere in a plane through the centre `[points, 3]`."""
    normal = coordinates(plane.dual())
    normal = normal / np.linalg.norm(normal)
    # Two unit directions in the plane: square to the normal and to its smallest coordinate axis, and square to both.
    across = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    across = across / np.linalg.norm(across)
    phi = np.linspace(0.0, 2.0 * np.pi, CIRCLE_POINTS)[:, None]
    return np.cos(phi) * across + np.sin(phi) * np.cross(normal, across)


def electron_probability(trajectory: core.Vector) -> np.ndarray:
    """The electron flavor's probability along a path of isospin vectors."""
    return (1.0 + coordinates(trajectory)[..., 2]) * 0.5


def draw_sphere(ax, trajectory: core.Vector, start_plane: core.Bivector, end_plane: core.Bivector) -> None:
    """The flavor sphere, its flavors at the poles, the isospin vector's path, and the plane it turns in at the
    start, dotted, and at the end."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 20))
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#cbd5e1", linewidth=0.4, alpha=0.35)
    for axis in np.eye(3):
        ax.plot(*np.stack([-1.2 * axis, 1.2 * axis]).T, color="#94a3b8", linestyle=":", linewidth=0.8)
    ax.scatter([0, 0], [0, 0], [1.0, -1.0], color=["#0284c7", "#e11d48"], s=30, zorder=5)
    ax.text(0, 0, 1.25, r"$\nu_e$", ha="center", va="center", fontsize=11, fontweight="bold", color="#0284c7")
    ax.text(0, 0, -1.25, r"$\nu_\mu$", ha="center", va="center", fontsize=11, fontweight="bold", color="#e11d48")
    path = coordinates(trajectory)                                           # [steps, 3]
    ax.plot(*path.T, color="#6366f1", linewidth=1.8)
    ax.scatter(*path[[0, -1]].T, color=["#10b981", "#f59e0b"], s=40, zorder=6)
    ax.plot(*great_circle(start_plane).T, color="#1e293b", linewidth=1.4, linestyle=":")
    ax.plot(*great_circle(end_plane).T, color="#1e293b", linewidth=1.8)
    ax.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT), zlim=(-LIMIT, LIMIT))
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.set_axis_off()


def draw_probabilities(ax, distances: np.ndarray, electron: np.ndarray) -> None:
    """The electron and muon flavors' probabilities along the distance, against zero, a half and one."""
    ax.plot(distances, electron, color="#0284c7", linewidth=2.0, label=r"$P(\nu_e)$")
    ax.plot(distances, 1 - electron, color="#e11d48", linewidth=1.8, linestyle="--", label=r"$P(\nu_\mu)$")
    for level, style in ((0.0, "-"), (0.5, ":"), (1.0, "-")):
        ax.axhline(level, color="#cbd5e1", linestyle=style, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set(xticks=[], yticks=[], ylim=(-0.05, 1.05), xlim=(distances[0], distances[-1]))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)


def panels() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    figure = plt.figure(figsize=(10, 4.5), dpi=120)
    return figure, figure.add_subplot(1, 2, 1, projection="3d"), figure.add_subplot(1, 2, 2)


def draw_precession(states: Iterable[core.Vector], plane: core.Bivector, distances: np.ndarray) -> plt.Figure:
    """The isospin vector turning in one plane, and the flavor probabilities oscillating."""
    trajectory = stack(list(states), axis=0)                                 # [steps] Vector
    figure, sphere, probabilities = panels()
    draw_sphere(sphere, trajectory, plane, plane)
    draw_probabilities(probabilities, distances, electron_probability(trajectory))
    figure.tight_layout()
    return figure


def draw_decoherence(states: Iterable[core.Vector], plane: core.Bivector, distances: np.ndarray, settled: float) -> plt.Figure:
    """The isospin vector spiralling inward, and the flavor probabilities settling where the dashed line says."""
    trajectory = stack(list(states), axis=0)                                 # [steps] Vector
    figure, sphere, probabilities = panels()
    draw_sphere(sphere, trajectory, plane, plane)
    draw_probabilities(probabilities, distances, electron_probability(trajectory))
    probabilities.axhline(settled, color="#64748b", linestyle="--", linewidth=1.2)
    figure.tight_layout()
    return figure


def draw_adiabatic(steps: Iterable[tuple[core.Vector, core.Bivector]], distances: np.ndarray, potentials: np.ndarray) -> plt.Figure:
    """The isospin vector following the plane as it tilts, and the flavor converting as the potential, dotted,
    falls away."""
    pairs = list(steps)
    trajectory = stack([state for state, _ in pairs], axis=0)               # [steps] Vector
    figure, sphere, probabilities = panels()
    draw_sphere(sphere, trajectory, pairs[0][1], pairs[-1][1])
    draw_probabilities(probabilities, distances, electron_probability(trajectory))
    probabilities.plot(distances, potentials / potentials.max(), color="#94a3b8", linestyle=":", linewidth=1.2)
    figure.tight_layout()
    return figure


def animate_precession(states: Iterable[core.Vector], plane: core.Bivector, distances: np.ndarray, n_frames: int) -> list[np.ndarray]:
    """The isospin vector's path and the flavor probabilities growing along the distance, frame by frame."""
    trajectory = stack(list(states), axis=0)                                 # [steps] Vector
    electron = electron_probability(trajectory)
    figure, sphere, probabilities = panels()
    frames = []
    for end in np.linspace(2, len(distances), n_frames, dtype=int):
        sphere.cla()
        probabilities.cla()
        draw_sphere(sphere, trajectory[:end], plane, plane)
        draw_probabilities(probabilities, distances[:end], electron[:end])
        probabilities.set_xlim(distances[0], distances[-1])
        frames.append(capture(figure))
    plt.close(figure)
    return frames
