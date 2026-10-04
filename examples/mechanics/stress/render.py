"""Drawing for the stressed cube: the cube seen from each frame with its face tractions, and Mohr's
circles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.mechanics.stress import core

# The cube's edges and faces, by the corners' indices: corner 4 i + 2 j + k sits at the i-th, j-th and
# k-th of -0.5 and 0.5 along x, y and z.
EDGES = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
FACES = np.array([(4, 5, 7, 6), (0, 2, 3, 1), (2, 6, 7, 3), (0, 1, 5, 4), (1, 3, 7, 5), (0, 4, 6, 2)])
# The arrows' lengths per unit traction, the colours of the frames' diameters on Mohr's circle, and the
# half width of the cube's view.
NORMAL_SCALE, SHEAR_SCALE = 0.4, 0.5
COLOURS = ("#0284c7", "#10b981", "#e11d48")
LIMIT = 1.15


def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x, y and z coordinates of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def draw_cube(ax, views: core.Views, frame: int, title: str) -> None:
    """The cube seen from one frame: the undeformed cube dashed, the deformed one shaded, the normal
    traction on each face in blue and the shear in red, and the principal directions dotted."""
    ghost, corners = coordinates(core.corners), coordinates(views.corners[frame])
    centres = coordinates(views.centres[frame])
    normal = coordinates(views.normal[frame]) * NORMAL_SCALE
    shear = coordinates(views.shear[frame]) * SHEAR_SCALE
    principal = coordinates(views.principal[frame])
    for a, b in EDGES:
        ax.plot(*zip(ghost[a], ghost[b]), color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.add_collection3d(Poly3DCollection(corners[FACES], alpha=0.25, facecolor="#38bdf8", edgecolor="#0284c7", linewidth=1.2))
    for a, b in EDGES:
        ax.plot(*zip(corners[a], corners[b]), color="#0f172a", linewidth=1.6)
    ax.quiver(*centres.T, *normal.T, color="#0284c7", arrow_length_ratio=0.25, linewidth=1.8)
    ax.quiver(*centres.T, *shear.T, color="#e11d48", arrow_length_ratio=0.28, linewidth=2.4)
    for direction in principal:
        ax.plot(*np.stack([-direction, direction]).T, color="#10b981", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_zlim(-LIMIT, LIMIT)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title, fontsize=11, pad=10)
    ax.view_init(elev=22, azim=25)
    ax.set_axis_off()


def draw_mohr(ax, views: core.Views, values: core.Scalar, frames: list[int], labels: list[str]) -> None:
    """Mohr's circles of the principal stresses, and for each frame the diameter joining the normal
    and shear traction on the faces facing x and y."""
    stresses = np.sort(values.to_array())
    angle = np.linspace(0, 2 * np.pi, 200)
    for low, high, style in [(0, 2, "-"), (0, 1, ":"), (1, 2, ":")]:
        centre, radius = (stresses[low] + stresses[high]) / 2, (stresses[high] - stresses[low]) / 2
        ax.plot(centre + radius * np.cos(angle), radius * np.sin(angle), color="#1e293b" if style == "-" else "#94a3b8",
                linestyle=style, linewidth=1.8 if style == "-" else 1.1)
    for stress, name in zip(stresses, ["$\\sigma_1$", "$\\sigma_2$", "$\\sigma_3$"]):
        ax.text(stress, -0.07, name, ha="center", va="top", fontsize=9, color="#10b981")
    normal, shear = coordinates(views.normal), coordinates(views.shear)      # [frames, 6, 3]
    for frame, label, colour in zip(frames, labels, COLOURS * len(frames)):
        # The face facing x: its normal traction along x and its shear along y; the face facing y, with
        # the opposite shear.
        x_face = (normal[frame, 0, 0], shear[frame, 0, 1])
        y_face = (normal[frame, 1, 1], -shear[frame, 0, 1])
        ax.plot(*zip(x_face, y_face), color=colour, linewidth=1.6, zorder=4, label=label)
        ax.scatter(*x_face, color=colour, s=65, zorder=5)
        ax.scatter(*y_face, s=65, zorder=5, facecolors="none", edgecolors=colour, linewidth=1.6)
    centre, radius = (stresses[0] + stresses[2]) / 2, (stresses[2] - stresses[0]) / 2
    margin = 1.3 * radius
    ax.set_xlim(centre - margin, centre + margin)
    ax.set_ylim(-margin, margin)
    ax.axhline(0, color="#94a3b8", linewidth=0.9, zorder=2)
    ax.axvline(0, color="#94a3b8", linewidth=0.9, zorder=2)
    ax.text(centre + 0.95 * margin, 0.04, "$\\sigma$", ha="right", va="bottom", fontsize=11, color="#64748b")
    ax.text(0.04, 0.95 * margin, "$\\tau$", ha="left", va="top", fontsize=11, color="#64748b")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("Mohr's circles", fontsize=11, pad=10)


def draw_frames(views: core.Views, values: core.Scalar, titles: list[str]) -> plt.Figure:
    """The cube seen from each frame, and Mohr's circles with each frame's diameter."""
    count = len(titles)
    figure = plt.figure(figsize=(4.5 * (count + 1), 4.5), dpi=120)
    for frame, title in enumerate(titles):
        draw_cube(figure.add_subplot(1, count + 1, frame + 1, projection="3d"), views, frame, title)
    ax = figure.add_subplot(1, count + 1, count + 1)
    draw_mohr(ax, views, values, list(range(count)), titles)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    figure.tight_layout()
    return figure


def animate_turning(views: core.Views, values: core.Scalar) -> list[np.ndarray]:
    """Each frame: the cube seen from it, and its diameter on Mohr's circles."""
    images = []
    for frame in range(views.corners.shape[0]):
        figure = plt.figure(figsize=(9.5, 4.5), dpi=100)
        draw_cube(figure.add_subplot(1, 2, 1, projection="3d"), views, frame, "turning in the shear plane")
        draw_mohr(figure.add_subplot(1, 2, 2), views, values, [frame], ["the faces facing x and y"])
        images.append(capture(figure))
        plt.close(figure)
    return images
