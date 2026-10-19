"""Drawing for the stressed cube: the cube seen from each frame with its face tractions, and Mohr's
circles."""

from __future__ import annotations

from collections.abc import Iterable

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
LIMIT = 0.75
# The principal directions are drawn out to this distance from the centre, just past the cube.
REACH = 0.75


def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x, y and z coordinates of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def draw_cube(ax, corners: core.Vector, centres: core.Vector, normal: core.Vector, shear: core.Vector,
              principal: core.Vector) -> None:
    """The cube seen from one frame: the undeformed cube dashed, the deformed one shaded, the normal
    traction on each face in blue and the shear in red, and the principal directions dotted."""
    ghost, corners = coordinates(core.corners), coordinates(corners)
    centres = coordinates(centres)
    normal = coordinates(normal) * NORMAL_SCALE
    shear = coordinates(shear) * SHEAR_SCALE
    principal = coordinates(principal)
    for a, b in EDGES:
        ax.plot(*zip(ghost[a], ghost[b]), color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.add_collection3d(Poly3DCollection(corners[FACES], alpha=0.25, facecolor="#38bdf8", edgecolor="#0284c7", linewidth=1.2))
    for a, b in EDGES:
        ax.plot(*zip(corners[a], corners[b]), color="#0f172a", linewidth=1.6)
    ax.quiver(*centres.T, *normal.T, color="#0284c7", arrow_length_ratio=0.25, linewidth=1.8)
    ax.quiver(*centres.T, *shear.T, color="#e11d48", arrow_length_ratio=0.28, linewidth=2.4)
    for direction in principal:
        ax.plot(*(REACH * np.stack([-direction, direction])).T, color="#10b981", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_zlim(-LIMIT, LIMIT)
    ax.set_box_aspect([1, 1, 1], zoom=1.25)
    ax.view_init(elev=22, azim=25)
    ax.set_axis_off()


def draw_mohr(ax, values: core.Scalar, normal: core.Vector, shear: core.Vector) -> None:
    """Mohr's circles of the principal stresses, and for each frame `[frames]` the diameter joining the
    normal and shear traction on the faces facing x, filled, and y, open, in the frame's colour."""
    stresses = np.sort(values.to_array())
    angle = np.linspace(0, 2 * np.pi, 200)
    for low, high, style in [(0, 2, "-"), (0, 1, ":"), (1, 2, ":")]:
        centre, radius = (stresses[low] + stresses[high]) / 2, (stresses[high] - stresses[low]) / 2
        ax.plot(centre + radius * np.cos(angle), radius * np.sin(angle), color="#1e293b" if style == "-" else "#94a3b8",
                linestyle=style, linewidth=1.8 if style == "-" else 1.1)
    ax.scatter(stresses, np.zeros(3), color="#1e293b", s=18, zorder=3)
    # The outer two outside the circle that passes through them, the middle one below its dot.
    for stress, name, shift, align in zip(stresses, ["$\\sigma_1$", "$\\sigma_2$", "$\\sigma_3$"], (-5, 0, 5),
                                          ("right", "center", "left")):
        ax.annotate(name, (stress, 0), xytext=(shift, -5), textcoords="offset points", ha=align, va="top",
                    fontsize=9, color="#1e293b")
    normal, shear = coordinates(normal), coordinates(shear)                  # [frames, 6, 3]
    for frame, colour in zip(range(len(normal)), COLOURS * len(normal)):
        # The face facing x: its normal traction along x and its shear along y; the face facing y, with
        # the opposite shear.
        x_face = (normal[frame, 0, 0], shear[frame, 0, 1])
        y_face = (normal[frame, 1, 1], -shear[frame, 0, 1])
        ax.plot(*zip(x_face, y_face), color=colour, linewidth=1.6, zorder=4)
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


def draw_frames(views: tuple[core.Vector, core.Vector, core.Vector, core.Vector, core.Vector],
                values: core.Scalar) -> plt.Figure:
    """The cube seen from each frame `[frames]`, and Mohr's circles with each frame's diameter."""
    corners, centres, normal, shear, principal = views
    count = corners.shape[0]
    figure = plt.figure(figsize=(4.5 * (count + 1), 4.5), dpi=120)
    for frame in range(count):
        draw_cube(figure.add_subplot(1, count + 1, frame + 1, projection="3d"),
                  corners[frame], centres[frame], normal[frame], shear[frame], principal[frame])
    draw_mohr(figure.add_subplot(1, count + 1, count + 1), values, normal, shear)
    figure.tight_layout()
    return figure


def animate_turning(values: core.Scalar, frames: Iterable[tuple[core.Vector, core.Vector, core.Vector, core.Vector,
                                                                   core.Vector]]) -> list[np.ndarray]:
    """Each frame of the cube seen from a turning frame, and its diameter on Mohr's circles."""
    images = []
    for corners, centres, normal, shear, principal in frames:
        figure = plt.figure(figsize=(9.5, 4.5), dpi=100)
        draw_cube(figure.add_subplot(1, 2, 1, projection="3d"), corners, centres, normal, shear, principal)
        draw_mohr(figure.add_subplot(1, 2, 2), values, normal[None], shear[None])
        images.append(capture(figure))
        plt.close(figure)
    return images
