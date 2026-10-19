"""The cube's faces, oriented patches and volume."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.mechanics.area_transport import core

FACES = np.array([(4, 5, 7, 6), (2, 3, 7, 6), (1, 3, 7, 5),
                  (0, 2, 3, 1), (0, 1, 5, 4), (0, 4, 6, 2)])
COLOURS = ("#2563eb", "#0d9488", "#f97316")
LIMIT = 0.95
ARROW_SCALE = 0.36


# --- plumbing -------------------------------------------------------------------------
def cube(ax: Axes, vertices: core.Point, centres: core.Point, patches: core.Patch) -> None:
    corners = vertices.cast(core.ga.subspace("yz zx xy")).kernel
    starts = centres.cast(core.ga.subspace("yz zx xy")).kernel
    # Each patch drawn as its normal, as long as its area.
    arrows = patches.cast(core.ga.subspace("x y z")).kernel
    ax.add_collection3d(Poly3DCollection(
        corners[FACES], facecolors=COLOURS * 2, alpha=0.20,
        edgecolors="#334155", linewidths=1.3,
    ))
    ax.quiver(*starts.T, *(arrows * ARROW_SCALE).T, color=COLOURS * 2,
              arrow_length_ratio=0.2, linewidth=2)
    ax.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT), zlim=(-LIMIT, LIMIT))
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.view_init(elev=23, azim=-57)
    ax.set_axis_off()


def measures(ax: Axes, patches: core.Patch, volume: core.Scalar) -> None:
    areas = patches[:3].norm().kernel[..., 0]
    values = np.concatenate([areas, volume.kernel])
    ax.bar(np.arange(len(values)), values, color=(*COLOURS, "#334155"), width=0.58)
    ax.axhline(1, color="#94a3b8", linestyle=":", linewidth=1)
    ax.set(xticks=[], ylim=(0, 1.35))
    ax.spines[["top", "right"]].set_visible(False)


def comparison(vertices: core.Point, centres: core.Point, patches: core.Patch, volume: core.Scalar) -> plt.Figure:
    """One column per deformation: the cube with its patches above, and below, the areas of the faces
    normal to x, y and z and the volume, against the unit cube's."""
    figure = plt.figure(figsize=(12, 6.5), layout="constrained")
    count = vertices.shape[0]
    for i in range(count):
        cube(figure.add_subplot(2, count, i + 1, projection="3d"), vertices[i], centres[i], patches[i])
        measures(figure.add_subplot(2, count, count + i + 1), patches[i], volume[i])
    return figure


def frame(vertices: core.Point, centres: core.Point, patches: core.Patch, volume: core.Scalar) -> np.ndarray:
    figure = plt.figure(figsize=(9, 4.3), dpi=85, layout="constrained")
    cube(figure.add_subplot(1, 2, 1, projection="3d"), vertices, centres, patches)
    measures(figure.add_subplot(1, 2, 2), patches, volume)
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(frames: Iterable[tuple[core.Point, core.Point, core.Patch, core.Scalar]]) -> list[np.ndarray]:
    return [frame(vertices, centres, patches, volume) for vertices, centres, patches, volume in frames]
