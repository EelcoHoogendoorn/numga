"""Active bonds, cyclic work and waves in a planar elastic lattice."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon

from numga import stack
from examples.util.animation import capture
from examples.mechanics.odd_elasticity import core

BACKGROUND = "#fbfaf7"
BOND = "#267c8b"
FORCE = "#db663d"
ANCHOR = "#35434d"
REFERENCE = "#d9dcda"
SWEPT = "#f6d9c9"
TRACK = "#f3b83f"
STRAIN = LinearSegmentedColormap.from_list("bond_extension", ["#4775bd", "#32666d", "#ed8b3d"])


# --- plumbing -------------------------------------------------------------------------
def _coordinates(points: core.Displacement) -> np.ndarray:
    """Positions read by pairing with the lines across the axes; sites become a batch axis."""
    lines = stack((core.mv.x, core.mv.y))
    return (lines & points.batch()[..., None]).to_array()




def _bounds(coordinates: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    low = coordinates.reshape(-1, 2).min(axis=0)
    high = coordinates.reshape(-1, 2).max(axis=0)
    padding = margin * np.max(high - low)
    return low - padding, high + padding


def _scene(figure: plt.Figure, bounds: tuple[np.ndarray, np.ndarray], position: tuple[float, float, float, float]) -> plt.Axes:
    ax = figure.add_axes(position, facecolor=BACKGROUND)
    low, high = bounds
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
    ax.set_axis_off()
    return ax


def bond_frames(
    anchor: core.Displacement, reference: core.Displacement, path: core.Displacement,
    states: Iterable[tuple[core.Displacement, core.Force]], force_scale: float,
) -> Iterator[np.ndarray]:
    """An endpoint follows its closed path, the area it has swept about its rest position shaded, while
    the bond's force changes."""
    fixed = _coordinates(anchor)
    rest = _coordinates(reference)
    loop = _coordinates(path)
    # Each force drawn along its metric direction, its dual, from the bond's moving end.
    positions, forces = (stack([value.batch() for value in values]) for values in zip(*states))   # [steps] Displacement-, Force+
    arrows = force_scale * forces.dual()                                     # [steps] Displacement-
    endpoints, offsets = _coordinates(positions), _coordinates(arrows)       # [steps, 2], [steps, 2]
    bounds = _bounds(np.concatenate((fixed[None], rest[None], loop, _coordinates(positions + arrows))), 0.08)

    figure = plt.figure(figsize=(6.2, 2.9), dpi=50, facecolor=BACKGROUND)
    ax = _scene(figure, bounds, (0.02, 0.02, 0.96, 0.96))
    swept = Polygon(np.stack((rest, rest)), closed=True, facecolor=SWEPT, edgecolor="none", zorder=0)
    ax.add_patch(swept)
    ax.plot(*np.stack((fixed, rest)).T, color=REFERENCE, linewidth=2, linestyle="--")
    ax.plot(*loop.T, color=REFERENCE, linewidth=1.5)
    ax.plot(*fixed, "o", color=ANCHOR, markersize=9)
    ax.plot(*rest, "+", color=ANCHOR, markersize=8, markeredgewidth=1)
    bond, = ax.plot([], [], color=BOND, linewidth=4, solid_capstyle="round")
    endpoint, = ax.plot([], [], "o", color=BOND, markersize=9)
    trail, = ax.plot([], [], color=BOND, linewidth=1.4, alpha=0.55)
    arrow = ax.quiver(*endpoints[0], 0, 0, angles="xy", scale_units="xy", scale=1,
                      color=FORCE, width=0.011, headwidth=4, headlength=5, zorder=5)

    for i, (position, offset) in enumerate(zip(endpoints, offsets)):
        swept.set_xy(np.concatenate((rest[None], endpoints[:i + 1])))
        bond.set_data(*np.stack((fixed, position)).T)
        endpoint.set_data([position[0]], [position[1]])
        trail.set_data(*endpoints[:i + 1].T)
        arrow.set_offsets(position)
        arrow.set_UVC(*offset)
        yield capture(figure)
    plt.close(figure)


def patch_frames(
    reference: core.Displacement, mesh: core.Mesh,
    states: Iterable[tuple[core.Displacement, core.Force]], force_scale: float,
) -> Iterator[np.ndarray]:
    """A deforming patch pushes on its boundary around its closed cycle."""
    rest = _coordinates(reference)
    boundary = mesh.exterior(0)
    # Each force on the boundary drawn along its metric direction, its dual, from its particle.
    places, forces = (stack([value.batch() for value in values]) for values in zip(*states))   # [steps, vertices] Displacement-, Force+
    arrows = force_scale * forces[:, boundary].dual()                        # [steps, boundary] Displacement-
    positions, offsets = _coordinates(places), _coordinates(arrows)          # [steps, vertices, 2], [steps, boundary, 2]
    tips = _coordinates(places[:, boundary] + arrows)                        # [steps, boundary, 2]
    bounds = _bounds(np.concatenate((rest, positions.reshape(-1, 2), tips.reshape(-1, 2))), 0.055)

    figure = plt.figure(figsize=(5.2, 3.6), dpi=50, facecolor=BACKGROUND)
    ax = _scene(figure, bounds, (0.02, 0.02, 0.96, 0.96))
    ax.add_collection(LineCollection(rest[mesh.edges], colors=REFERENCE, linewidths=1))
    bonds = LineCollection([], colors=BOND, linewidths=1.7)
    ax.add_collection(bonds)
    nodes = ax.scatter(*rest.T, s=13, c=ANCHOR, linewidths=0, zorder=3)
    quivers = ax.quiver(*rest[boundary].T, *np.zeros_like(rest[boundary]).T,
                        angles="xy", scale_units="xy", scale=1, color=FORCE,
                        width=0.005, headwidth=3.8, headlength=4.8, zorder=4)

    for position, offset in zip(positions, offsets):
        bonds.set_segments(position[mesh.edges])
        nodes.set_offsets(position)
        quivers.set_offsets(position[boundary])
        quivers.set_UVC(*offset.T)
        yield capture(figure)
    plt.close(figure)


def wave_frames(
    reference: core.Displacement, mesh: core.Mesh,
    states: Iterable[tuple[core.Displacement, core.Scalar]],
    displacement_scale: float, extension_scale: float, tracked_index: int,
) -> Iterator[np.ndarray]:
    """Moving bonds coloured by signed extension, with one particle's short trajectory marked."""
    rest = _coordinates(reference)
    figure = plt.figure(figsize=(6.5, 3.7), dpi=50, facecolor=BACKGROUND)
    ax = _scene(figure, _bounds(rest, 0.035), (0.015, 0.015, 0.97, 0.97))
    ax.add_collection(LineCollection(rest[mesh.edges], colors=REFERENCE, linewidths=0.65))
    bonds = LineCollection([], linewidths=1.15, cmap=STRAIN, norm=Normalize(-extension_scale, extension_scale))
    ax.add_collection(bonds)
    nodes = ax.scatter(*rest.T, s=3, c=ANCHOR, linewidths=0, zorder=3)
    trail, = ax.plot([], [], color=TRACK, linewidth=2.2, zorder=4)
    tracked, = ax.plot([], [], "o", color=TRACK, markersize=7, markeredgecolor=ANCHOR,
                       markeredgewidth=0.7, zorder=5)

    n_trail_frames = 33
    history = []
    for positions, extensions in states:
        # The motion from the rest positions, exaggerated.
        position = _coordinates(reference + displacement_scale * (positions - reference))
        bonds.set_segments(position[mesh.edges])
        bonds.set_array(extensions.batch().to_array())
        nodes.set_offsets(position)
        history.append(position[tracked_index])
        trajectory = np.stack(history[-n_trail_frames:])
        trail.set_data(*trajectory.T)
        tracked.set_data([position[tracked_index, 0]], [position[tracked_index, 1]])
        yield capture(figure)
    plt.close(figure)
