"""Active bonds, cyclic work and waves in a planar elastic lattice, drawn on the GPU in the lattice's own algebra."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import concatenate, stack
from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import panel
from examples.mechanics.odd_elasticity import core

BACKGROUND = "#fbfaf7"
BOND, FORCE, ANCHOR, REFERENCE, SWEPT, TRACK = (
    to_rgb(colour) for colour in ("#267c8b", "#db663d", "#35434d", "#d9dcda", "#f6d9c9", "#f3b83f"))
STRAIN = LinearSegmentedColormap.from_list("bond_extension", ["#4775bd", "#32666d", "#ed8b3d"])


# --- plumbing -------------------------------------------------------------------------
def _bounds(points: list[core.Displacement], margin: float) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and the highest readings across and up of batches of positions, read by pairing with the lines
    across the axes, widened by a margin as a share of the larger span, `[2]` each; sites become a batch axis."""
    lines = stack((core.mv.x, core.mv.y))
    readings = concatenate([lines[:, None] & batch.batch().reshape(-1) for batch in points], axis=-1)   # [2, points] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)
    padding = margin * (high - low).max()
    return (low - padding).to_array(), (high + padding).to_array()


def _scene(figure: plt.Figure, bounds: tuple[np.ndarray, np.ndarray], position: tuple[float, float, float, float]) -> plt.Axes:
    ax = figure.add_axes(position, facecolor=BACKGROUND)
    low, high = bounds
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
    ax.set_axis_off()
    return ax


def _arrows(n_arrows: int) -> simplices.Topology:
    """Arrows from tails to tips listed after them: an edge from each of the first `n_arrows` points to the one
    `n_arrows` after it."""
    return simplices.Topology(2 * n_arrows, np.stack([np.arange(n_arrows), n_arrows + np.arange(n_arrows)], axis=-1))


def bond_frames(
    anchor: core.Displacement, reference: core.Displacement, path: core.Displacement,
    states: Iterable[tuple[core.Displacement, core.Force]], force_scale: float,
) -> Iterator[np.ndarray]:
    """An endpoint follows its closed path, the area it has swept about its rest position shaded, while
    the bond's force changes."""
    # Each force drawn along its metric direction, its dual, from the bond's moving end.
    positions, forces = (stack([value.batch() for value in values]) for values in zip(*states))   # [steps] Displacement-, Force+
    arrows = force_scale * forces.dual()                                     # [steps] Displacement-
    bounds = _bounds([anchor, reference, path, positions + arrows], 0.08)

    figure = plt.figure(figsize=(6.2, 2.9), dpi=50, facecolor=BACKGROUND)
    ax = _scene(figure, bounds, (0.02, 0.02, 0.96, 0.96))
    camera = panel(ax, core.mv.x, core.mv.y)
    # The swept area, a fan from the rest position over the path so far.
    swept_corners = concatenate([reference[None], positions])                # [steps + 1] Displacement-
    n_steps = len(positions)
    fan = np.stack([np.zeros(n_steps - 1, dtype=int), np.arange(1, n_steps), np.arange(2, n_steps + 1)], axis=-1)
    guides = [simplices.strokes(concatenate([anchor[None], reference[None]]), simplices.Topology.chains([2]), REFERENCE,
                                1.5, camera, dashes=(5.0, 3.0), cap=simplices.Cap.BUTT),
              simplices.strokes(path, simplices.Topology.chains([len(path)]), REFERENCE, 1.0, camera),
              simplices.markers(anchor[None], ANCHOR, 3.0, camera),
              # The rest position a ring: a dot of the background over a larger one.
              simplices.markers(reference[None], ANCHOR, 2.5, camera),
              simplices.markers(reference[None], to_rgb(BACKGROUND), 1.5, camera)]
    image = simplices.overlay(ax, guides)

    for step, (position, arrow) in enumerate(zip(positions, arrows)):
        bond = concatenate([anchor[None], position[None]])                   # [2] Displacement-
        force = concatenate([position[None], (position + arrow)[None]])      # [2] Displacement-
        image.set_data(simplices.composited([
            contours.picture(swept_corners, simplices.Topology(n_steps + 1, fan[:step]), np.zeros(n_steps + 1), SWEPT,
                             camera),
            *guides,
            simplices.strokes(positions[:step + 1], simplices.Topology.chains([step + 1]), BOND, 1.0, camera,
                              opacity=0.55),
            simplices.strokes(bond, simplices.Topology.chains([2]), BOND, 3.0, camera),
            simplices.markers(position[None], BOND, 3.0, camera),
            # The force's arrow, a stroke ending in a chevron.
            simplices.strokes(force, _arrows(1), FORCE, 3.0, camera, chevrons=np.ones(1), chevron=12.0)]))
        yield capture(figure)
    plt.close(figure)


def patch_frames(
    reference: core.Displacement, topology: core.Topology,
    states: Iterable[tuple[core.Displacement, core.Force]], force_scale: float,
) -> Iterator[np.ndarray]:
    """A deforming patch pushes on its boundary around its closed cycle."""
    boundary = topology.exterior[0]
    # Each force on the boundary drawn along its metric direction, its dual, from its particle.
    places, forces = (stack([value.batch() for value in values]) for values in zip(*states))   # [steps, vertices] Displacement-, Force+
    arrows = force_scale * forces[:, boundary].dual()                        # [steps, boundary] Displacement-
    bounds = _bounds([reference, places, places[:, boundary] + arrows], 0.055)

    figure = plt.figure(figsize=(5.2, 3.6), dpi=50, facecolor=BACKGROUND)
    ax = _scene(figure, bounds, (0.02, 0.02, 0.96, 0.96))
    camera = panel(ax, core.mv.x, core.mv.y)
    rest = simplices.strokes(reference, topology, REFERENCE, 1.0, camera)
    forces_topology = _arrows(len(boundary))
    image = simplices.overlay(ax, [rest])

    for position, arrow in zip(places, arrows):
        tails = position[boundary]                                           # [boundary] Displacement-
        # The forces' arrows, strokes ending in chevrons.
        image.set_data(simplices.composited([
            rest,
            simplices.strokes(position, topology, BOND, 1.5, camera),
            simplices.markers(position, ANCHOR, 2.0, camera),
            simplices.strokes(concatenate([tails, tails + arrow]), forces_topology, FORCE, 1.5, camera,
                              chevrons=np.ones(len(boundary)), chevron=6.0)]))
        yield capture(figure)
    plt.close(figure)


def wave_frames(
    reference: core.Displacement, topology: core.Topology,
    states: Iterable[tuple[core.Displacement, core.Scalar]],
    displacement_scale: float, extension_scale: float, tracked_index: int,
) -> Iterator[np.ndarray]:
    """Moving bonds coloured by signed extension, with one particle's short trajectory marked."""
    figure = plt.figure(figsize=(6.5, 3.7), dpi=50, facecolor=BACKGROUND)
    ax = _scene(figure, _bounds([reference], 0.035), (0.015, 0.015, 0.97, 0.97))
    camera = panel(ax, core.mv.x, core.mv.y)
    rest = simplices.strokes(reference, topology, REFERENCE, 1.0, camera)
    strain = Normalize(-extension_scale, extension_scale)
    image = simplices.overlay(ax, [rest])

    n_trail_frames = 33
    history = []
    for positions, extensions in states:
        # The motion from the rest positions, exaggerated.
        position = (reference + displacement_scale * (positions - reference)).batch()   # [vertices] Displacement-
        tracked = position[tracked_index][None]                              # [1] Displacement-
        history.append(tracked)
        trajectory = concatenate(history[-n_trail_frames:])                  # [trail] Displacement-
        image.set_data(simplices.composited([
            rest,
            simplices.strokes(position, topology, STRAIN(strain(extensions.batch().to_array()))[:, :3], 1.2, camera),
            simplices.markers(position, ANCHOR, 1.2, camera),
            simplices.strokes(trajectory, simplices.Topology.chains([len(trajectory)]), TRACK, 2.5, camera),
            simplices.markers(tracked, ANCHOR, 4.0, camera),
            simplices.markers(tracked, TRACK, 3.3, camera)]))
        yield capture(figure)
    plt.close(figure)
