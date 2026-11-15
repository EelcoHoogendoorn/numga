"""GPU rendering of active bonds, shear cycles, and overdamped waves in the planar lattice algebra."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from examples.mechanics.odd_elasticity import core
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay
from numga import concatenate, stack
from numga.render import contours, drawing, simplices
from numga.topology import Topology

background_colour = "#fbfaf7"
bond_colour, force_colour, anchor_colour, reference_colour, swept_colour, track_colour = (
    to_rgb(colour) for colour in ("#267c8b", "#db663d", "#35434d", "#d9dcda", "#f6d9c9", "#f3b83f")
)
strain_cmap = LinearSegmentedColormap.from_list("bond_extension", ["#4775bd", "#32666d", "#ed8b3d"])


# --- plumbing -------------------------------------------------------------------------
def scene_bounds(batches: list[core.Displacement], margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Axis pairing over `batches` determines padded `[2]` lower and upper coordinate bounds."""
    axes = stack((core.mv.x, core.mv.y))                                     # [2] Force+
    readings = concatenate([axes[:, None] & batch.batch().reshape(-1) for batch in batches], axis=-1)  # [2, ...] Scalar+
    low = readings.min(axis=-1)                                              # [2] Scalar+
    high = readings.max(axis=-1)                                             # [2] Scalar+
    padding = margin * (high - low).max()                                    # Scalar+
    return (low - padding).to_array(), (high + padding).to_array()


def scene_axes(figure: plt.Figure, bounds: tuple[np.ndarray, np.ndarray], position: tuple[float, float, float, float]) -> plt.Axes:
    """Equal-aspect borderless axes span `bounds` at `position` on `figure`."""
    ax = figure.add_axes(position, facecolor=background_colour)
    low, high = bounds
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
    ax.set_axis_off()
    return ax


def arrow_topology(n_arrows: int) -> Topology:
    """Paired tail and tip vertex indices define `n_arrows` directed line segments."""
    return Topology(2 * n_arrows, np.stack([np.arange(n_arrows), n_arrows + np.arange(n_arrows)], axis=-1))


def bond_frames(
    anchor: core.Displacement, reference: core.Displacement, path: core.Displacement,
    force_scale: float, states: Iterable[tuple[core.Displacement, core.Force, core.Scalar]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames as a bond endpoint traces a closed path, shading swept area and drawing the bond force."""
    recorded = list(states)
    n_steps = len(recorded)
    # Each force arrow extends along its metric dual direction from the moving endpoint.
    positions = stack([position.batch() for position, _, _ in recorded])     # [n_steps] Displacement-
    forces = stack([force.batch() for _, force, _ in recorded])              # [n_steps] Force+
    arrows = force_scale * forces.dual()                                     # [n_steps] Displacement-
    bounds = scene_bounds([anchor, reference, path, positions + arrows], 0.08)

    figure = plt.figure(figsize=(6.2, 2.9), dpi=50, facecolor=background_colour)
    ax = scene_axes(figure, bounds, (0.02, 0.02, 0.96, 0.96))
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    # Triangle fan from the rest position over the path traversed so far.
    swept_corners = concatenate([reference[None], positions])                # [n_steps + 1] Displacement-
    fan = np.stack([np.zeros(n_steps - 1, dtype=int), np.arange(1, n_steps), np.arange(2, n_steps + 1)], axis=-1)
    guides = [
        simplices.strokes(concatenate([anchor[None], reference[None]]), Topology.chains([2]), reference_colour,
                          1.5, camera, dashes=(5.0, 3.0), cap=simplices.Cap.BUTT),
        simplices.strokes(path, Topology.chains([len(path)]), reference_colour, 1.0, camera),
        simplices.markers(anchor[None], anchor_colour, 3.0, camera),
        # Ring marker at the rest position formed by a background disc over a larger dark disc.
        simplices.markers(reference[None], anchor_colour, 2.5, camera),
        simplices.markers(reference[None], to_rgb(background_colour), 1.5, camera),
    ]
    image = overlay(ax, guides)

    for step, (position, arrow) in enumerate(zip(positions, arrows)):
        bond = concatenate([anchor[None], position[None]])                   # [2] Displacement-
        force_segment = concatenate([position[None], (position + arrow)[None]])  # [2] Displacement-
        image.set_data(drawing.composited([
            contours.picture(swept_corners, Topology(n_steps + 1, fan[:step]), np.zeros(n_steps + 1), swept_colour,
                             camera),
            *guides,
            simplices.strokes(positions[:step + 1], Topology.chains([step + 1]), bond_colour, 1.0, camera,
                              opacity=0.55),
            simplices.strokes(bond, Topology.chains([2]), bond_colour, 3.0, camera),
            simplices.markers(position[None], bond_colour, 3.0, camera),
            simplices.strokes(force_segment, arrow_topology(1), force_colour, 3.0, camera,
                              chevrons=np.ones(1), chevron=12.0),
        ]))
        yield capture(figure)
    plt.close(figure)


def patch_frames(
    reference: core.Displacement, topology: Topology, force_scale: float,
    states: Iterable[tuple[core.Displacement, core.Force, core.Scalar]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of a deforming lattice patch and its boundary forces around a shear cycle."""
    n_vertices = len(reference.batch())
    boundary = topology.exterior[0]
    n_boundary = len(boundary)
    recorded = list(states)
    n_steps = len(recorded)
    # Boundary forces extend along their metric dual directions from exterior vertices.
    places = stack([place.batch() for place, _, _ in recorded])              # [n_steps, n_vertices] Displacement-
    forces = stack([force.batch() for _, force, _ in recorded])              # [n_steps, n_vertices] Force+
    arrows = force_scale * forces[:, boundary].dual()                        # [n_steps, n_boundary] Displacement-
    bounds = scene_bounds([reference, places, places[:, boundary] + arrows], 0.055)

    figure = plt.figure(figsize=(5.2, 3.6), dpi=50, facecolor=background_colour)
    ax = scene_axes(figure, bounds, (0.02, 0.02, 0.96, 0.96))
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    rest = simplices.strokes(reference, topology, reference_colour, 1.0, camera)
    forces_topology = arrow_topology(n_boundary)
    image = overlay(ax, [rest])

    for position, arrow in zip(places, arrows):
        tails = position[boundary]                                           # [n_boundary] Displacement-
        image.set_data(drawing.composited([
            rest,
            simplices.strokes(position, topology, bond_colour, 1.5, camera),
            simplices.markers(position, anchor_colour, 2.0, camera),
            simplices.strokes(concatenate([tails, tails + arrow]), forces_topology, force_colour, 1.5, camera,
                              chevrons=np.ones(n_boundary), chevron=6.0),
        ]))
        yield capture(figure)
    plt.close(figure)


def wave_frames(
    reference: core.Displacement, topology: Topology, displacement_scale: float,
    extension_scale: float, tracked_index: int, states: Iterable[tuple[core.Displacement, core.Scalar]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of lattice bonds coloured by extension, trailing one particle's trajectory."""
    n_vertices = len(reference.batch())
    figure = plt.figure(figsize=(6.5, 3.7), dpi=50, facecolor=background_colour)
    ax = scene_axes(figure, scene_bounds([reference], 0.035), (0.015, 0.015, 0.97, 0.97))
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    rest = simplices.strokes(reference, topology, reference_colour, 1.0, camera)
    strain = Normalize(-extension_scale, extension_scale)
    image = overlay(ax, [rest])

    n_trail_frames = 33
    history = []
    for positions, extensions in states:
        # Exaggerate displacements about the rest lattice so small wave amplitudes remain visible.
        position = (reference + displacement_scale * (positions - reference)).batch()  # [n_vertices] Displacement-
        tracked_site = position[tracked_index][None]                         # [1] Displacement-
        history.append(tracked_site)
        trajectory = concatenate(history[-n_trail_frames:])                  # [...] Displacement-
        n_trail = len(trajectory)
        image.set_data(drawing.composited([
            rest,
            simplices.strokes(position, topology, strain_cmap(strain(extensions.batch().to_array()))[:, :3], 1.2, camera),
            simplices.markers(position, anchor_colour, 1.2, camera),
            simplices.strokes(trajectory, Topology.chains([n_trail]), track_colour, 2.5, camera),
            simplices.markers(tracked_site, anchor_colour, 4.0, camera),
            simplices.markers(tracked_site, track_colour, 3.3, camera),
        ]))
        yield capture(figure)
    plt.close(figure)
