"""Coxeter projections of the E8 roots and their simultaneous planar rotations."""

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Circle

from numga.extensor import stack

from examples.util.animation import capture
from examples.math.e8.core import Vector

background = "#101219"
orbit_colours = np.array([
    [0.98, 0.72, 0.27],
    [0.96, 0.39, 0.32],
    [0.83, 0.40, 0.72],
    [0.60, 0.48, 0.92],
    [0.32, 0.63, 0.95],
    [0.24, 0.80, 0.82],
    [0.35, 0.82, 0.59],
    [0.76, 0.84, 0.40],
])


def coordinates(vectors: Vector, horizontal: Vector, vertical: Vector) -> np.ndarray:
    """The coordinates of vectors in a drawing plane, their inner products with its two orthonormal vectors."""
    return (stack((horizontal, vertical), axis=-1) | vectors[..., None]).to_array()


def draw_roots(roots: Vector, horizontal: Vector, vertical: Vector) -> plt.Figure:
    """The roots in the Coxeter plane, coloured by orbit, with nearest neighbours joined."""
    points = coordinates(roots, horizontal, vertical)
    flat = roots.reshape((-1,))
    distances = (flat[:, None] - flat[None, :]).scalar_norm_squared().to_array()
    nearest = distances[distances > 1e-8].min()
    first, second = np.nonzero(np.triu(np.isclose(distances, nearest), k=1))
    projected = points.reshape((-1, 2))
    # A root's part in the drawing plane is as long as its contraction onto the plane's unit bivector.
    reach = (roots | (horizontal ^ vertical)).norm().to_array()               # [steps, orbits]
    colours = np.tile(orbit_colours, (roots.shape[0], 1))
    edge_colours = (colours[first] + colours[second]) / 2
    segments = np.stack([projected[first], projected[second]], axis=1)

    figure, ax = plt.subplots(figsize=(7, 7), facecolor=background)
    ax.add_collection(LineCollection(segments, colors=edge_colours, alpha=0.17, linewidths=0.33))
    ax.scatter(*projected.T, c=colours, s=12, edgecolors=background, linewidths=0.35, zorder=3)
    radius = reach.max() * 1.05
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1)
    plt.close(figure)
    return figure


def animate_planes(turning_roots: Iterable[Vector], roots: Vector, horizontal: Vector,
                   vertical: Vector) -> Iterator[np.ndarray]:
    """Eight coloured roots moving in each invariant plane, with trails of equal duration."""
    projected = coordinates(roots[:, None, :], horizontal[None, :, None], vertical[None, :, None])
    positions = np.stack([
        coordinates(frame_roots[None, :], horizontal[:, None], vertical[:, None])
        for frame_roots in turning_roots
    ])
    n_frames, n_planes, n_orbits = positions.shape[:3]
    # The radius of each root's orbit in each plane: the length of its contraction onto the plane's unit bivector.
    planes = horizontal ^ vertical                                            # [planes] Bivector+
    orbit_radii = (roots[:, None, :] | planes[None, :, None]).norm().to_array()   # [steps, planes, orbits]
    radius = orbit_radii.max() * 1.08
    # The points each trail runs through, over the share of the loop it covers: one Coxeter step of thirty.
    n_trail, trail_share = 7, 1 / 30
    history = np.round(np.linspace(-trail_share * n_frames, 0, n_trail)).astype(int)
    trail_alpha = np.linspace(0.12, 0.85, n_trail - 1)

    figure, axes = plt.subplots(1, n_planes, figsize=(4.4, 1.14), facecolor=background)
    riders, trails = [], []
    for index, ax in enumerate(axes):
        arcs = []
        for orbit, colour in enumerate(orbit_colours):
            orbit_radius = orbit_radii[0, index, orbit]
            ax.add_patch(Circle((0, 0), orbit_radius, fill=False, color=colour, alpha=0.26, linewidth=0.29))
            for alpha in trail_alpha:
                arc = Arc((0, 0), 2 * orbit_radius, 2 * orbit_radius, color=colour, alpha=alpha, linewidth=0.79)
                ax.add_patch(arc)
                arcs.append(arc)
        trails.append(arcs)
        dots = projected[:, index].reshape((-1, 2))
        ax.scatter(*dots.T, c=np.tile(orbit_colours, (roots.shape[0], 1)), s=0.58, alpha=0.35, linewidths=0)
        riders.append(ax.scatter(*positions[0, index].T, c=orbit_colours, s=4.5, edgecolors="white", linewidths=0.2, zorder=4))
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.025)

    for frame in range(n_frames):
        for index in range(n_planes):
            trail = positions[(frame + history) % n_frames, index].transpose(1, 0, 2)
            # The arcs take the angles of the trail's points in the plane, read from their two coordinates.
            angles = np.degrees(np.unwrap(np.arctan2(trail[..., 1], trail[..., 0]), axis=1))
            for arc, start, stop in zip(trails[index], angles[:, :-1].flat, angles[:, 1:].flat):
                arc.theta1, arc.theta2 = start, stop
            riders[index].set_offsets(positions[frame, index])
        yield capture(figure)
    plt.close(figure)
