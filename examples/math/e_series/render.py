"""The nested exceptional root systems in one diagram and one common projection."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from examples.math.e8.core import Scalar, Vector
from examples.math.e8.render import BACKGROUND, COLOURS, coordinates

FAMILY_COLOURS = COLOURS[[5, 0, 3]]


def format_scalars(values: Scalar) -> str:
    """A compact numeric readout for dimensions and winding rates."""
    return np.array2string(values.to_array(), precision=6, suppress_small=True)


def draw_diagram(simple_roots: Vector, ranks: np.ndarray, names: tuple[str, ...]) -> plt.Figure:
    """The E8 diagram, coloured by the stage at which each simple root enters."""
    inner_products = (simple_roots[:, None] | simple_roots[None, :]).to_array()
    first, second = np.nonzero(np.triu(inner_products < 0, k=1))
    positions = np.array([[0, 0], [2, 1], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]])
    stages = np.searchsorted(ranks, np.arange(len(simple_roots)), side="right")
    segments = np.stack([positions[first], positions[second]], axis=1)
    edge_colours = FAMILY_COLOURS[np.maximum(stages[first], stages[second])]

    figure, ax = plt.subplots(figsize=(8, 2.4), facecolor=BACKGROUND)
    ax.add_collection(LineCollection(segments, colors=edge_colours, alpha=0.7, linewidths=1.5))
    ax.scatter(*positions.T, c=FAMILY_COLOURS[stages], s=120, edgecolors=BACKGROUND, linewidths=1, zorder=3)
    for index, position in enumerate(positions):
        ax.text(*position, str(index + 1), color=BACKGROUND, fontsize=8, ha="center", va="center", zorder=4)
    ax.plot([0, 0, 4, 4], [-0.3, -0.4, -0.4, -0.3], color=FAMILY_COLOURS[0], linewidth=0.8)
    for name, colour, horizontal in zip(names, FAMILY_COLOURS, (2, 5, 6)):
        ax.text(horizontal, -0.6, name, color=colour, fontsize=12, ha="center", va="center")
    ax.set(xlim=(-0.35, 6.35), ylim=(-0.8, 1.2), aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(0.025, 0.025, 0.975, 0.975)
    plt.close(figure)
    return figure


def draw_nested_roots(root_systems: tuple[Vector, ...], horizontal: Vector, vertical: Vector,
                      names: tuple[str, ...]) -> plt.Figure:
    """E6 inside E7 inside E8, all projected onto E8's Coxeter plane."""
    roots = root_systems[-1].reshape((-1,))                                   # [roots] Vector
    projected = coordinates(roots, horizontal, vertical)
    # Membership is measured in the full root space, before any projection.
    memberships = np.stack([
        ((roots[:, None] - subset.reshape((-1,))[None, :])
         .scalar_norm_squared().to_array().min(axis=-1) < 1e-8)
        for subset in root_systems
    ])
    stages = memberships.argmax(axis=0)
    counts = np.bincount(stages, minlength=len(names))
    colours = FAMILY_COLOURS[stages]
    distances = (roots[:, None] - roots[None, :]).scalar_norm_squared().to_array()
    nearest = distances[distances > 1e-8].min()
    first, second = np.nonzero(np.triu(np.isclose(distances, nearest), k=1))
    segments = np.stack([projected[first], projected[second]], axis=1)
    edge_stages = np.maximum(stages[first], stages[second])
    edge_colours = np.column_stack([FAMILY_COLOURS[edge_stages], np.full(len(first), 0.13)])

    figure, ax = plt.subplots(figsize=(7, 7.3), facecolor=BACKGROUND)
    ax.add_collection(LineCollection(segments, colors=edge_colours, linewidths=0.35))
    ax.scatter(*projected.T, c=colours, s=18, edgecolors=BACKGROUND, linewidths=0.4, zorder=3)
    radius = np.linalg.norm(projected, axis=-1).max() * 1.045
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
    ax.set_axis_off()
    labels = (f"{names[0]}: {counts[0]}", *(f"{name}: +{count}" for name, count in zip(names[1:], counts[1:])))
    for name, colour, position in zip(labels, FAMILY_COLOURS, (0.23, 0.5, 0.77)):
        figure.text(position, 0.028, name, color=colour, fontsize=11, ha="center", va="center")
    figure.subplots_adjust(0.01, 0.06, 0.99, 0.995)
    plt.close(figure)
    return figure
