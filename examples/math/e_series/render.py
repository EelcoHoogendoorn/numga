"""The nested exceptional root systems in one diagram and one common projection."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from examples.math.e8.core import Vector
from examples.math.e8.render import background, orbit_colours, coordinates
from examples.util import charts

family_colours = orbit_colours[[5, 0, 3]]


@charts.style()
def draw_diagram(simple_roots: Vector, ranks: np.ndarray, names: tuple[str, ...]) -> plt.Figure:
    """The E8 diagram, coloured by the stage at which each simple root enters."""
    inner_products = simple_roots[:, None] | simple_roots[None, :]
    first, second = np.nonzero(np.triu(inner_products < 0, k=1))
    positions = np.array([[0, 0], [2, 1], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]])
    stages = np.searchsorted(ranks, np.arange(len(simple_roots)), side="right")
    segments = np.stack([positions[first], positions[second]], axis=1)
    edge_colours = family_colours[np.maximum(stages[first], stages[second])]

    # Over half the column: eight nodes in a row, a numeral in each, and the three names under them.
    figure, ax = charts.subplots(460, 145, facecolor=background)
    ax.add_collection(LineCollection(segments, colors=edge_colours, alpha=0.7))
    # Each node wide enough for its numeral, at the size of a tick label, and rimmed a frame's width.
    ax.scatter(*positions.T, c=family_colours[stages], s=charts.points(16) ** 2, edgecolors=background,
               linewidths=charts.FRAME_WIDTH, zorder=3)
    for index, position in enumerate(positions):
        ax.text(*position, str(index + 1), color=background, fontsize=charts.points(charts.TICK_PX), ha="center",
                va="center", zorder=4)
    ax.plot([0, 0, 4, 4], [-0.3, -0.4, -0.4, -0.3], color=family_colours[0], linewidth=charts.FRAME_WIDTH)
    for name, colour, horizontal in zip(names, family_colours, (2, 5, 6)):
        ax.text(horizontal, -0.6, name, color=colour, ha="center", va="center")
    ax.set(xlim=(-0.35, 6.35), ylim=(-0.8, 1.2), aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(0.025, 0.025, 0.975, 0.975)
    plt.close(figure)
    return figure


@charts.style()
def draw_nested_roots(root_systems: tuple[Vector, ...], horizontal: Vector, vertical: Vector,
                      names: tuple[str, ...]) -> plt.Figure:
    """E6 inside E7 inside E8, all projected onto E8's Coxeter plane."""
    roots = root_systems[-1].reshape((-1,))                                   # [roots] Vector+
    projected = coordinates(roots, horizontal, vertical)
    # Membership is measured in the full root space, before any projection.
    memberships = np.stack([
        ((roots[:, None] - subset.reshape((-1,))[None, :])
         .scalar_norm_squared() < 1e-8).any(axis=-1)
        for subset in root_systems
    ])
    stages = memberships.argmax(axis=0)
    counts = np.bincount(stages, minlength=len(names))
    colours = family_colours[stages]
    distances = (roots[:, None] - roots[None, :]).scalar_norm_squared()
    distinct_distances = distances[distances > 1e-8]
    nearest = distinct_distances.min()
    first, second = np.nonzero(np.triu((distances - nearest).abs() <= (1e-8 + 1e-5 * nearest.abs()), k=1))
    segments = np.stack([projected[first], projected[second]], axis=1)
    edge_stages = np.maximum(stages[first], stages[second])
    edge_colours = np.column_stack([family_colours[edge_stages], np.full(len(first), 0.065)])

    # Most of the column, for the dense lattice of roots under the haze of their edges.
    figure, ax = charts.subplots(680, 710, facecolor=background)
    # The edges a faint haze, a frame's width.
    ax.add_collection(LineCollection(segments, colors=edge_colours, linewidths=charts.FRAME_WIDTH))
    ax.scatter(*projected.T, c=colours, edgecolors="none", zorder=3)
    projection = horizontal * (Vector | horizontal) + vertical * (Vector | vertical)
    radii = projection(roots).norm()
    radius = float((radii.max() * 1.045).to_array())
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
    ax.set_axis_off()
    labels = (f"{names[0]}: {counts[0]}", *(f"{name}: +{count}" for name, count in zip(names[1:], counts[1:])))
    for name, colour, position in zip(labels, family_colours, (0.23, 0.5, 0.77)):
        figure.text(position, 0.028, name, color=colour, ha="center", va="center")
    figure.subplots_adjust(0.01, 0.06, 0.99, 0.995)
    plt.close(figure)
    return figure
