"""The nested exceptional root systems in one diagram and one common projection."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from numga import NumpyContext
from numga.algebras import PGA2D
from numga.render import simplices
from numga.topology import Topology

from examples.math.e_series.core import Plane
from examples.util import charts
from examples.util.panels import offset_panel, overlay, panel

background = "#101219"
family_colours = np.array([
    [0.24, 0.80, 0.82],
    [0.98, 0.72, 0.27],
    [0.60, 0.48, 0.92],
])
# The diagram is laid out in a plane of its own.
layout = NumpyContext(PGA2D).multivector
# The diagram's nodes' radius in pixels, wide enough for a numeral at the size of a tick label; how much the diagram's
# and the lattice's edges cover.
node_radius = 8.0
diagram_opacity, haze_opacity = 0.7, 0.065


@charts.style()
def diagram_figure(simple_roots: Plane, ranks: np.ndarray, names: tuple[str, ...]) -> plt.Figure:
    """The E8 Dynkin diagram colours each simple root by the stage at which it enters."""
    n_roots = len(simple_roots)
    inner_products = simple_roots[:, None] | simple_roots[None, :]            # [n_roots, n_roots] Scalar+
    first, second = np.nonzero(np.triu(inner_products < 0, k=1))
    positions = np.array([[0, 0], [2, 1], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]])
    stages = np.searchsorted(ranks, np.arange(n_roots), side="right")
    edge_colours = family_colours[np.maximum(stages[first], stages[second])]

    # Over half the column: eight nodes in a row, a numeral in each, and the three names under them.
    figure, ax = charts.subplots(460, 145, facecolor=background)
    ax.set(xlim=(-0.35, 6.35), ylim=(-0.8, 1.2), aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(0.025, 0.025, 0.975, 0.975)
    camera = panel(ax, layout.x, layout.y, layout.w)
    nodes = layout.point(positions)                                           # [8] Point-
    # A bracket under the first stage's six nodes.
    bracket = layout.point([[0, -0.3], [0, -0.4], [4, -0.4], [4, -0.3]])      # [4] Point-
    overlay(ax, [
        simplices.strokes(nodes, Topology(len(positions), np.stack([first, second], axis=-1)), edge_colours,
                          charts.LINE_PX, camera, opacity=diagram_opacity),
        simplices.markers(nodes, np.array(to_rgb(background)), node_radius + charts.FRAME_PX, camera),
        simplices.markers(nodes, family_colours[stages], node_radius, camera),
        simplices.strokes(bracket, Topology.chains([4]), family_colours[0], charts.FRAME_PX, camera)])
    for index, position in enumerate(positions):
        ax.text(*position, str(index + 1), color=background, fontsize=charts.points(charts.TICK_PX), ha="center",
                va="center", zorder=4)
    for name, colour, horizontal in zip(names, family_colours, (2, 5, 6)):
        ax.text(horizontal, -0.6, name, color=colour, ha="center", va="center")
    plt.close(figure)
    return figure


@charts.style()
def nested_roots_figure(root_systems: tuple[Plane, ...], horizontal: Plane, vertical: Plane,
                        names: tuple[str, ...]) -> plt.Figure:
    """E8's Coxeter plane displays the nested E6, E7 and E8 root systems together."""
    roots = root_systems[-1].reshape((-1,))                                   # [n_roots] Plane+
    n_roots = len(roots)
    # Membership is measured in the full root space, before any projection.
    memberships = np.stack([
        ((roots[:, None] - subset.reshape((-1,))[None, :])
         .scalar_norm_squared() < 1e-8).any(axis=-1)
        for subset in root_systems
    ])
    stages = memberships.argmax(axis=0)
    counts = np.bincount(stages, minlength=len(names))
    colours = family_colours[stages]
    distances = (roots[:, None] - roots[None, :]).scalar_norm_squared()       # [n_roots, n_roots] Scalar+
    distinct_distances = distances[distances > 1e-8]
    nearest = distinct_distances.min()
    first, second = np.nonzero(np.triu((distances - nearest).abs() <= (1e-8 + 1e-5 * nearest.abs()), k=1))
    edge_stages = np.maximum(stages[first], stages[second])

    # Most of the column, for the dense lattice of roots under the haze of their edges.
    figure, ax = charts.subplots(680, 710, facecolor=background)
    # A root's part in the drawing plane is as long as its contraction onto the plane's unit bivector.
    radii = (roots | (horizontal ^ vertical)).norm()                          # [n_roots] Scalar+
    radius = float((radii.max() * 1.045).to_array())
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(0.01, 0.06, 0.99, 0.995)
    # The camera reads roots, planes, as they are, along the drawing plane's two directions: `horizontal | root` pairs
    # as `root & horizontal.dual()`. The edges form a faint haze of one frame's width.
    camera = offset_panel(ax, horizontal.dual(), vertical.dual())
    neighbours = Topology(n_roots, np.stack([first, second], axis=-1))
    overlay(ax, [
        simplices.strokes(roots, neighbours, family_colours[edge_stages], charts.FRAME_PX, camera,
                          opacity=haze_opacity),
        simplices.markers(roots, colours, charts.MARKER_PX / 2, camera)])
    labels = (f"{names[0]}: {counts[0]}", *(f"{name}: +{count}" for name, count in zip(names[1:], counts[1:])))
    for name, colour, position in zip(labels, family_colours, (0.23, 0.5, 0.77)):
        figure.text(position, 0.028, name, color=colour, ha="center", va="center")
    plt.close(figure)
    return figure
