"""The seven multiplication triples of the octonions, drawn as the Fano plane."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from examples.math.octonions import core
from examples.util import charts

line_colour = "#287fa0"
point_colour = "0.15"
# The figure's width and height in pixels, for seven points and seven lines; the side of a point, wide enough for its
# name, and of an arrowhead, in pixels.
figure_size = (300, 275)
point_side = 20
arrow_side = 16
# Where each basis vector is drawn: the corners of a triangle at b, d and g, the midpoints of its sides at f,
# c and e, and its centre at a, so that every triple lies on a side, a median, or the circle through the
# midpoints.
corner = 1.0
places_by_name = {
    "a": (0.0, 0.0),
    "b": (0.0, corner), "d": (-corner * np.sqrt(3) / 2, -corner / 2), "g": (corner * np.sqrt(3) / 2, -corner / 2),
    "f": (-corner * np.sqrt(3) / 4, corner / 4), "e": (corner * np.sqrt(3) / 4, corner / 4), "c": (0.0, -corner / 2),
}
arrow_style = dict(arrowstyle="-|>", mutation_scale=charts.points(arrow_side), color=line_colour, linewidth=0)
# How far along each segment, and along each arc of the circle, its arrowhead sits: off the middle, where the
# medians cross the circle.
segment_fraction, arc_fraction = 0.65, 0.3


# --- plumbing -------------------------------------------------------------------------
def triples(cycles: core.Trivector) -> list[str]:
    """The three basis vectors of each cycle, in the order its product runs: `a` times `b` is `c` for abc."""
    names = cycles.gatype.output_subspace.blade_names
    kernel = cycles.kernel                                                      # [cycles, blades]
    blade = np.abs(kernel).argmax(axis=-1)
    return [names[k] if kernel[row, k] > 0 else names[k][0] + names[k][2] + names[k][1] for row, k in enumerate(blade)]


def arrow(ax: plt.Axes, start: np.ndarray, end: np.ndarray) -> None:
    """An arrowhead part of the way from start to end, pointing towards end."""
    place = start + segment_fraction * (end - start)
    step = (end - start) / np.linalg.norm(end - start) * 0.02
    ax.add_patch(FancyArrowPatch(place - step, place + step, **arrow_style))


@charts.style()
def draw_fano(cycles: core.Trivector) -> plt.Figure:
    """The Fano plane: each basis vector a point, each multiplication triple a line, the sides and medians
    of the triangle and the circle through the midpoints, its arrows running in the order of the product."""
    figure, ax = charts.subplots(*figure_size, layout="constrained")
    radius = corner / 2
    ax.add_patch(Circle((0, 0), radius, fill=False, color=line_colour))
    for triple in triples(cycles):
        places = np.array([places_by_name[name] for name in triple])           # [3, 2]
        if np.allclose(np.linalg.norm(places, axis=-1), radius):
            # On the circle: an arrowhead on each arc, running from each point to the next.
            angles = np.arctan2(places[:, 1], places[:, 0])
            following = angles[[1, 2, 0]]
            for start, turn in zip(angles, (following - angles) % (2 * np.pi)):
                # The shorter way round from each point to the next, and part of the way along it.
                way = turn if turn < np.pi else turn - 2 * np.pi
                angle = start + arc_fraction * way
                tangent = np.array([-np.sin(angle), np.cos(angle)]) * np.sign(way)
                point = radius * np.array([np.cos(angle), np.sin(angle)])
                ax.add_patch(FancyArrowPatch(point - tangent * 0.02, point + tangent * 0.02, **arrow_style))
            continue
        # On a straight line: the two points furthest apart are its ends.
        span = places[:, None] - places[None, :]
        first, last = np.unravel_index(np.linalg.norm(span, axis=-1).argmax(), (3, 3))
        ax.plot(*places[[first, last]].T, color=line_colour)
        for start, end in zip(places, places[[1, 2, 0]]):
            # Arrowheads on the segments the cycle runs along without passing through the third point.
            if np.linalg.norm(end - start) < np.linalg.norm(places[first] - places[last]) - 1e-9:
                arrow(ax, start, end)
    for name, place in places_by_name.items():
        ax.scatter(*place, s=charts.points(point_side) ** 2, color="white", edgecolors=point_colour, zorder=3)
        ax.text(*place, name, ha="center", va="center", color=point_colour, zorder=4)
    ax.set(xlim=(-1.05, 1.05), ylim=(-0.75, 1.15), aspect="equal")
    ax.set_axis_off()
    return figure
