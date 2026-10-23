"""The seven multiplication triples of the octonions, drawn as the Fano plane."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from examples.math.octonions import core

LINE_COLOUR = "#287fa0"
POINT_COLOUR = "0.15"
# Where each basis vector is drawn: the corners of a triangle at b, d and g, the midpoints of its sides at f,
# c and e, and its centre at a, so that every triple lies on a side, a median, or the circle through the
# midpoints.
CORNER = 1.0
PLACES = {
    "a": (0.0, 0.0),
    "b": (0.0, CORNER), "d": (-CORNER * np.sqrt(3) / 2, -CORNER / 2), "g": (CORNER * np.sqrt(3) / 2, -CORNER / 2),
    "f": (-CORNER * np.sqrt(3) / 4, CORNER / 4), "e": (CORNER * np.sqrt(3) / 4, CORNER / 4), "c": (0.0, -CORNER / 2),
}
ARROW = dict(arrowstyle="-|>", mutation_scale=14, color=LINE_COLOUR, linewidth=0)
# How far along each segment, and along each arc of the circle, its arrowhead sits: off the middle, where the
# medians cross the circle.
ALONG, AROUND = 0.65, 0.3


# --- plumbing -------------------------------------------------------------------------
def triples(cycles: core.Trivector) -> list[str]:
    """The three basis vectors of each cycle, in the order its product runs: `a` times `b` is `c` for abc."""
    names = cycles.gatype.output_subspace.blade_names
    kernel = cycles.kernel                                                      # [cycles, blades]
    blade = np.abs(kernel).argmax(axis=-1)
    return [names[k] if kernel[row, k] > 0 else names[k][0] + names[k][2] + names[k][1] for row, k in enumerate(blade)]


def arrow(ax: plt.Axes, start: np.ndarray, end: np.ndarray) -> None:
    """An arrowhead part of the way from start to end, pointing towards end."""
    place = start + ALONG * (end - start)
    step = (end - start) / np.linalg.norm(end - start) * 0.02
    ax.add_patch(FancyArrowPatch(place - step, place + step, **ARROW))


def draw_fano(cycles: core.Trivector) -> plt.Figure:
    """The Fano plane: each basis vector a point, each multiplication triple a line, the sides and medians
    of the triangle and the circle through the midpoints, its arrows running in the order of the product."""
    figure, ax = plt.subplots(figsize=(4.6, 4.6), layout="constrained")
    radius = CORNER / 2
    ax.add_patch(Circle((0, 0), radius, fill=False, color=LINE_COLOUR, linewidth=1.6))
    for triple in triples(cycles):
        places = np.array([PLACES[name] for name in triple])                   # [3, 2]
        if np.allclose(np.linalg.norm(places, axis=-1), radius):
            # On the circle: an arrowhead on each arc, running from each point to the next.
            angles = np.arctan2(places[:, 1], places[:, 0])
            following = angles[[1, 2, 0]]
            for start, turn in zip(angles, (following - angles) % (2 * np.pi)):
                # The shorter way round from each point to the next, and part of the way along it.
                way = turn if turn < np.pi else turn - 2 * np.pi
                angle = start + AROUND * way
                tangent = np.array([-np.sin(angle), np.cos(angle)]) * np.sign(way)
                point = radius * np.array([np.cos(angle), np.sin(angle)])
                ax.add_patch(FancyArrowPatch(point - tangent * 0.02, point + tangent * 0.02, **ARROW))
            continue
        # On a straight line: the two points furthest apart are its ends.
        span = places[:, None] - places[None, :]
        first, last = np.unravel_index(np.linalg.norm(span, axis=-1).argmax(), (3, 3))
        ax.plot(*places[[first, last]].T, color=LINE_COLOUR, linewidth=1.6)
        for start, end in zip(places, places[[1, 2, 0]]):
            # Arrowheads on the segments the cycle runs along without passing through the third point.
            if np.linalg.norm(end - start) < np.linalg.norm(places[first] - places[last]) - 1e-9:
                arrow(ax, start, end)
    for name, place in PLACES.items():
        ax.scatter(*place, s=260, color="white", edgecolors=POINT_COLOUR, linewidths=1.4, zorder=3)
        ax.text(*place, name, ha="center", va="center", fontsize=12, color=POINT_COLOUR, zorder=4)
    ax.set(xlim=(-1.05, 1.05), ylim=(-0.75, 1.15), aspect="equal")
    ax.set_axis_off()
    return figure
