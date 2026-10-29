"""The seven multiplication triples of the octonions, drawn as the Fano plane."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D
from examples.math.octonions import core
from examples.util import charts

line_colour = "#287fa0"
point_colour = "0.15"
# The figure's width and height in pixels, for seven points and seven lines; the side of a point, wide enough for its
# name, and of an arrowhead, in pixels.
figure_size = (300, 275)
point_side = 20
arrow_side = 16
# The page the diagram is drawn on, its points antivectors about the middle.
page = NumpyContext(VGA2D).multivector
Antivector = VGA2D.gatype.antivector()                                         # Antivector-
# Where each basis vector is drawn: the corners of a triangle at b, d and g, the top turned by thirds of a turn, the
# midpoints of its sides at f, c and e, and its centre at a, so that every triple lies on a side, a median, or the
# circle through the midpoints.
corner = 1.0
thirds = (page.xy * (-np.array([0, 2, 4]) * np.pi / 3 / 2)).exp()                # [corners] Rotor+
b, d, g = thirds >> (corner * page.y.dual())                                    # [] Antivector-, Antivector-, Antivector-
places_by_name = {"a": 0 * b, "b": b, "d": d, "g": g, "f": (b + d) / 2, "e": (b + g) / 2, "c": (d + g) / 2}
arrow_style = dict(arrowstyle="-|>", mutation_scale=charts.points(arrow_side), color=line_colour, linewidth=0)
# How far along each segment, and along each arc of the circle, its arrowhead sits: off the middle, where the
# medians cross the circle.
segment_fraction, arc_fraction = 0.65, 0.3


# --- plumbing -------------------------------------------------------------------------
def xy(points: Antivector) -> np.ndarray:
    """x and y of points of the page `[..., 2]`: their pairings with the planes x and y."""
    return (stack((page.x, page.y)) & points[..., None]).to_array()


def triples(cycles: core.Trivector) -> list[str]:
    """The three basis vectors of each cycle, in the order its product runs: `a` times `b` is `c` for abc."""
    names = cycles.gatype.output_subspace.blade_names
    mv = cycles.context.multivector
    blades = stack([getattr(mv, name) for name in names])                      # [blades] Trivector+
    # A cycle's coefficient on each basis trivector, by its scalar product with the blade's reverse.
    coefficients = (cycles[:, None].scalar_product(blades.reverse()) / blades.scalar_norm_squared()).to_array()
    blade = np.abs(coefficients).argmax(axis=-1)
    return [names[k] if coefficients[row, k] > 0 else names[k][0] + names[k][2] + names[k][1]
            for row, k in enumerate(blade)]


def arrow(ax: plt.Axes, place: Antivector, heading: Antivector) -> None:
    """An arrowhead at a place, pointing along a heading."""
    step = heading.normalized() * 0.02                                          # [] Antivector-
    ax.add_patch(FancyArrowPatch(*xy(stack([place - step, place + step])), **arrow_style))


@charts.style()
def draw_fano(cycles: core.Trivector) -> plt.Figure:
    """The Fano plane: each basis vector a point, each multiplication triple a line, the sides and medians
    of the triangle and the circle through the midpoints, its arrows running in the order of the product."""
    figure, ax = charts.subplots(*figure_size, layout="constrained")
    radius = corner / 2
    ax.add_patch(Circle((0, 0), radius, fill=False, color=line_colour))
    for triple in triples(cycles):
        places = stack([places_by_name[name] for name in triple])              # [3] Antivector-
        following = places[[1, 2, 0]]                                          # [3] Antivector-
        if np.allclose(places.norm().to_array(), radius):
            # On the circle: an arrowhead on each arc, part of the way along the shorter turn from each point to the
            # next, pointing the way the turn runs.
            turns = (1 + following * places.inverse()).normalized()            # [3] Rotor+
            generators = turns.log()                                            # [3] Bivector+
            heads = (generators * arc_fraction).exp() >> places                # [3] Antivector-
            for head, heading in zip(heads, generators.commutator(heads)):
                arrow(ax, head, heading)
            continue
        # On a straight line: the two points furthest apart are its ends.
        spans = (places[:, None] - places[None, :]).norm().to_array()          # [3, 3]
        first, last = np.unravel_index(spans.argmax(), (3, 3))
        ax.plot(*xy(places[[first, last]]).T, color=line_colour)
        for start, end in zip(places, following):
            # Arrowheads on the segments the cycle runs along without passing through the third point.
            if (end - start).norm().to_array() < spans.max() - 1e-9:
                arrow(ax, start + segment_fraction * (end - start), end - start)
    for name, place in places_by_name.items():
        ax.scatter(*xy(place), s=charts.points(point_side) ** 2, color="white", edgecolors=point_colour, zorder=3)
        ax.text(*xy(place), name, ha="center", va="center", color=point_colour, zorder=4)
    ax.set(xlim=(-1.05, 1.05), ylim=(-0.75, 1.15), aspect="equal")
    ax.set_axis_off()
    return figure
