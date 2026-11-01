"""The seven multiplication triples of the octonions, drawn as the Fano plane."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D
from examples.math.octonions import core
from examples.util import charts
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import flat_panel

line_colour = "#287fa0"
point_colour = "0.15"
# The figure's width and height in pixels, for seven points and seven lines; the side of a point, wide enough for its
# name, and the length of the strokes of the chevron at an arrow's tip, in pixels; the points along the circle.
figure_size = (300, 275)
point_side = 20
chevron_length = 7.0
n_around = 120
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
# How far along each segment, and along each arc of the circle, its arrow sits: off the middle, where the medians
# cross the circle; and how far on its tip lies, and back its stroke starts, on the page.
segment_fraction, arc_fraction = 0.65, 0.3
tip_step, tail_step = 0.02, 0.06


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


def arrow(place: Antivector, heading: Antivector) -> Antivector:
    """The ends of a short arrow at a place, pointing along a heading `[ends]`: the end of its stroke and its tip."""
    unit = heading.normalized()                                                # [] Antivector-
    return stack([place - tail_step * unit, place + tip_step * unit])


@charts.style()
def draw_fano(cycles: core.Trivector) -> plt.Figure:
    """The Fano plane: each basis vector a point, each multiplication triple a line, the sides and medians
    of the triangle and the circle through the midpoints, its arrows running in the order of the product."""
    figure, ax = charts.subplots(*figure_size, layout="constrained")
    ax.set(xlim=(-1.05, 1.05), ylim=(-0.75, 1.15), aspect="equal")
    ax.set_axis_off()
    radius = corner / 2
    angles = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    circle = (page.xy * (-angles / 2)).exp() >> (radius * page.y.dual())       # [around] Antivector-
    sides, arrows = [], []
    # Each triple a cycle, from each point to the next in the order of the product, the last back to the first.
    cycle = simplices.Topology.loops([3])
    for triple in triples(cycles):
        places = stack([places_by_name[name] for name in triple])              # [3] Antivector-
        starts, ends = (cycle.incidence[1, 0] * places.field()).batch()        # [3] Antivector-, Antivector-
        if np.all((places.norm() - radius).abs() <= 1e-8 + 1e-5 * radius):
            # On the circle: an arrow on each arc, part of the way along the shorter turn from each point to the
            # next, pointing the way the turn runs.
            turns = (1 + ends * starts.inverse()).normalized()                 # [3] Rotor+
            generators = turns.log()                                            # [3] Bivector+
            heads = (generators * arc_fraction).exp() >> starts                # [3] Antivector-
            arrows += [arrow(head, heading) for head, heading in zip(heads, generators.commutator(heads))]
            continue
        # On a straight line: the two points furthest apart are its ends.
        spans = (places[:, None] - places[None, :]).norm()                    # [3, 3] Scalar+
        first, last = np.unravel_index(spans.reshape(-1).argmax(), (3, 3))
        sides.append(places[[first, last]])
        for start, end in zip(starts, ends):
            # Arrows on the segments the cycle runs along without passing through the third point.
            if (end - start).norm() < spans[first, last] - 1e-9:
                arrows.append(arrow(start + segment_fraction * (end - start), end - start))
    points = stack(list(places_by_name.values()))                              # [points] Antivector-
    arrow_ends = stack(arrows)                                                 # [arrows, ends] Antivector-
    camera = flat_panel(ax, VGA2D)
    ink, dark = np.array(to_rgb(line_colour)), np.array(to_rgb(point_colour))
    simplices.overlay(ax, [
        simplices.strokes(circle, simplices.Topology.loops([n_around]), ink, charts.LINE_PX, camera),
        simplices.strokes(stack(sides).reshape(-1), simplices.Topology.chains([2] * len(sides)), ink, charts.LINE_PX,
                          camera),
        # Each arrow a stroke along its line, as wide as the line, and a chevron at its tip.
        simplices.strokes(arrow_ends.reshape(-1), simplices.Topology.chains([2] * len(arrows)), ink, charts.LINE_PX,
                          camera, chevrons=np.ones(len(arrows)), chevron=chevron_length),
        # Each point white within a dark ring: a dot over a larger dot.
        simplices.markers(points, dark, (point_side + charts.LINE_PX) / 2, camera),
        simplices.markers(points, np.ones(3), (point_side - charts.LINE_PX) / 2, camera)])
    for name, place in places_by_name.items():
        ax.text(*xy(place), name, ha="center", va="center", color=point_colour, zorder=4)
    return figure
