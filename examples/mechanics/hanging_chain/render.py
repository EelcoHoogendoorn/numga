"""Drawing for the hanging chain: every Newton iterate of the chain between its pins, over the catenary;
and the largest unbalanced force on a bead at each iterate."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, concatenate, stack

from examples.util import charts
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import flat_panel

EARLY = np.array([0.80, 0.80, 0.82])
LATE = np.array([0.15, 0.15, 0.18])
CHAIN = np.array([0.20, 0.40, 0.70])
CATENARY = np.array([0.85, 0.30, 0.20])
PIN = np.array([0.15, 0.15, 0.18])
# The catenary's dashes and gaps, in pixels; the radius of the last iterate's beads and of the pins' dots, a chart's
# marker across.
CATENARY_DASHES = (11.0, 5.0)
BEAD_RADIUS = charts.MARKER_PX / 2
# The margin around the chains, as a share of their extent either way.
MARGIN = 0.05


def readings(points: Extensor) -> Extensor:
    """Points of the plane read across and up `[2, points]`: their pairings with the lines x = 0 and y = 0."""
    lines = points.context.multivector
    return stack((lines.x, lines.y))[:, None] & points.reshape(-1)


@charts.style()
def iterates(chains, residual, catenary, pins) -> plt.Figure:
    """Two panels. Left: each iterate of the chain as the polyline from pin to pin through its beads,
    earlier ones lighter, the last with its beads marked, over the catenary dashed. Right: the largest
    unbalanced force on a bead at each iterate, in bead weights, on a log scale."""
    beads = chains.batch()                                                    # [iterates, beads] Point-
    n_iterates, n_beads = beads.shape
    ends = pins.batch()                                                       # [pins] Point-
    # Each iterate from pin to pin through its beads.
    lines = concatenate([ends[0].broadcast_to((n_iterates, 1)), beads, ends[1].broadcast_to((n_iterates, 1))],
                        axis=1)                                               # [iterates, beads + 2] Point-
    largest = np.asarray(residual.batch().to_array()).max(axis=-1)

    # Four fifths of the column: the chain's iterates, wide and shallow, beside a short convergence curve.
    figure, (shape, convergence) = charts.subplots(640, 280, ncols=2, gridspec_kw={"width_ratios": [1.3, 1]})
    convergence.semilogy(np.arange(len(largest)), largest, color=CHAIN, marker="o")
    convergence.set_xlabel("Newton step")
    convergence.set_ylabel("unbalanced force, in weights")
    convergence.grid(True, which="major", alpha=0.3)
    # The chains' panel framed about every iterate and the catenary.
    drawn = concatenate([readings(lines), readings(catenary)], axis=-1)       # [2, points] Scalar+
    low, high = drawn.min(axis=-1).to_array(), drawn.max(axis=-1).to_array()
    padding = MARGIN * (high - low)
    shape.set(xlim=(low[0] - padding[0], high[0] + padding[0]), ylim=(low[1] - padding[1], high[1] + padding[1]),
              aspect="equal")
    shape.axis("off")
    figure.tight_layout()

    camera = flat_panel(shape, chains.context.algebra)
    n_points = n_beads + 2
    # Earlier iterates lighter, each in its shade along its edges.
    shades = EARLY + np.linspace(0, 1, n_iterates)[:-1, None] * (LATE - EARLY)   # [iterates - 1, 3]
    simplices.overlay(shape, [
        simplices.strokes(lines[:-1].reshape(-1), simplices.Topology.chains([n_points] * (n_iterates - 1)),
                          np.repeat(shades, n_points - 1, axis=0), charts.LINE_PX, camera),
        # The catenary dashed and twice as wide as the chains, so that it shows under the last.
        simplices.strokes(catenary, simplices.Topology.chains([len(catenary)]), CATENARY, 2 * charts.LINE_PX, camera,
                          dashes=CATENARY_DASHES, cap=simplices.Cap.BUTT),
        simplices.strokes(lines[-1], simplices.Topology.chains([n_points]), CHAIN, charts.LINE_PX, camera),
        simplices.markers(lines[-1], CHAIN, BEAD_RADIUS, camera),
        simplices.markers(ends, PIN, BEAD_RADIUS, camera)])
    return figure
