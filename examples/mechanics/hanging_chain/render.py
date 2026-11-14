"""The hanging-chain renderer plots every Newton iterate over the inextensible catenary beside the unbalanced force residual."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, concatenate, stack
from numga.render import simplices
from numga.topology import Topology
from examples.util import charts
from examples.util.panels import offset_panel, overlay

early_shade = np.array([0.80, 0.80, 0.82])                                      # [3]
late_shade = np.array([0.15, 0.15, 0.18])                                       # [3]
chain_color = np.array([0.20, 0.40, 0.70])                                      # [3]
catenary_color = np.array([0.85, 0.30, 0.20])                                   # [3]
pin_color = np.array([0.15, 0.15, 0.18])                                        # [3]
catenary_dashes = (11.0, 5.0)
bead_radius = charts.MARKER_PX / 2
frame_margin = 0.05


def point_coordinates(points: Extensor) -> Extensor:
    """Pairing planar points with the coordinate lines reads their horizontal and vertical coordinates (`[2, ...] Scalar+`)."""
    mv = points.context.multivector
    return stack((mv.x, mv.y))[:, None] & points.reshape(-1)                    # [2, ...] Scalar+


@charts.style()
def iterates(chains: Extensor, residual: Extensor, catenary: Extensor, pins: Extensor) -> plt.Figure:
    """A two-panel figure plots the Newton iterates over the dashed catenary beside the maximum unbalanced force per step."""
    n_iterates, n_beads = chains.batch().shape
    n_samples = len(catenary)
    n_points = n_beads + 2
    beads = chains.batch()                                                      # [n_iterates, n_beads] Point-
    ends = pins.batch()                                                         # [2] Point-
    # Extend each iterate from the left pin through its beads to the right pin.
    polylines = concatenate(
        [ends[0].broadcast_to((n_iterates, 1)), beads, ends[1].broadcast_to((n_iterates, 1))],
        axis=1,
    )                                                                           # [n_iterates, n_beads + 2] Point-
    max_residual = np.asarray(residual.batch().max(axis=-1).to_array())         # [n_iterates]

    figure, (shape_panel, convergence_panel) = charts.subplots(
        640, 280, ncols=2, gridspec_kw={"width_ratios": [1.3, 1]}
    )
    convergence_panel.semilogy(np.arange(n_iterates), max_residual, color=chain_color, marker="o")
    convergence_panel.set_xlabel("Newton step")
    convergence_panel.set_ylabel("unbalanced force, in weights")
    convergence_panel.grid(True, which="major", alpha=0.3)

    # Frame the geometry panel around every iterate and the catenary.
    extent_coords = concatenate(
        [point_coordinates(polylines), point_coordinates(catenary)],
        axis=-1,
    )                                                                           # [2, n_iterates * n_points + n_samples] Scalar+
    low_bounds = extent_coords.min(axis=-1).to_array()                          # [2]
    high_bounds = extent_coords.max(axis=-1).to_array()                         # [2]
    padding = frame_margin * (high_bounds - low_bounds)                         # [2]
    shape_panel.set(
        xlim=(low_bounds[0] - padding[0], high_bounds[0] + padding[0]),
        ylim=(low_bounds[1] - padding[1], high_bounds[1] + padding[1]),
        aspect="equal",
    )
    shape_panel.axis("off")
    figure.tight_layout()

    mv = chains.context.multivector
    camera = offset_panel(shape_panel, mv.x, mv.y)
    shades = early_shade + np.linspace(0, 1, n_iterates)[:-1, None] * (late_shade - early_shade)    # [n_iterates - 1, 3]
    overlay(
        shape_panel,
        [
            simplices.strokes(
                polylines[:-1].reshape(-1),
                Topology.chains([n_points] * (n_iterates - 1)),
                np.repeat(shades, n_points - 1, axis=0),
                charts.LINE_PX,
                camera,
            ),
            # Draw the catenary dashed and twice as wide so it remains visible beneath the final iterate.
            simplices.strokes(
                catenary,
                Topology.chains([n_samples]),
                catenary_color,
                2 * charts.LINE_PX,
                camera,
                dashes=catenary_dashes,
                cap=simplices.Cap.BUTT,
            ),
            simplices.strokes(polylines[-1], Topology.chains([n_points]), chain_color, charts.LINE_PX, camera),
            simplices.markers(polylines[-1], chain_color, bead_radius, camera),
            simplices.markers(ends, pin_color, bead_radius, camera),
        ],
    )
    return figure
