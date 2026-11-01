"""Emission patterns around a coil, in the plane that holds the coil's axis, drawn on the GPU in the algebra of
space, read by the planes x and z."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import panel
from examples.orientation.wu_experiment.core import Direction, Scalar, Spin

coil_radius = 0.55
# The coil where the plane of the panels cuts it, a wire at each side: the wire's radius and its rim's width in
# pixels, and the shares of the radius the cross and the dot take that mark its current going into and coming out of
# the page.
wire_radius, wire_width = 9.7, 2.5
cross_share, dot_share = 0.5, 0.3
coil_colour, curve_colour, axis_colour = (to_rgb(colour) for colour in ("#262b32", "#df7246", "#aab4bf"))
# The widths of the emission curve and the coil's axis, and the axis's dashes and gaps, in pixels; how much the
# emission's fill covers.
curve_width, axis_width, axis_dashes = 2.5, 1.0, (4.0, 2.5)
fill_opacity = 0.25


def draw_experiment(spins: Spin, directions: Direction, rates: Scalar) -> plt.Figure:
    """One panel per case `[cases]`: the emission rate along each direction `[cases, directions]` as a
    polar curve in the xz plane, and the coil cut by that plane, with the sense of its current."""
    mv = directions.context.multivector
    # The sense of each coil's current, read by pairing its patch with the vertical direction; the polar curve, each
    # direction drawn out to its rate, framed by its largest reading across x or up z.
    senses = (spins & mv.z.dual()).to_array()                                     # [cases]
    polar = rates * directions                                                    # [cases, directions] Direction-
    limit = float((stack((mv.x, mv.z))[:, None] & polar.reshape(-1)).abs().max().to_array()) * 1.1
    n_directions = polar.shape[-1]
    # The emission filled as triangles fanning out from the origin, first of the points, across each step of the
    # closed curve.
    origin = mv.x.dual() * np.zeros(1)                                            # [1] Direction-
    steps = np.arange(1, n_directions)
    fan = simplices.Topology(n_directions + 1, np.stack([np.zeros(n_directions - 1, dtype=int), steps, steps + 1], -1))
    axis = mv.z.dual() * np.array([-limit, limit])                                # [ends] Direction-
    # The cross's arms, a quarter turn apart, each across the wire either way, reaching as far across as up.
    arms = mv.x.dual() + mv.z.dual() * np.array([1.0, -1.0])                      # [arms] Direction-

    figure, panels = plt.subplots(1, len(senses), figsize=(9.2, 3.6), dpi=100)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
    for ax, sense, curve in zip(panels, senses, polar):
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
        camera = panel(ax, mv.x, mv.z)
        # With x to the right and z up, +y runs into the page: a current counterclockwise seen from above goes
        # into the page at +x and comes out of it at -x.
        inward, outward = mv.x.dual() * (np.sign(sense) * coil_radius), mv.x.dual() * (-np.sign(sense) * coil_radius)
        wires = concatenate([inward[None], outward[None]])                       # [wires] Direction-
        reach = cross_share * wire_radius * camera.pixel
        cross = concatenate([inward - arms * reach, inward + arms * reach])       # [2 * arms] Direction-
        simplices.overlay(ax, [
            contours.picture(concatenate([origin, curve]), fan, np.zeros(n_directions + 1), curve_colour, camera,
                             opacity=fill_opacity),
            simplices.strokes(curve, simplices.Topology.chains([n_directions]), curve_colour, curve_width, camera),
            simplices.strokes(axis, simplices.Topology.chains([2]), axis_colour, axis_width, camera, dashes=axis_dashes,
                              cap=simplices.Cap.BUTT),
            simplices.markers(wires, coil_colour, wire_radius + wire_width / 2, camera),
            simplices.markers(wires, np.ones(3), wire_radius - wire_width / 2, camera),
            simplices.strokes(cross, simplices.Topology(4, np.array([[0, 2], [1, 3]])), coil_colour, wire_width, camera,
                              cap=simplices.Cap.BUTT),
            simplices.markers(outward[None], coil_colour, dot_share * wire_radius, camera)])
    return figure
