"""Polar emission plots in the vertical coil plane display electron rates alongside cross-and-dot coil cross-sections."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from numga.topology import Topology
from numga.render import contours, simplices
from examples.util.panels import offset_panel, overlay
from examples.orientation.wu_experiment.core import Direction, Scalar, Spin

coil_radius = 0.55
# Wire marker radius and rim stroke width in pixels, and cross/dot fractions indicating current into and out of the page.
wire_radius, wire_width = 9.7, 2.5
cross_share, dot_share = 0.5, 0.3
coil_colour, curve_colour, axis_colour = (to_rgb(colour) for colour in ("#262b32", "#df7246", "#aab4bf"))
# Polar curve width, dashed symmetry-axis width and dash pattern in pixels, and lobe fill opacity.
curve_width, axis_width, axis_dashes = 2.5, 1.0, (4.0, 2.5)
fill_opacity = 0.25


def experiment_figure(spins: Spin, directions: Direction, rates: Scalar) -> plt.Figure:
    """Polar emission curves for `rates` (`[n_cases, n_directions] Scalar+`) along `directions`
    (`[n_directions] Direction-`) and coil `spins` (`[n_cases] Spin-`) render as side-by-side `xz` cross-sections."""
    mv = directions.context.multivector
    # Pairing each coil patch with the vertical direction reads the current circulation sense.
    senses = (spins & mv.z.dual()).to_array()                                     # [n_cases]
    polar_points = rates * directions                                             # [n_cases, n_directions] Direction-
    limit = float((stack((mv.x, mv.z))[:, None] & polar_points.reshape(-1)).abs().max().to_array()) * 1.1
    n_directions = polar_points.shape[-1]
    # Cone triangulation fanning from the origin to fill each closed polar lobe.
    origin = mv.x.dual() * np.zeros(1)                                            # [1] Direction-
    loop = Topology.loops([n_directions])
    fan = loop.cone()
    symmetry_axis = mv.z.dual() * np.array([-limit, limit])                       # [2] Direction-
    # Diagonal cross arms marking current into the page.
    cross_arms = mv.x.dual() + mv.z.dual() * np.array([1, -1])                    # [2] Direction-

    figure, panels = plt.subplots(1, len(senses), figsize=(9.2, 3.6), dpi=100)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.05)
    for ax, sense, curve in zip(panels, senses, polar_points):
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
        camera = offset_panel(ax, mv.x, mv.z)
        # With `x` right and `z` up, `+y` points into the page so counterclockwise current from above enters at `+x`.
        inward_wire = mv.x.dual() * (np.sign(sense) * coil_radius)                # [] Direction-
        outward_wire = mv.x.dual() * (-np.sign(sense) * coil_radius)              # [] Direction-
        wire_centres = concatenate([inward_wire[None], outward_wire[None]])       # [2] Direction-
        arm_reach = cross_share * wire_radius * camera.pixel
        cross_points = concatenate([inward_wire - cross_arms * arm_reach,
                                    inward_wire + cross_arms * arm_reach])        # [4] Direction-
        overlay(ax, [
            contours.picture(concatenate([curve, origin]), fan, np.zeros(n_directions + 1), curve_colour, camera,
                             opacity=fill_opacity),
            simplices.strokes(curve, loop, curve_colour, curve_width, camera),
            simplices.strokes(symmetry_axis, Topology.chains([2]), axis_colour, axis_width, camera, dashes=axis_dashes,
                              cap=simplices.Cap.BUTT),
            simplices.markers(wire_centres, coil_colour, wire_radius + wire_width / 2, camera),
            simplices.markers(wire_centres, np.ones(3), wire_radius - wire_width / 2, camera),
            simplices.strokes(cross_points, Topology(4, np.array([[0, 2], [1, 3]])), coil_colour, wire_width, camera,
                              cap=simplices.Cap.BUTT),
            simplices.markers(outward_wire[None], coil_colour, dot_share * wire_radius, camera)])
    return figure
