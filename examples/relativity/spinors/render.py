"""Renderers draw spinor interference ports, light-cone Weyl currents, and Majorana phase-sweep curves."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.relativity.spinors import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay

right_colour, left_colour, ink_colour = "#267fa9", "#d17c24", "#263b51"
# Interference ports and conjugation eigenspaces share the chiral palette.
port_colour, fixed_colour, negated_colour = right_colour, right_colour, left_colour
cross_colour, label_colour = "0.85", "0.45"
rim_colour, cone_fill = "0.8", "0.97"
n_rim_samples = 119
port_places, port_radius, rim_radius = (-0.7, 0.7), 0.45, 0.47
port_face, port_rim = "0.94", "0.7"
spin_width, current_width, chevron_length = 3.0, 2 * charts.LINE_PX, 4.0


# --- plumbing -------------------------------------------------------------------------
def rgb_colour(colour: str) -> np.ndarray:
    """Matplotlib colour strings convert to three-channel floating-point RGB arrays."""
    return np.array(to_rgb(colour))


def camera_units(camera: Camera) -> tuple[core.Antivector, core.Antivector]:
    """Dualizing the camera planes and dividing each by its self-pairing gives unit horizontal and vertical directions."""
    return tuple(plane.dual() / (plane & plane.dual()) for plane in (camera.across, camera.up))


def origin_cross(ax: plt.Axes, camera: Camera, horizontal_label: str, vertical_label: str) -> np.ndarray:
    """A faint named cross through the origin replaces framed coordinate axes and ticks."""
    (left_edge, right_edge), (bottom_edge, top_edge) = ax.get_xlim(), ax.get_ylim()
    horizontal, vertical = camera_units(camera)
    ends = stack((left_edge * horizontal, right_edge * horizontal, bottom_edge * vertical, top_edge * vertical))  # [4] Antivector-
    ax.annotate(horizontal_label, (1, 0), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", color=label_colour)
    ax.annotate(vertical_label, (0, 1), xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=label_colour)
    ax.set_axis_off()
    return simplices.strokes(ends, Topology.chains([2, 2]), rgb_colour(cross_colour), charts.FRAME_PX, camera,
                             cap=simplices.Cap.BUTT)


def plain_axes(ax: plt.Axes, horizontal_label: str, vertical_label: str) -> None:
    """Light unnumbered spines along zero and the left border label the two plotted quantities."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.75")
    ax.spines["bottom"].set_position(("data", 0))
    ax.tick_params(color="0.75")
    ax.set(xticks=[], yticks=[], xlabel=horizontal_label, ylabel=vertical_label)


def rotation_animation(
    states: Iterable[tuple[core.Antivector, core.Scalar, core.Scalar]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames showing the rotating spin direction beside the sum and difference interference ports."""
    mv = core.mv
    with charts.style():
        figure, (spin_axis, port_axis) = charts.subplots(400, 190, ncols=2, layout="constrained")
        spin_axis.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
        port_axis.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
        port_axis.set_axis_off()
        for centre, label in zip(port_places, ("sum port", "difference port")):
            port_axis.text(centre, -0.7, label, ha="center")
        camera = offset_panel(spin_axis, mv.x, mv.z)
        horizontal, vertical = camera_units(camera)
        rim_angles = np.linspace(0, 2 * np.pi, n_rim_samples, endpoint=False)
        rim_turns = (mv.zx * (-rim_angles / 2)).exp()                           # [n_rim_samples] Spinor+
        circle = rim_turns >> horizontal                                        # [n_rim_samples] Antivector-
        base_layers = [
            origin_cross(spin_axis, camera, "x", "z"),
            simplices.strokes(circle, Topology.loops([n_rim_samples]), rgb_colour(rim_colour), charts.FRAME_PX, camera),
        ]
        spin_overlay = overlay(spin_axis, base_layers)
        lamp_camera = offset_panel(port_axis, mv.x, mv.z)
        centres = stack([place * horizontal for place in port_places])          # [2] Antivector-
        rim_px = rim_radius / lamp_camera.pixel
        port_layers = [
            simplices.markers(centres, rgb_colour(port_rim), rim_px, lamp_camera),
            simplices.markers(centres, rgb_colour(port_face), rim_px - charts.FRAME_PX, lamp_camera),
        ]
        port_overlay = overlay(port_axis, port_layers)
        for spin_vector, sum_density, difference_density in states:
            arrow = stack((0 * spin_vector, spin_vector))                       # [2] Antivector-
            intensities = np.clip(stack((sum_density, difference_density)).to_array(), 0, 1)
            spin_overlay.set_data(drawing.composited(base_layers + [
                simplices.strokes(arrow, Topology.chains([2]), rgb_colour(ink_colour), spin_width, camera,
                                  chevrons=np.ones(1), chevron=chevron_length * spin_width, cap=simplices.Cap.BUTT)]))
            port_overlay.set_data(drawing.composited(port_layers + [
                simplices.markers(centres[port][None], rgb_colour(port_colour), port_radius / lamp_camera.pixel,
                                  lamp_camera, opacity=float(intensity))
                for port, intensity in enumerate(intensities)]))
            yield capture(figure)
        plt.close(figure)


@charts.style()
def chirality_figure(
    left_current: core.Antivector, right_current: core.Antivector, dirac_current: core.Antivector,
) -> plt.Figure:
    """Left and right null Weyl currents and their timelike Dirac sum sit inside the future light cone of the z–t plane."""
    mv = dirac_current.context.multivector
    tips = stack((left_current, right_current, dirac_current))                  # [3] Antivector-
    tip_coordinates = (stack((mv.z, mv.t)) & tips[..., None]).to_array()        # [3, 2]
    extent = float(tip_coordinates[:, 1].max()) * 1.15
    figure, ax = charts.subplots(340, 200, layout="constrained")
    for (horizontal_coord, vertical_coord), name, colour in zip(
        tip_coordinates, ("left", "right", "Dirac"), (left_colour, right_colour, ink_colour),
    ):
        ax.annotate(name, xy=(horizontal_coord, vertical_coord), xytext=(7, 3), textcoords="offset points", color=colour)
    ax.set(xlim=(-extent, extent), ylim=(0, extent), aspect="equal")
    camera = offset_panel(ax, mv.z, mv.t)
    cross_layer = origin_cross(ax, camera, "z", "t")
    horizontal, vertical = camera_units(camera)
    # The upper half of the light cone spans the forward and backward null rays from the origin.
    cone = stack((extent * (vertical - horizontal), 0 * vertical, extent * (vertical + horizontal)))  # [3] Antivector-
    arrows = stack((0 * tips, tips), axis=-1).reshape(-1)                       # [6] Antivector-
    colours = np.stack([rgb_colour(colour) for colour in (left_colour, right_colour, ink_colour)])
    overlay(ax, [
        contours.picture(cone, Topology(3, np.array([[0, 1, 2]])), np.zeros(3), rgb_colour(cone_fill), camera),
        cross_layer,
        simplices.strokes(cone, Topology.chains([3]), rgb_colour(rim_colour), charts.FRAME_PX, camera),
        simplices.strokes(arrows, Topology.chains([2] * 3), colours, current_width, camera,
                          chevrons=np.ones(3), chevron=chevron_length * current_width, cap=simplices.Cap.BUTT)])
    return figure


@charts.style()
def majorana_figure(
    phases: np.ndarray, fixed_density: core.Scalar, negated_density: core.Scalar,
) -> plt.Figure:
    """Crossing density curves trace how a phase turn moves a Majorana state between the two charge-conjugation eigenspaces."""
    figure, ax = charts.subplots(400, 220, layout="constrained")
    ax.plot(phases / np.pi, fixed_density.to_array(), color=fixed_colour, label="conjugation +1")
    ax.plot(phases / np.pi, negated_density.to_array(), color=negated_colour, label="conjugation −1")
    ax.set(xlim=(phases[0] / np.pi, phases[-1] / np.pi), ylim=(0, 1.05))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    plain_axes(ax, "phase angle", "density")
    ax.set(xticks=[0, 1, 2], xticklabels=["0", "π", "2π"])
    return figure
