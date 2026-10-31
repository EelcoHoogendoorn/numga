"""Spinor observables, chiral currents and charge conjugation."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera, panel
from examples.relativity.spinors import core

RIGHT_COLOUR, LEFT_COLOUR, INK = "#267fa9", "#d17c24", "#263b51"
# The interference ports and the conjugation eigenspaces share the chirality palette.
PORT_COLOUR, FIXED_COLOUR, NEGATED_COLOUR = RIGHT_COLOUR, RIGHT_COLOUR, LEFT_COLOUR
# The faint cross through the origin of a plot, and the colour of its lines' names.
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.45"
# The spin's circle and the light cone, faint, and the cone's inside; the samples round the circle.
RIM_COLOUR, CONE_FILL = "0.8", "0.97"
N_RIM_SAMPLES = 120
# The ports: their places across, their radius and their rims' outer radius, in the panel's units; their faces and
# rims.
PORT_PLACES, PORT_RADIUS, RIM_RADIUS = (-0.7, 0.7), 0.45, 0.47
PORT_FACE, PORT_RIM = "0.94", "0.7"
# The spin's arrow and the currents', bold, in pixels: their widths, and the length of the strokes of the chevrons at
# their tips, in widths of the arrow.
SPIN_WIDTH, CURRENT_WIDTH, CHEVRON_LENGTH = 3.0, 2 * charts.LINE_PX, 4.0


# --- plumbing -----------------------------------------------------------------------
def ink(colour: str) -> np.ndarray:
    """A colour's red, green and blue."""
    return np.array(to_rgb(colour))


def units(camera: Camera) -> tuple[core.Antivector, core.Antivector]:
    """The directions the camera reads as one across and one up: the duals of its planes, each over its own reading."""
    return tuple(plane.dual() / (plane & plane.dual()) for plane in (camera.across, camera.up))


def cross(ax: plt.Axes, camera: Camera, across: str, up: str) -> np.ndarray:
    """No scales and no frame: a faint cross through the origin, its lines named at their ends; the cross's
    picture."""
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    along, rise = units(camera)
    ends = stack((left * along, right * along, bottom * rise, top * rise))   # [4] Antivector-
    ax.annotate(across, (1, 0), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", color=LABEL_COLOUR)
    ax.annotate(up, (0, 1), xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=LABEL_COLOUR)
    ax.set_axis_off()
    return simplices.strokes(ends, simplices.Mesh.chains([2, 2]), ink(CROSS_COLOUR), charts.FRAME_PX, camera,
                             cap=simplices.Cap.BUTT)


def plain_axes(ax: plt.Axes, across: str, up: str) -> None:
    """Axes named by what runs along them, the quantity's zero as the horizontal one, light and without numbers."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.75")
    ax.spines["bottom"].set_position(("data", 0))
    ax.tick_params(color="0.75")
    ax.set(xticks=[], yticks=[], xlabel=across, ylabel=up)


def animate_rotation(
    spin: core.Antivector, bright: core.Scalar, dark: core.Scalar,
) -> Iterator[np.ndarray]:
    """The spin direction beside two interference outputs against a fixed reference."""
    mv = spin.context.multivector
    with charts.style():
        # The spin's circle beside the two ports, each about 190 pixels across.
        figure, (axis, ports) = charts.subplots(400, 190, ncols=2, layout="constrained")
        axis.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
        ports.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
        ports.set_axis_off()
        for centre, label in zip(PORT_PLACES, ("sum port", "difference port")):
            ports.text(centre, -0.7, label, ha="center")
        # The spin turns in the plane the camera sees, x across and z up.
        camera = panel(axis, mv.x, mv.z)
        along, rise = units(camera)
        turns = (mv.zx * (-np.linspace(0, 2 * np.pi, N_RIM_SAMPLES) / 2)).exp()
        circle = turns >> along                                               # [rim samples] Antivector-
        beneath = [cross(axis, camera, "x", "z"),
                   simplices.strokes(circle, simplices.Mesh.chains([N_RIM_SAMPLES]), ink(RIM_COLOUR), charts.FRAME_PX,
                                     camera)]
        shown = simplices.overlay(axis, beneath)
        # The ports as lamps, each lit as bright as its output, over its face within its rim.
        lamps = panel(ports, mv.x, mv.z)
        centres = stack([place * along for place in PORT_PLACES])             # [ports] Antivector-
        rim = RIM_RADIUS / lamps.pixel
        faces = [simplices.markers(centres, ink(PORT_RIM), rim, lamps),
                 simplices.markers(centres, ink(PORT_FACE), rim - charts.FRAME_PX, lamps)]
        lit = simplices.overlay(ports, faces)
        for direction, intensities in zip(spin, np.clip(stack((bright, dark), axis=-1).to_array(), 0, 1)):
            arrow = stack((0 * direction, direction))                         # [2] Antivector-
            shown.set_data(simplices.composited(beneath + [
                simplices.strokes(arrow, simplices.Mesh.chains([2]), ink(INK), SPIN_WIDTH, camera,
                                  chevrons=np.ones(1), chevron=CHEVRON_LENGTH * SPIN_WIDTH, cap=simplices.Cap.BUTT)]))
            lit.set_data(simplices.composited(faces + [
                simplices.markers(centres[port][None], ink(PORT_COLOUR), PORT_RADIUS / lamps.pixel, lamps,
                                  opacity=intensity) for port, intensity in enumerate(intensities)]))
            yield capture(figure)
        plt.close(figure)


@charts.style()
def draw_chirality(
    left: core.Antivector, right: core.Antivector, current: core.Antivector,
) -> plt.Figure:
    """Two null currents and their timelike sum in the z–t plane."""
    mv = current.context.multivector
    tips = stack((left, right, current))                                      # [currents] Antivector-
    extent = float((mv.t & tips).max().to_array()) * 1.15
    # Three arrows in the light cone's upper half, and room for their names and the cross's.
    figure, ax = charts.subplots(340, 200, layout="constrained")
    for (across, up), name, colour in zip(np.stack(((mv.z & tips).to_array(), (mv.t & tips).to_array()), axis=-1),
                                          ("left", "right", "Dirac"), (LEFT_COLOUR, RIGHT_COLOUR, INK)):
        ax.annotate(name, xy=(across, up), xytext=(7, 3), textcoords="offset points", color=colour)
    ax.set(xlim=(-extent, extent), ylim=(0, extent), aspect="equal")
    camera = panel(ax, mv.z, mv.t)
    lines = cross(ax, camera, "z", "t")
    along, rise = units(camera)
    # The light cone's upper half: the null directions either way along z, from the origin.
    cone = stack((extent * (rise - along), 0 * rise, extent * (rise + along)))   # [3] Antivector-
    # The currents bold, twice a curve's width, each from the origin, ending in a chevron.
    arrows = stack((0 * tips, tips), axis=-1).reshape(-1)                      # [currents * 2] Antivector-
    colours = np.stack([ink(colour) for colour in (LEFT_COLOUR, RIGHT_COLOUR, INK)])
    simplices.overlay(ax, [
        contours.picture(cone, simplices.Mesh(3, np.array([[0, 1, 2]])), np.zeros(3), ink(CONE_FILL), camera),
        lines,
        simplices.strokes(cone, simplices.Mesh.chains([3]), ink(RIM_COLOUR), charts.FRAME_PX, camera),
        simplices.strokes(arrows, simplices.Mesh.chains([2] * 3), colours, CURRENT_WIDTH, camera,
                          chevrons=np.ones(3), chevron=CHEVRON_LENGTH * CURRENT_WIDTH, cap=simplices.Cap.BUTT)])
    return figure


@charts.style()
def draw_majorana(
    phases: np.ndarray, even_density: core.Scalar, odd_density: core.Scalar,
) -> plt.Figure:
    """A phase turn trades density between the two charge-conjugation eigenspaces."""
    # Half the column: two crossing curves under their legend.
    figure, ax = charts.subplots(400, 220, layout="constrained")
    ax.plot(phases / np.pi, even_density.to_array(), color=FIXED_COLOUR, label="conjugation +1")
    ax.plot(phases / np.pi, odd_density.to_array(), color=NEGATED_COLOUR, label="conjugation −1")
    ax.set(xlim=(phases[0] / np.pi, phases[-1] / np.pi), ylim=(0, 1.05))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    plain_axes(ax, "phase angle", "density")
    # The phases that keep the Majorana condition, the only values marked.
    ax.set(xticks=[0, 1, 2], xticklabels=["0", "π", "2π"])
    return figure

