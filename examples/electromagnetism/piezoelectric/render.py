"""The two bonded piezoelectric layers, their electrodes, and the strain of the bent strip."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import concatenate, stack
from examples.electromagnetism.piezoelectric import core
from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera, offset_panel

Topology = core.Topology

compression_colour = "#286580"
extension_colour = "#df6435"
interface_colour = "#53636b"
rest_colour = "#c5cdd1"
strain_colours = LinearSegmentedColormap.from_list(
    "piezoelectric_strain", (compression_colour, "#f5f4ef", extension_colour))
electrode_colours = LinearSegmentedColormap.from_list(
    "piezoelectric_voltage", (compression_colour, "#8b969b", extension_colour))
# The clamp's face and edge colours, its edge's width, and how many hatches cross it.
clamp_face, clamp_edge, clamp_width, n_hatches = "#edf0f0", "#a0aaaf", 1.1, 4
# Widths, dashes and radii in pixels: the device's rest outline, its interface and tip, its electrodes and its
# poling arrows, with the length of the strokes of the chevrons at their tips; the mesh's edges, electrodes, interface
# and clamped nodes.
rest_width, rest_dashes = 1.25, (5.5, 5.5)
bond_width, end_width, electrode_width = 1.25, 1.4, 2.1
poling_width, poling_chevron, poling_opacity = 1.0, 4.0, 0.75
mesh_width, mesh_electrode_width, mesh_interface_width, clamped_radius = 0.8, 3.5, 1.7, 1.5


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """A named colour's red, green and blue."""
    return np.array(to_rgb(colour))


def bounds(points: core.Point) -> tuple[core.Scalar, core.Scalar]:
    """The lowest and the highest readings of points across x and y, `[2] Scalar+` each: the box about them."""
    mv = points.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & points.batch()                   # [2, points] Scalar+
    return readings.min(axis=-1), readings.max(axis=-1)


def scalars(values: core.Scalar) -> np.ndarray:
    """One scalar per vertex or triangle, at the drawing boundary."""
    return np.asarray(values.batch().to_array())


def chain(points: core.Point, colour: str, width: float, camera: Camera, **style) -> np.ndarray:
    """A polyline through points `[points] Point-`, in order, as a stroke."""
    return simplices.strokes(points, Topology.chains([len(points)]), rgb(colour), width, camera, **style)


def clamp(low: core.Point, high: core.Point, depth: core.Point, camera: Camera) -> list[np.ndarray]:
    """A hatched clamp, a block from its face between `low` and `high` back by `depth`: its fill, outline and hatches,
    rising from its back to its face at forty-five degrees."""
    corners = stack((low - depth, low, high, high - depth))                     # [4] Point-
    span = high - low                                                           # [] Point-
    # The depth as a share of the face's length: how far up the face a hatch rises across the block.
    rise = float((depth.squared() / span.squared()).square_root().to_array())
    starts = np.linspace(0, 1 - rise, n_hatches)
    hatches = stack((low - depth + span * starts, low + span * (starts + rise)), axis=-1).reshape(-1)   # [hatches * 2] Point-
    # The block, two triangles, outlined by their border.
    block = Topology(4, np.array([[0, 1, 2], [0, 2, 3]]))
    return [contours.picture(corners, block, np.zeros(4), rgb(clamp_face), camera),
            simplices.strokes(hatches, Topology.chains([2] * n_hatches), rgb(clamp_edge), clamp_width, camera,
                              cap=simplices.Cap.BUTT),
            simplices.strokes(corners, block.border, rgb(clamp_edge), clamp_width, camera)]


class DeviceView:
    """The strip over its rest outline and clamp, pinned once and redrawn per state, with fixed deformation and colour
    scales across frames."""

    def __init__(self, topology: core.Strip, vertices: core.Point, areas: core.Scalar, displacement_scale: float,
                 voltage_scale: float, strain_scale: float) -> None:
        self.topology, self.vertices = topology, vertices
        low, high = bounds(vertices)                                            # [2] Scalar+, [2] Scalar+
        self.displacement_scale = displacement_scale
        self.voltage_norm = Normalize(-voltage_scale / 2, voltage_scale / 2)
        self.strain_norm = Normalize(-strain_scale, strain_scale)
        # Each face's area shared among its corners, for area-weighted readings of face values at the vertices.
        self.corner_shares = ~topology.incidence[2, 0].mean(axis=0)             # Scalar+[vertices <- faces]
        self.areas, self.vertex_areas = areas, self.corner_shares * areas
        length, self.thickness = high - low                                     # [] Scalar+, [] Scalar+
        middle = (low[1] + high[1]) / 2                                         # [] Scalar+

        # The strip's electrodes and interface run in order along it, and its tip from the bottom up.
        self.poles = np.linspace(0, len(topology.top) - 1, 9, dtype=int)[1:-1]
        # Each poling arrow a chain of two points, its tail and its tip, with a chevron at its tip.
        self.arrows = Topology.chains([2] * (2 * len(self.poles)))
        # The electrodes, top and bottom, as chains; each edge coloured by its electrode.
        self.electrodes = Topology.chains([len(topology.top), len(topology.bottom)])
        self.electrode_edges = np.array([len(topology.top) - 1, len(topology.bottom) - 1])

        self.figure, ax = plt.subplots(figsize=(5.25, 1.4), dpi=100)
        self.figure.subplots_adjust(0.015, 0.025, 0.995, 0.975)
        half_height = stack((3.3 * self.thickness, 0.11 * length)).max()        # [] Scalar+
        limits = stack((low[0] - 0.045 * length, high[0] + 0.025 * length,
                        middle - half_height, middle + half_height)).to_array()
        ax.set(xlim=limits[:2], ylim=limits[2:], aspect="equal")
        ax.set_axis_off()
        self.view = offset_panel(ax, core.mv.x, core.mv.y)
        # The clamp, twice the strip's thickness, its face along the strip's clamped edge.
        foot, head = (topology.reading(topology.clamped[[0, -1]], 0) * vertices).batch()   # [] Point-, [] Point-
        low, high = foot - (head - foot) / 2, head + (head - foot) / 2          # [] Point-, [] Point-
        outline = topology.reading(np.concatenate((topology.bottom, topology.tip, topology.top[::-1])), 0)   # Scalar+[outline <- vertices]
        rest_outline = (outline * vertices).batch()                             # [outline] Point-
        self.under = [chain(rest_outline, rest_colour, rest_width, self.view, dashes=rest_dashes),
                      *clamp(low, high, core.axis * (0.025 * length), self.view)]
        self.face = chain(stack((low, high)), interface_colour, rest_width, self.view)
        # The strained body, a picture of the deformed mesh shaded anew at each state.
        self.image = simplices.overlay(ax, self.under)

    def update(self, displacement: core.Point, potential: core.Scalar,
               axial_strain: core.Scalar) -> plt.Figure:
        deformed = self.vertices + self.displacement_scale * displacement           # Point-[vertices]
        # Area-weighted nodal readings give a continuous display of the piecewise-constant element strain.
        # The strain comes per triangle, a batch over the faces; as a field over them it reaches the vertices.
        strain = scalars(self.corner_shares * (self.areas * axial_strain.field()) / self.vertex_areas)
        places = deformed.batch()                                               # [vertices] Point-
        body = contours.picture(places, self.topology, strain, strain_colours(self.strain_norm(strain))[:, :3], self.view)

        top, bottom = self.topology.reading(self.topology.top, 0), self.topology.reading(self.topology.bottom, 0)
        electrode_potential = stack(((top * potential).sites.mean(), (bottom * potential).sites.mean())).to_array()
        # The two electrode colours read their voltage difference, independent of the potential's zero.
        electrode_potential -= electrode_potential.mean()
        electrode_ink = np.repeat(electrode_colours(self.voltage_norm(electrode_potential))[:, :3],
                                  self.electrode_edges, axis=0)                 # [electrode edges, 3]

        # The poling arrows, across the strip from electrode to electrode: up in the upper layer, a quarter of the way
        # down from the top, and down in the lower layer, a quarter of the way up from the bottom.
        upper = (self.topology.reading(self.topology.top[self.poles], 0) * deformed).batch()  # [poles] Point-
        lower = (self.topology.reading(self.topology.bottom[self.poles], 0) * deformed).batch()  # [poles] Point-
        separation = upper - lower                                                  # [poles] Point-
        across = separation / separation.squared().square_root()                    # [poles] Point-
        centres = concatenate((0.75 * upper + 0.25 * lower, 0.25 * upper + 0.75 * lower))   # [arrows] Point-
        arrows = 0.32 * self.thickness * concatenate((across, -across))                     # [arrows] Point-
        poling = stack((centres - arrows / 2, centres + arrows / 2), axis=-1).reshape(-1)   # [arrows * 2] Point-
        poling_arrows = simplices.strokes(poling, self.arrows, rgb(interface_colour), poling_width, self.view,
                                          chevrons=np.ones(2 * len(self.poles)), chevron=poling_chevron,
                                          opacity=poling_opacity)

        self.image.set_data(simplices.composited(self.under + [
            body,
            chain((self.topology.reading(self.topology.interface, 0) * deformed).batch(), interface_colour, bond_width,
                  self.view),
            chain((self.topology.reading(self.topology.tip, 0) * deformed).batch(), interface_colour, end_width, self.view),
            self.face,
            simplices.strokes((self.topology.reading(np.concatenate((self.topology.top, self.topology.bottom)), 0) * deformed).batch(),
                              self.electrodes, electrode_ink, electrode_width, self.view),
            poling_arrows]))
        return self.figure


def draw_mesh(topology: core.Strip, vertices: core.Point) -> plt.Figure:
    """The clamped end of the strip's mesh: its triangles, the electrodes along its top and bottom, the bonded
    interface between its layers, and the clamp holding its first column of nodes."""
    low, high = bounds(vertices)                                                # [2] Scalar+, [2] Scalar+
    thickness = high[1] - low[1]                                                # [] Scalar+
    # The clamped end, three thicknesses of the strip along it, with a margin around it.
    limits = stack((low[0] - 0.12 * thickness, low[0] + 3 * thickness,
                    low[1] - 0.12 * thickness, high[1] + 0.12 * thickness)).to_array()
    figure, ax = plt.subplots(figsize=(3.0, 1.45), dpi=100)
    figure.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=limits[:2], ylim=limits[2:], aspect="equal")
    ax.set_axis_off()
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    places = vertices.batch()                                                   # [vertices] Point-
    foot, head = (topology.reading(topology.clamped[[0, -1]], 0) * vertices).batch()   # [] Point-, [] Point-
    margin = 0.12 * (head - foot)                                               # [] Point-
    # The electrodes and the interface run in order along the strip.
    simplices.overlay(ax, [
        *clamp(foot - margin, head + margin, core.axis * (0.12 * thickness), camera),
        simplices.strokes(places, topology, rgb(rest_colour), mesh_width, camera),
        *[chain((topology.reading(nodes, 0) * vertices).batch(), colour, width, camera, cap=simplices.Cap.BUTT)
          for nodes, colour, width in ((topology.top, extension_colour, mesh_electrode_width),
                                       (topology.bottom, compression_colour, mesh_electrode_width),
                                       (topology.interface, interface_colour, mesh_interface_width))],
        simplices.markers((topology.reading(topology.clamped, 0) * vertices).batch(), rgb(interface_colour), clamped_radius,
                          camera)])
    return figure


def draw_device(topology: core.Strip, vertices: core.Point, areas: core.Scalar, displacement: core.Point, potential: core.Scalar,
                axial_strain: core.Scalar, displacement_scale: float,
                voltage_scale: float, strain_scale: float) -> plt.Figure:
    """The clamped strip, with signed axial strain on its layers and signed voltage on its electrodes."""
    view = DeviceView(topology, vertices, areas, displacement_scale, voltage_scale, strain_scale)
    return view.update(displacement, potential, axial_strain)


def animate(topology: core.Strip, vertices: core.Point, areas: core.Scalar, states: Iterator[tuple[core.Point, core.Scalar, core.Scalar]],
            displacement_scale: float, voltage_scale: float, strain_scale: float) -> Iterator[np.ndarray]:
    """Successive equilibria, with the camera and all three scales held fixed."""
    view = DeviceView(topology, vertices, areas, displacement_scale, voltage_scale, strain_scale)
    try:
        for state in states:
            yield capture(view.update(*state))
    finally:
        plt.close(view.figure)
