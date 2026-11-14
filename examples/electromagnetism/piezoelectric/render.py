"""Renders the bonded piezoelectric strip, its electrodes, and its axial bending strain."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import concatenate, stack
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.electromagnetism.piezoelectric import core
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay

if TYPE_CHECKING:
    from examples.electromagnetism.piezoelectric.scenarios import Strip

compression_colour = "#286580"
extension_colour = "#df6435"
interface_colour = "#53636b"
rest_colour = "#c5cdd1"
strain_colours = LinearSegmentedColormap.from_list(
    "piezoelectric_strain", (compression_colour, "#f5f4ef", extension_colour))
electrode_colours = LinearSegmentedColormap.from_list(
    "piezoelectric_voltage", (compression_colour, "#8b969b", extension_colour))
clamp_face, clamp_edge, clamp_width, n_hatches = "#edf0f0", "#a0aaaf", 1.1, 4
rest_width, rest_dashes = 1.25, (5.5, 5.5)
bond_width, end_width, electrode_width = 1.25, 1.4, 2.1
poling_width, poling_chevron, poling_opacity, n_poles = 1.0, 4.0, 0.75, 7
edge_width, strip_electrode_width, strip_interface_width, clamp_marker_radius = 0.8, 3.5, 1.7, 1.5


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """Converts a named matplotlib colour string into an RGB float array (`[3]`)."""
    return np.array(to_rgb(colour))


def bounds(points: core.Point) -> tuple[core.Scalar, core.Scalar]:
    """Pairings against the spatial axes read the minimum and maximum coordinates (`[2] Scalar+`) across `points`."""
    mv = points.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & points.batch()                      # [2, ...] Scalar+
    return readings.min(axis=-1), readings.max(axis=-1)


def scalars(values: core.Scalar) -> np.ndarray:
    """Unpacks scalar extensor values into a NumPy array at the plotting boundary."""
    return np.asarray(values.batch().to_array())


def chain(points: core.Point, colour: str, width: float, camera: Camera, **style) -> np.ndarray:
    """Renders an ordered polyline stroke through `points` (`[...] Point-`)."""
    return simplices.strokes(points, Topology.chains([len(points)]), rgb(colour), width, camera, **style)


def clamp(low: core.Point, high: core.Point, depth: core.Point, camera: Camera) -> list[np.ndarray]:
    """Renders a hatched support block extending behind the clamped edge between `low` and `high` by `depth`."""
    corners = stack((low - depth, low, high, high - depth))                       # [4] Point-
    span = high - low                                                             # Point-
    # Express the block depth as a fraction of the clamped face length so hatches cross at forty-five degrees.
    rise = float((depth.antinorm() / span.antinorm()).to_array())
    starts = np.linspace(0, 1 - rise, n_hatches)
    hatches = stack((low - depth + span * starts, low + span * (starts + rise)), axis=-1).reshape(-1)  # [n_hatches * 2] Point-
    block = Topology(4, np.array([[0, 1, 2], [0, 2, 3]]))
    return [
        contours.picture(corners, block, np.zeros(4), rgb(clamp_face), camera),
        simplices.strokes(hatches, Topology.chains([2] * n_hatches), rgb(clamp_edge), clamp_width, camera,
                          cap=simplices.Cap.BUTT),
        simplices.strokes(corners, block.border, rgb(clamp_edge), clamp_width, camera),
    ]


class DeviceView:
    """Holds the static clamp, reference outline and fixed colour scales for redrawing the deformed bimorph."""

    def __init__(self, topology: Strip, vertices: core.Point, areas: core.Scalar, displacement_scale: float,
                 voltage_scale: float, strain_scale: float) -> None:
        V, E, F = topology.n_elements
        self.topology, self.vertices = topology, vertices
        low, high = bounds(vertices)                                              # [2] Scalar+, [2] Scalar+
        self.displacement_scale = displacement_scale
        self.voltage_norm = Normalize(-voltage_scale / 2, voltage_scale / 2)
        self.strain_norm = Normalize(-strain_scale, strain_scale)
        # Share each triangle's area equally across its three corners for area-weighted vertex strain shading.
        self.corner_shares = topology.incidence[2, 0].mean(axis=0).adjugate()     # Scalar+[V <- F]
        self.areas, self.vertex_areas = areas, self.corner_shares * areas
        strip_length, self.thickness = high - low                                 # Scalar+, Scalar+
        middle = (low[1] + high[1]) / 2                                           # Scalar+

        self.pole_indices = np.linspace(0, len(topology.top_nodes) - 1, n_poles + 2, dtype=int)[1:-1]
        self.arrows = Topology.chains([2] * (2 * n_poles))
        self.electrodes = Topology.chains([len(topology.top_nodes), len(topology.bottom_nodes)])
        self.electrode_edges = np.array([len(topology.top_nodes) - 1, len(topology.bottom_nodes) - 1])

        self.figure, ax = plt.subplots(figsize=(5.25, 1.4), dpi=100)
        self.figure.subplots_adjust(0.015, 0.025, 0.995, 0.975)
        half_height = stack((3.3 * self.thickness, 0.11 * strip_length)).max()    # Scalar+
        limits = stack((low[0] - 0.045 * strip_length, high[0] + 0.025 * strip_length,
                        middle - half_height, middle + half_height)).to_array()
        ax.set(xlim=limits[:2], ylim=limits[2:], aspect="equal")
        ax.set_axis_off()
        mv = vertices.context.multivector
        self.view = offset_panel(ax, mv.x, mv.y)
        foot, head = (topology.reading(topology.clamp_nodes[[0, -1]], 0) * vertices).batch()  # Point-, Point-
        clamp_low, clamp_high = foot - (head - foot) / 2, head + (head - foot) / 2  # Point-, Point-
        outline_nodes = np.concatenate((topology.bottom_nodes, topology.tip_nodes, topology.top_nodes[::-1]))
        n_outline = len(outline_nodes)
        rest_outline = (topology.reading(outline_nodes, 0) * vertices).batch()    # [n_outline] Point-
        self.backdrop = [
            chain(rest_outline, rest_colour, rest_width, self.view, dashes=rest_dashes),
            *clamp(clamp_low, clamp_high, mv.x.dual() * (0.025 * strip_length), self.view),
        ]
        self.face = chain(stack((clamp_low, clamp_high)), interface_colour, rest_width, self.view)
        self.image = overlay(ax, self.backdrop)

    def update(self, displacement: core.Point, potential: core.Scalar,
               axial_strain: core.Scalar) -> plt.Figure:
        """Updates the rendered strip figure for a single `(displacement, potential, axial_strain)` state."""
        V, E, F = self.topology.n_elements
        deformed_vertices = self.vertices + self.displacement_scale * displacement  # Point-[V]
        nodal_strain = scalars(self.corner_shares * (self.areas * axial_strain.field()) / self.vertex_areas)
        places = deformed_vertices.batch()                                        # [V] Point-
        body = contours.picture(
            places, self.topology, nodal_strain, strain_colours(self.strain_norm(nodal_strain))[:, :3], self.view,
        )

        top = self.topology.reading(self.topology.top_nodes, 0)
        bottom = self.topology.reading(self.topology.bottom_nodes, 0)
        electrode_potential = stack(((top * potential).sites.mean(), (bottom * potential).sites.mean())).to_array()
        # Center the two terminal potentials so the electrode colours depend only on the terminal voltage difference.
        electrode_potential -= electrode_potential.mean()
        n_electrode_edges = int(self.electrode_edges.sum())
        electrode_ink = np.repeat(
            electrode_colours(self.voltage_norm(electrode_potential))[:, :3], self.electrode_edges, axis=0,
        )                                                                         # [n_electrode_edges, 3]

        n_arrows = 2 * n_poles
        upper = (self.topology.reading(self.topology.top_nodes[self.pole_indices], 0) * deformed_vertices).batch()     # [n_poles] Point-
        lower = (self.topology.reading(self.topology.bottom_nodes[self.pole_indices], 0) * deformed_vertices).batch()  # [n_poles] Point-
        separation = upper - lower                                                # [n_poles] Point-
        through_thickness = separation / separation.antinorm()                    # [n_poles] Point-
        centres = concatenate((0.75 * upper + 0.25 * lower, 0.25 * upper + 0.75 * lower))  # [n_arrows] Point-
        arrow_spans = 0.32 * self.thickness * concatenate((through_thickness, -through_thickness))  # [n_arrows] Point-
        poling_strokes = stack((centres - arrow_spans / 2, centres + arrow_spans / 2), axis=-1).reshape(-1)  # [n_arrows * 2] Point-
        poling_arrows = simplices.strokes(
            poling_strokes, self.arrows, rgb(interface_colour), poling_width, self.view,
            chevrons=np.ones(n_arrows), chevron=poling_chevron, opacity=poling_opacity,
        )

        electrode_nodes = np.concatenate((self.topology.top_nodes, self.topology.bottom_nodes))
        self.image.set_data(drawing.composited(self.backdrop + [
            body,
            chain((self.topology.reading(self.topology.interface_nodes, 0) * deformed_vertices).batch(),
                  interface_colour, bond_width, self.view),
            chain((self.topology.reading(self.topology.tip_nodes, 0) * deformed_vertices).batch(),
                  interface_colour, end_width, self.view),
            self.face,
            simplices.strokes((self.topology.reading(electrode_nodes, 0) * deformed_vertices).batch(),
                              self.electrodes, electrode_ink, electrode_width, self.view),
            poling_arrows,
        ]))
        return self.figure


def draw_strip(topology: Strip, vertices: core.Point) -> plt.Figure:
    """Draws the clamped end of the triangulated strip with its electrodes, central bond and pinned vertices."""
    V, E, F = topology.n_elements
    low, high = bounds(vertices)                                                  # [2] Scalar+, [2] Scalar+
    strip_thickness = high[1] - low[1]                                            # Scalar+
    limits = stack((low[0] - 0.12 * strip_thickness, low[0] + 3 * strip_thickness,
                    low[1] - 0.12 * strip_thickness, high[1] + 0.12 * strip_thickness)).to_array()
    figure, ax = plt.subplots(figsize=(3.0, 1.45), dpi=100)
    figure.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=limits[:2], ylim=limits[2:], aspect="equal")
    ax.set_axis_off()
    mv = vertices.context.multivector
    camera = offset_panel(ax, mv.x, mv.y)
    places = vertices.batch()                                                     # [V] Point-
    foot, head = (topology.reading(topology.clamp_nodes[[0, -1]], 0) * vertices).batch()  # Point-, Point-
    margin = 0.12 * (head - foot)                                                 # Point-
    overlay(ax, [
        *clamp(foot - margin, head + margin, mv.x.dual() * (0.12 * strip_thickness), camera),
        simplices.strokes(places, topology, rgb(rest_colour), edge_width, camera),
        *[chain((topology.reading(nodes, 0) * vertices).batch(), colour, stroke_width, camera, cap=simplices.Cap.BUTT)
          for nodes, colour, stroke_width in ((topology.top_nodes, extension_colour, strip_electrode_width),
                                              (topology.bottom_nodes, compression_colour, strip_electrode_width),
                                              (topology.interface_nodes, interface_colour, strip_interface_width))],
        simplices.markers((topology.reading(topology.clamp_nodes, 0) * vertices).batch(),
                          rgb(interface_colour), clamp_marker_radius, camera),
    ])
    return figure


def draw_device(topology: Strip, vertices: core.Point, areas: core.Scalar, displacement: core.Point,
                potential: core.Scalar, axial_strain: core.Scalar, displacement_scale: float,
                voltage_scale: float, strain_scale: float) -> plt.Figure:
    """Draws a single equilibrium state of the clamped bimorph strip."""
    view = DeviceView(topology, vertices, areas, displacement_scale, voltage_scale, strain_scale)
    return view.update(displacement, potential, axial_strain)


def animate(topology: Strip, vertices: core.Point, areas: core.Scalar,
            displacement_scale: float, voltage_scale: float, strain_scale: float,
            states: Iterator[tuple[core.Point, core.Scalar, core.Scalar]]) -> Iterator[np.ndarray]:
    """Yields rendered RGB frames across `states` with fixed camera framing and colour scales."""
    view = DeviceView(topology, vertices, areas, displacement_scale, voltage_scale, strain_scale)
    try:
        for state in states:
            yield capture(view.update(*state))
    finally:
        plt.close(view.figure)
