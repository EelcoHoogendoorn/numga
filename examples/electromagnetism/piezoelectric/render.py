"""The two bonded piezoelectric layers, their electrodes, and the strain of the bent strip."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation

from numga import stack
from examples.electromagnetism.piezoelectric import core
from examples.util.animation import capture


compression_colour = "#286580"
extension_colour = "#df6435"
interface_colour = "#53636b"
rest_colour = "#c5cdd1"
strain_colours = LinearSegmentedColormap.from_list(
    "piezoelectric_strain", (compression_colour, "#f5f4ef", extension_colour))
electrode_colours = LinearSegmentedColormap.from_list(
    "piezoelectric_voltage", (compression_colour, "#8b969b", extension_colour))


def coordinates(points: core.Point) -> np.ndarray:
    """Positions read by the slices across x and y, with the drawing coordinates last."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def scalars(values: core.Scalar) -> np.ndarray:
    """One scalar per vertex or triangle, at the drawing boundary."""
    return np.asarray(values.batch().to_array())


class DeviceView:
    """The strip's moving artists, with fixed deformation and colour scales across frames."""

    def __init__(self, mesh: core.Mesh, vertices: core.Point, areas: core.Scalar, displacement_scale: float,
                 voltage_scale: float, strain_scale: float) -> None:
        self.mesh = mesh
        self.rest = coordinates(vertices)
        self.displacement_scale = displacement_scale
        self.voltage_norm = Normalize(-voltage_scale / 2, voltage_scale / 2)
        self.strain_norm = Normalize(-strain_scale, strain_scale)
        self.corner_weights = np.repeat(scalars(areas), 3)
        self.vertex_weights = np.bincount(mesh.faces.ravel(), weights=self.corner_weights)
        left, lower = self.rest.min(axis=0)
        right, upper = self.rest.max(axis=0)
        length, self.thickness = right - left, upper - lower
        middle = (upper + lower) / 2

        self.top = mesh.top[np.argsort(self.rest[mesh.top, 0])]
        self.bottom = mesh.bottom[np.argsort(self.rest[mesh.bottom, 0])]
        self.tip = mesh.tip[np.argsort(self.rest[mesh.tip, 1])]
        self.interface = mesh.interface[np.argsort(self.rest[mesh.interface, 0])]
        self.poles = np.linspace(0, len(self.top) - 1, 9, dtype=int)[1:-1]

        self.figure, ax = plt.subplots(figsize=(5.25, 1.4), dpi=100)
        self.figure.subplots_adjust(0.015, 0.025, 0.995, 0.975)
        half_height = max(3.3 * self.thickness, 0.11 * length)
        ax.set(xlim=(left - 0.045 * length, right + 0.025 * length),
               ylim=(middle - half_height, middle + half_height), aspect="equal")
        ax.set_axis_off()
        ax.add_patch(Rectangle((left - 0.025 * length, middle - self.thickness),
                               0.025 * length, 2 * self.thickness,
                               facecolor="#edf0f0", edgecolor="#a0aaaf", linewidth=0.8,
                               hatch="////", zorder=1))
        ax.plot([left, left], [middle - self.thickness, middle + self.thickness],
                color=interface_colour, linewidth=0.8, zorder=5)

        rest_outline = np.concatenate((self.bottom, self.tip, self.top[::-1]))
        ax.plot(*self.rest[rest_outline].T, color=rest_colour, linewidth=0.9,
                linestyle=(0, (4, 4)), zorder=0)
        self.triangulation = Triangulation(self.rest[:, 0], self.rest[:, 1], mesh.faces)
        self.body = ax.tripcolor(self.triangulation, np.zeros(len(self.rest)), shading="gouraud",
                                 cmap=strain_colours, norm=self.strain_norm, edgecolors="none", zorder=2)
        self.bond, = ax.plot([], [], color=interface_colour, linewidth=0.9, zorder=4)
        self.end, = ax.plot([], [], color=interface_colour, linewidth=1, zorder=4)
        self.electrodes = LineCollection([], linewidths=1.5, zorder=5, capstyle="round")
        ax.add_collection(self.electrodes)
        # The arrows belong to the material: the upper layer is poled up and the lower layer down.
        count = 2 * len(self.poles)
        self.poling = ax.quiver(np.zeros(count), np.zeros(count), np.zeros(count), np.zeros(count),
                                color=interface_colour, angles="xy", scale_units="xy", scale=1,
                                pivot="middle", width=0.0015, headwidth=3.1, headlength=4,
                                headaxislength=3.5, alpha=0.75, zorder=6)

    def update(self, displacement: core.Point, potential: core.Scalar,
               axial_strain: core.Scalar) -> plt.Figure:
        points = self.rest + self.displacement_scale * coordinates(displacement)
        # Area-weighted nodal readings give a continuous display of the piecewise-constant element strain.
        strain = np.bincount(self.mesh.faces.ravel(),
                             weights=self.corner_weights * np.repeat(scalars(axial_strain), 3))
        self.triangulation.x, self.triangulation.y = points.T
        self.body.set_array(strain / self.vertex_weights)
        self.bond.set_data(*points[self.interface].T)
        self.end.set_data(*points[self.tip].T)

        potential = scalars(potential)
        electrode_potential = np.array([potential[self.top].mean(), potential[self.bottom].mean()])
        # The two electrode colours read their voltage difference, independent of the potential's zero.
        electrode_potential -= electrode_potential.mean()
        self.electrodes.set_segments([points[self.top], points[self.bottom]])
        self.electrodes.set_colors(electrode_colours(self.voltage_norm(electrode_potential)))

        upper = points[self.top[self.poles]]
        lower = points[self.bottom[self.poles]]
        separation = upper - lower
        direction = separation / np.linalg.norm(separation, axis=-1, keepdims=True)
        centres = np.concatenate((0.75 * upper + 0.25 * lower, 0.25 * upper + 0.75 * lower))
        arrows = 0.32 * self.thickness * np.concatenate((direction, -direction))
        self.poling.set_offsets(centres)
        self.poling.set_UVC(*arrows.T)
        return self.figure


def draw_mesh(mesh: core.Mesh, vertices: core.Point) -> plt.Figure:
    """The clamped end of the strip's mesh: its triangles, the electrodes along its top and bottom, the bonded
    interface between its layers, and the clamp holding its first column of nodes."""
    rest = coordinates(vertices)
    left, lower = rest.min(axis=0)
    upper = rest.max(axis=0)[1]
    thickness = upper - lower
    shown = 3 * thickness
    figure, ax = plt.subplots(figsize=(3.0, 1.45), dpi=100)
    figure.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(left - 0.12 * thickness, left + shown), ylim=(lower - 0.12 * thickness, upper + 0.12 * thickness),
           aspect="equal")
    ax.set_axis_off()
    ax.add_patch(Rectangle((left - 0.12 * thickness, lower - 0.12 * thickness), 0.12 * thickness, 1.24 * thickness,
                           facecolor="#edf0f0", edgecolor="#a0aaaf", linewidth=0.8, hatch="////", zorder=1))
    ax.triplot(Triangulation(rest[:, 0], rest[:, 1], mesh.faces), color=rest_colour, linewidth=0.5, zorder=2)
    for nodes, colour, width in ((mesh.top, extension_colour, 2.5), (mesh.bottom, compression_colour, 2.5),
                                 (mesh.interface, interface_colour, 1.2)):
        ordered = nodes[np.argsort(rest[nodes, 0])]
        ax.plot(*rest[ordered].T, color=colour, linewidth=width, solid_capstyle="butt", zorder=3)
    ax.plot(*rest[mesh.clamped].T, "o", color=interface_colour, markersize=2.2, zorder=4)
    return figure


def draw_device(mesh: core.Mesh, vertices: core.Point, areas: core.Scalar, displacement: core.Point, potential: core.Scalar,
                axial_strain: core.Scalar, displacement_scale: float,
                voltage_scale: float, strain_scale: float) -> plt.Figure:
    """The clamped strip, with signed axial strain on its layers and signed voltage on its electrodes."""
    view = DeviceView(mesh, vertices, areas, displacement_scale, voltage_scale, strain_scale)
    return view.update(displacement, potential, axial_strain)


def animate(mesh: core.Mesh, vertices: core.Point, areas: core.Scalar, states: Iterator[tuple[core.Point, core.Scalar, core.Scalar]],
            displacement_scale: float, voltage_scale: float, strain_scale: float) -> Iterator[np.ndarray]:
    """Successive equilibria, with the camera and all three scales held fixed."""
    view = DeviceView(mesh, vertices, areas, displacement_scale, voltage_scale, strain_scale)
    try:
        for state in states:
            yield capture(view.update(*state))
    finally:
        plt.close(view.figure)
