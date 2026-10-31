"""The two bonded piezoelectric layers, their electrodes, and the strain of the bent strip."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation

from numga import concatenate, stack
from examples.electromagnetism.piezoelectric import core
from examples.util.animation import capture
from examples.util.wgsl import contours, pinned, solids
from examples.util.wgsl.flow import axes_view


compression_colour = "#286580"
extension_colour = "#df6435"
interface_colour = "#53636b"
rest_colour = "#c5cdd1"
strain_colours = LinearSegmentedColormap.from_list(
    "piezoelectric_strain", (compression_colour, "#f5f4ef", extension_colour))
electrode_colours = LinearSegmentedColormap.from_list(
    "piezoelectric_voltage", (compression_colour, "#8b969b", extension_colour))
# The observer's slice of space, which carries the strip's events into the renderer's space.
space_slice = core.mv.t ^ (core.Point & core.time)                              # [] Point- <- Point-


def coordinates(points: core.Point) -> np.ndarray:
    """Positions read by the slices across x and y, with the drawing coordinates last."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def scalars(values: core.Scalar) -> np.ndarray:
    """One scalar per vertex or triangle, at the drawing boundary."""
    return np.asarray(values.batch().to_array())


class DeviceView:
    """The strip's moving artists, with fixed deformation and colour scales across frames."""

    def __init__(self, mesh: core.Strip, vertices: core.Point, areas: core.Scalar, displacement_scale: float,
                 voltage_scale: float, strain_scale: float) -> None:
        self.mesh, self.vertices = mesh, vertices
        self.rest = coordinates(vertices)
        self.displacement_scale = displacement_scale
        self.voltage_norm = Normalize(-voltage_scale / 2, voltage_scale / 2)
        self.strain_norm = Normalize(-strain_scale, strain_scale)
        # Each face's area shared among its corners, for area-weighted readings of face values at the vertices.
        self.corner_shares = ~mesh.incidence[2, 0].mean(axis=0)                 # Scalar+[vertices <- faces]
        self.areas, self.vertex_areas = areas, self.corner_shares * areas
        left, lower = self.rest.min(axis=0)
        right, upper = self.rest.max(axis=0)
        length, self.thickness = right - left, upper - lower
        middle = (upper + lower) / 2

        # The strip's electrodes and interface run in order along it, and its tip from the bottom up.
        self.poles = np.linspace(0, len(mesh.top) - 1, 9, dtype=int)[1:-1]

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

        rest_outline = np.concatenate((mesh.bottom, mesh.tip, mesh.top[::-1]))
        ax.plot(*self.rest[rest_outline].T, color=rest_colour, linewidth=0.9,
                linestyle=(0, (4, 4)), zorder=0)
        # The strained body, a picture of the deformed mesh shaded anew at each state.
        self.view = axes_view(ax)
        self.body = pinned(ax, np.zeros(self.view.shape + (4,)), (*ax.get_xlim(), *ax.get_ylim()))
        self.body.set_zorder(2)
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
        deformed = self.vertices + self.displacement_scale * displacement           # Point-[vertices]
        points = coordinates(deformed)
        # Area-weighted nodal readings give a continuous display of the piecewise-constant element strain.
        # The strain comes per triangle, a batch over the faces; as a field over them it reaches the vertices.
        strain = scalars(self.corner_shares * (self.areas * axial_strain.field()) / self.vertex_areas)
        places = solids.origin + solids.context.include_from_algebra(space_slice(deformed.batch()))   # [vertices] Point-
        self.body.set_data(contours.picture(places, self.mesh, strain, strain_colours(self.strain_norm(strain))[:, :3],
                                            self.view))
        self.bond.set_data(*points[self.mesh.interface].T)
        self.end.set_data(*points[self.mesh.tip].T)

        potential = scalars(potential)
        electrode_potential = np.array([potential[self.mesh.top].mean(), potential[self.mesh.bottom].mean()])
        # The two electrode colours read their voltage difference, independent of the potential's zero.
        electrode_potential -= electrode_potential.mean()
        self.electrodes.set_segments([points[self.mesh.top], points[self.mesh.bottom]])
        self.electrodes.set_colors(electrode_colours(self.voltage_norm(electrode_potential)))

        # The poling arrows, across the strip from electrode to electrode: up in the upper layer, a quarter of the way
        # down from the top, and down in the lower layer, a quarter of the way up from the bottom.
        upper = deformed.batch()[self.mesh.top[self.poles]]                         # [poles] Point-
        lower = deformed.batch()[self.mesh.bottom[self.poles]]                      # [poles] Point-
        separation = upper - lower                                                  # [poles] Point-
        across = separation / separation.squared().square_root()                    # [poles] Point-
        centres = concatenate((0.75 * upper + 0.25 * lower, 0.25 * upper + 0.75 * lower))   # [arrows] Point-
        arrows = 0.32 * self.thickness * concatenate((across, -across))             # [arrows] Point-
        self.poling.set_offsets(coordinates(centres))
        self.poling.set_UVC(*coordinates(arrows).T)
        return self.figure


def draw_mesh(mesh: core.Strip, vertices: core.Point) -> plt.Figure:
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
    # The electrodes and the interface run in order along the strip.
    for nodes, colour, width in ((mesh.top, extension_colour, 2.5), (mesh.bottom, compression_colour, 2.5),
                                 (mesh.interface, interface_colour, 1.2)):
        ax.plot(*rest[nodes].T, color=colour, linewidth=width, solid_capstyle="butt", zorder=3)
    ax.plot(*rest[mesh.clamped].T, "o", color=interface_colour, markersize=2.2, zorder=4)
    return figure


def draw_device(mesh: core.Strip, vertices: core.Point, areas: core.Scalar, displacement: core.Point, potential: core.Scalar,
                axial_strain: core.Scalar, displacement_scale: float,
                voltage_scale: float, strain_scale: float) -> plt.Figure:
    """The clamped strip, with signed axial strain on its layers and signed voltage on its electrodes."""
    view = DeviceView(mesh, vertices, areas, displacement_scale, voltage_scale, strain_scale)
    return view.update(displacement, potential, axial_strain)


def animate(mesh: core.Strip, vertices: core.Point, areas: core.Scalar, states: Iterator[tuple[core.Point, core.Scalar, core.Scalar]],
            displacement_scale: float, voltage_scale: float, strain_scale: float) -> Iterator[np.ndarray]:
    """Successive equilibria, with the camera and all three scales held fixed."""
    view = DeviceView(mesh, vertices, areas, displacement_scale, voltage_scale, strain_scale)
    try:
        for state in states:
            yield capture(view.update(*state))
    finally:
        plt.close(view.figure)
