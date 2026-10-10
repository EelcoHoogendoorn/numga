"""Current paths in a conducting disc beneath a local magnetic field."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.tri import LinearTriInterpolator, Triangulation, TriContourSet

from examples.animation import capture
from examples.electromagnetism.eddy_brake import core
from examples.mesh import Mesh

if TYPE_CHECKING:
    from IPython.display import Image as Shown

IMAGE_SAMPLES = 256
ANIMATION_DPI = 120
FIBRE_REACH = 0.94
FIBRE_COLOUR = (0.16, 0.43, 0.56, 0.8)
FIBRE_WIDTH = 1.35
CURRENT_GAMMA = 0.35
STREAMLINE_GAMMA = 0.25
CURRENT_LEVELS = 10
CONTOUR_PASSES = 2
ARROW_PHASE = (np.sqrt(5) - 1) / 2
ARROW_LENGTH = 0.035
HEAT_GAMMA = 0.3
CURRENT_COLOURS = LinearSegmentedColormap.from_list("current", ("#f8f5ed", "#edb15f", "#c95832"))
HEAT_COLOURS = LinearSegmentedColormap.from_list("heat", ("#fbf9f4", "#e58b3a", "#8f2d1b"))


# --- plumbing -------------------------------------------------------------------------
def parallel_fibres(radius: float, curves: int, samples: int) -> core.Vector:
    """Parallel chords marking the direction of straight body-fixed fibres."""
    offsets = np.linspace(-FIBRE_REACH, FIBRE_REACH, curves)
    fractions = np.linspace(-1, 1, samples)
    half_lengths = radius * np.sqrt(1 - offsets ** 2)
    return (core.mv.x * half_lengths[:, None] * fractions[None, :]
            + core.mv.y * radius * offsets[:, None])


def radial_fibres(inner_radius: float, outer_radius: float, curves: int, samples: int) -> core.Vector:
    """Radial fibre guides between the isotropic centre and rim bands."""
    rays = 2 * curves
    angles = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    distances = np.linspace(inner_radius, outer_radius, samples)
    directions = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x           # [rays] Vector
    return directions[:, None] * distances[None, :]


def circular_fibres(inner_radius: float, outer_radius: float, curves: int, samples: int) -> core.Vector:
    """Closed fibre rings between the isotropic centre and rim bands."""
    radii = np.linspace(inner_radius, outer_radius, curves)
    angles = np.linspace(0, 2 * np.pi, samples)
    directions = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x           # [samples] Vector
    return directions[None, :] * radii[:, None]


def rotation_marks(orientations: core.Rotor, radius: float) -> np.ndarray:
    """Material rim positions, `[frames, cases, 2]`."""
    return (orientations >> (core.mv.x * (0.94 * radius))).cast(core.ga.subspace("x y")).kernel


def rotation_marker(ax: plt.Axes) -> PathCollection:
    """A fluorescent rim knob showing the material's rotation."""
    return ax.scatter([], [], s=180, c="#ff00c8", edgecolors="white", linewidths=1.5,
                      zorder=6, clip_on=False)


def vertex_currents(mesh: Mesh, currents: core.Vector) -> np.ndarray:
    """Area-weighted current at vertices for drawing a batch of face fields `[cases] Vector[F]`."""
    current = currents.cast(core.ga.subspace("x y")).kernel
    areas = mesh.triangle_areas.cast(core.ga.subspace.scalar()).kernel[:, 0]
    vertex_count = len(mesh.vertices.batch())
    cases = np.arange(len(current))
    vertex_current = np.zeros((len(current), vertex_count, 2))
    np.add.at(vertex_current, (cases[:, None, None], mesh.faces[None, :, :]),
              (areas[None, :, None] * current)[:, :, None, :])
    vertex_areas = np.zeros(vertex_count)
    np.add.at(vertex_areas, mesh.faces, areas[:, None])
    return vertex_current / vertex_areas[None, :, None]


def vertex_scalars(mesh: Mesh, values: np.ndarray) -> np.ndarray:
    """Area-weighted vertex colours from scalar face fields `[frames, cases, F]`."""
    areas = mesh.triangle_areas.cast(core.ga.subspace.scalar()).kernel[:, 0]
    vertex_count = len(mesh.vertices.batch())
    face_values = values.reshape(-1, len(mesh.faces))
    batches = np.arange(len(face_values))
    vertex_values = np.zeros((len(face_values), vertex_count))
    np.add.at(vertex_values, (batches[:, None, None], mesh.faces[None, :, :]),
              (face_values * areas)[:, :, None])
    vertex_areas = np.zeros(vertex_count)
    np.add.at(vertex_areas, mesh.faces, areas[:, None])
    return (vertex_values / vertex_areas).reshape(*values.shape[:-1], vertex_count)


def smooth_contour(curve: np.ndarray, passes: int) -> np.ndarray:
    """Chaikin corner cutting on a closed `[points, 2]` display curve."""
    points = curve[:-1]
    for _ in range(passes):
        following = np.roll(points, -1, axis=0)
        points = np.stack((0.75 * points + 0.25 * following,
                           0.25 * points + 0.75 * following), axis=1).reshape(-1, 2)
    return np.concatenate((points, points[:1]))


def current_contours(ax: plt.Axes, triangles: Triangulation, streamfunction: np.ndarray,
                     current: np.ndarray, levels: np.ndarray, arrow_length: float,
                     colour: tuple[float, float, float, float]) -> TriContourSet:
    """Closed current paths at prescribed levels, with geometrically anchored direction arrows."""
    contours = ax.tricontour(triangles, streamfunction, levels=levels, colors=[colour],
                             linewidths=0.8, linestyles="solid", zorder=3)
    locate = triangles.get_trifinder()
    paths = []
    for level, pieces in enumerate(contours.allsegs):
        display_paths = []
        for raw in (curve for curve in pieces if len(curve) > 2):
            # Read the solved current's direction on the unsmoothed contour's longest segment.
            segments = raw[1:] - raw[:-1]
            longest = np.linalg.norm(segments, axis=-1).argmax()
            midpoint = (raw[longest] + raw[longest + 1]) / 2
            direction = np.sign(segments[longest] @ current[locate(*midpoint)])
            curve = smooth_contour(raw, CONTOUR_PASSES)
            display_paths.append(Path(curve, closed=True))
            starts, ends = curve[:-1], curve[1:]
            lengths = np.linalg.norm(ends - starts, axis=-1)
            distance = np.concatenate(([0], np.cumsum(lengths)))
            perimeter = distance[-1]
            # Anchor at a horizontal crossing, independent of the contour tracer's start vertex.
            height = (curve[:, 1].min() + curve[:, 1].max()) / 2
            crossings = np.flatnonzero((starts[:, 1] > height) != (ends[:, 1] > height))
            fractions = (height - starts[crossings, 1]) / (ends[crossings, 1] - starts[crossings, 1])
            horizontal = starts[crossings, 0] + fractions * (ends[crossings, 0] - starts[crossings, 0])
            rightmost = horizontal.argmax()
            anchor = distance[crossings[rightmost]] + fractions[rightmost] * lengths[crossings[rightmost]]
            half_length = min(arrow_length, perimeter / 8) / 2
            # Stagger arrows between levels, retaining their phase as each path evolves.
            samples = (anchor + direction * ((level * ARROW_PHASE % 1) * perimeter
                       + np.array([-half_length, half_length]))) % perimeter
            points = np.stack((np.interp(samples, distance, curve[:, 0]),
                               np.interp(samples, distance, curve[:, 1])), axis=-1)
            # Follow the smoothed curve itself, so arrows stay on the displayed current path.
            offsets = direction * (distance[:-1] - samples[0]) % perimeter
            inside = np.flatnonzero((offsets > 0) & (offsets < 2 * half_length))
            inside = inside[np.argsort(offsets[inside])]
            points = np.concatenate((points[:1], curve[inside], points[1:]))
            ax.add_patch(FancyArrowPatch(path=Path(points), arrowstyle="-|>", mutation_scale=7,
                                         shrinkA=0, shrinkB=0, linewidth=0.8, color=colour, zorder=3))
        paths.append(Path.make_compound_path(*display_paths))
    contours.set_paths(paths)
    return contours


def draw(mesh: Mesh, response: core.Response, centre: core.Vector,
         width: float, labels: tuple[str, ...]) -> plt.Figure:
    """The materials' solved currents, with magnitude on one colour scale and field width dashed."""
    positions = mesh.vertices.cast(core.ga.subspace("x y")).kernel
    field_centre = centre.cast(core.ga.subspace("x y")).kernel
    triangles = Triangulation(positions[:, 0], positions[:, 1], mesh.faces)

    # Average the face currents at shared vertices solely to draw a continuous field.
    # The streamlines follow this interpolated solution; they are not prescribed paths.
    vertex_current = vertex_currents(mesh, response.current)
    extent = np.abs(positions).max()
    sample_axis = np.linspace(-extent, extent, IMAGE_SAMPLES)
    horizontal, vertical = np.meshgrid(sample_axis, sample_axis)
    magnitude = np.linalg.norm(vertex_current, axis=-1)
    current_scale = PowerNorm(CURRENT_GAMMA, vmin=0, vmax=magnitude.max())
    boundary = mesh.edges[mesh.boundary_edges]

    figure, panels = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4.2),
                                  squeeze=False, layout="constrained")
    for ax, label, vectors, speed in zip(panels[0], labels, vertex_current, magnitude):
        along = LinearTriInterpolator(triangles, vectors[:, 0])(horizontal, vertical)
        across = LinearTriInterpolator(triangles, vectors[:, 1])(horizontal, vertical)
        ax.tripcolor(triangles, speed, shading="gouraud", cmap=CURRENT_COLOURS,
                     norm=current_scale, rasterized=True)
        ax.streamplot(sample_axis, sample_axis, along, across, color="#493c32",
                       density=1.35, linewidth=0.8, arrowsize=0.8)
        ax.add_collection(LineCollection(positions[boundary], colors="#746b60", linewidths=0.8))
        ax.add_patch(Circle(field_centre, width, facecolor="none", edgecolor="#334155",
                            linewidth=1, linestyle=(0, (4, 4))))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_title(label, fontsize=11)
        ax.set_axis_off()
    return figure


def animate(mesh: Mesh, frames: Iterable[core.Motion], centre: core.Vector,
            width: float, labels: tuple[str, ...], fibres: tuple[core.Vector, ...]) -> list[np.ndarray]:
    """Batched rotating materials and currents, with `[curves, samples]` body-fixed guides per panel."""
    from numga import stack

    states = list(frames)
    positions = mesh.vertices.cast(core.ga.subspace("x y")).kernel
    field_centre = centre.cast(core.ga.subspace("x y")).kernel
    triangles = Triangulation(positions[:, 0], positions[:, 1], mesh.faces)
    face_current = stack([state.response.current for state in states], axis=0)  # [frames, cases] Vector[F]
    current = vertex_currents(mesh, face_current.reshape(-1))
    current = current.reshape(len(states), len(labels), len(positions), 2)
    magnitude = np.linalg.norm(current, axis=-1)
    peak = magnitude.max()
    current_scale = PowerNorm(CURRENT_GAMMA, vmin=0, vmax=peak)
    extent = np.abs(positions).max()
    sample_axis = np.linspace(-extent, extent, IMAGE_SAMPLES)
    horizontal, vertical = np.meshgrid(sample_axis, sample_axis)

    # Fibre curves and one rim mark are fixed to each body. The magnetic patches stay still.
    marks = rotation_marks(stack([state.orientation for state in states]), extent)

    figure = plt.figure(figsize=(3.3 * len(labels), 3.6), dpi=ANIMATION_DPI)
    boundary = mesh.edges[mesh.boundary_edges]
    panels = []
    for i, label in enumerate(labels):
        ax = figure.add_axes(((i + 0.035) / len(labels), 0.03, 0.93 / len(labels), 0.86))
        colours = ax.tripcolor(triangles, magnitude[0, i], shading="gouraud", cmap=CURRENT_COLOURS,
                               norm=current_scale, rasterized=True)
        threads = LineCollection([], colors=FIBRE_COLOUR, linewidths=FIBRE_WIDTH, zorder=2)
        ax.add_collection(threads)
        marker = rotation_marker(ax)
        ax.add_collection(LineCollection(positions[boundary], colors="#746b60", linewidths=0.8))
        ax.add_patch(Circle(field_centre, width, facecolor="none", edgecolor="#334155",
                            linewidth=1, linestyle=(0, (4, 4)), zorder=4))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
        panels.append((ax, colours, threads, marker, len(ax.patches)))

    images = []
    for state, vectors, speed, frame_marks in zip(states, current, magnitude, marks):
        paths = []
        for i, (ax, colours, threads, marker, fixed_patches) in enumerate(panels):
            colours.set_array(speed[i])
            threads.set_segments((state.orientation[i] >> fibres[i]).cast(core.ga.subspace("x y")).kernel)
            marker.set_offsets(frame_marks[i:i + 1])
            along = LinearTriInterpolator(triangles, vectors[i, :, 0])(horizontal, vertical)
            across = LinearTriInterpolator(triangles, vectors[i, :, 1])(horizontal, vertical)
            # A shared power scale keeps weak currents visible while still fading to zero.
            paths.append(ax.streamplot(sample_axis, sample_axis, along, across,
                                         color=(0.29, 0.24, 0.20, (speed[i].max() / peak) ** STREAMLINE_GAMMA),
                                         density=1.2, linewidth=0.8, arrowsize=0.8, zorder=3))
        images.append(capture(figure))
        for path, (ax, colours, threads, marker, fixed_patches) in zip(paths, panels):
            path.lines.remove()
            for arrow in tuple(ax.patches)[fixed_patches:]:
                arrow.remove()
    plt.close(figure)
    return images


def animate_heat(mesh: Mesh, frames: Iterable[core.ThermalMotion], centre: core.Vector,
                 width: float, labels: tuple[str, ...], fibres: tuple[core.Vector, ...]) -> list[np.ndarray]:
    """Accumulated heat per area, carried by the moving material faces on one common scale."""
    from numga import stack

    states = list(frames)
    # Face heat is energy. Dividing by each material face's area gives the density to colour.
    density = stack([state.heat for state in states], axis=0) / mesh.triangle_areas  # [frames, cases] Scalar[F]
    values = density.cast(core.ga.subspace.scalar()).kernel[..., 0]
    heat_scale = PowerNorm(HEAT_GAMMA, vmin=0, vmax=values.max())
    values = vertex_scalars(mesh, values)
    reference = mesh.vertices.cast(core.ga.subspace("x y")).kernel
    extent = np.linalg.norm(reference, axis=-1).max()
    field_centre = centre.cast(core.ga.subspace("x y")).kernel
    boundary = mesh.edges[mesh.boundary_edges]
    marks = rotation_marks(stack([state.orientation for state in states]), extent)

    figure = plt.figure(figsize=(3.3 * len(labels), 3.6), dpi=ANIMATION_DPI)
    panels = []
    for i, label in enumerate(labels):
        ax = figure.add_axes(((i + 0.035) / len(labels), 0.03, 0.93 / len(labels), 0.86))
        threads = LineCollection([], colors=FIBRE_COLOUR, linewidths=FIBRE_WIDTH, zorder=2)
        ax.add_collection(threads)
        marker = rotation_marker(ax)
        outline = LineCollection([], colors="#746b60", linewidths=0.8, zorder=3)
        ax.add_collection(outline)
        ax.add_patch(Circle(field_centre, width, facecolor="none", edgecolor="#334155",
                            linewidth=1, linestyle=(0, (4, 4)), zorder=4))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
        panels.append((ax, threads, marker, outline))

    images = []
    for state, densities, frame_marks in zip(states, values, marks):
        positions = state.vertices.cast(core.ga.subspace("x y")).kernel
        surfaces = []
        for i, (ax, threads, marker, outline) in enumerate(panels):
            # The display mesh moves with the material; vertex colours interpolate face heat.
            triangles = Triangulation(positions[i, :, 0], positions[i, :, 1], mesh.faces)
            surfaces.append(ax.tripcolor(triangles, densities[i], shading="gouraud",
                                          cmap=HEAT_COLOURS, norm=heat_scale, rasterized=True, zorder=1))
            outline.set_segments(positions[i, boundary])
            threads.set_segments((state.orientation[i] >> fibres[i]).cast(core.ga.subspace("x y")).kernel)
            marker.set_offsets(frame_marks[i:i + 1])
        images.append(capture(figure))
        for surface in surfaces:
            surface.remove()
    plt.close(figure)
    return images


def animate_inductive(mesh: Mesh, frames: Iterable[core.InductiveMotion], centre: core.Vector,
                      width: float, labels: tuple[str, ...], fibres: tuple[core.Vector, ...]) -> list[np.ndarray]:
    """Current build-up and decay on a moving material mesh, on one scale across cases and time."""
    from numga import stack

    states = list(frames)
    reference = mesh.vertices.cast(core.ga.subspace("x y")).kernel
    # The current belongs to material faces; turn its direction into the laboratory frame.
    field = stack([state.orientation >> state.current for state in states], axis=0)  # [frames, cases] Vector[F]
    magnitude = field.norm().cast(core.ga.subspace.scalar()).kernel[..., 0]
    current = field.cast(core.ga.subspace("x y")).kernel
    streamfunction = stack([state.streamfunction for state in states], axis=0).cast(core.ga.subspace.scalar()).kernel[..., 0]
    # Each panel keeps its contour values throughout the animation, including current decay.
    # Separate ranges keep the weak material cases legible; colours share one magnitude scale.
    fractions = (np.arange(-CURRENT_LEVELS, CURRENT_LEVELS) + 0.5) / CURRENT_LEVELS
    levels = np.abs(streamfunction).max(axis=(0, 2))[:, None] * fractions
    peak = magnitude.max()
    opacity = np.divide(magnitude.max(axis=-1), peak, out=np.zeros(magnitude.shape[:2]),
                         where=peak > 0) ** STREAMLINE_GAMMA
    current_scale = PowerNorm(CURRENT_GAMMA, vmin=0, vmax=peak)
    vertex_magnitude = vertex_scalars(mesh, magnitude)
    extent = np.linalg.norm(reference, axis=-1).max()
    field_centre = centre.cast(core.ga.subspace("x y")).kernel
    boundary = mesh.edges[mesh.boundary_edges]
    marks = rotation_marks(stack([state.orientation for state in states]), extent)

    figure = plt.figure(figsize=(3.3 * len(labels), 3.6), dpi=ANIMATION_DPI)
    panels = []
    for i, label in enumerate(labels):
        ax = figure.add_axes(((i + 0.035) / len(labels), 0.03, 0.93 / len(labels), 0.86))
        threads = LineCollection([], colors=FIBRE_COLOUR, linewidths=FIBRE_WIDTH, zorder=2)
        ax.add_collection(threads)
        marker = rotation_marker(ax)
        outline = LineCollection([], colors="#746b60", linewidths=0.8, zorder=3)
        ax.add_collection(outline)
        ax.add_patch(Circle(field_centre, width, facecolor="none", edgecolor="#334155",
                            linewidth=1, linestyle=(0, (4, 4)), zorder=4))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
        panels.append((ax, threads, marker, outline, len(ax.patches)))

    images = []
    for state, vectors, speed, stream, alpha, frame_marks in zip(
            states, current, vertex_magnitude, streamfunction, opacity, marks):
        positions = state.vertices.cast(core.ga.subspace("x y")).kernel
        paths = []
        surfaces = []
        for i, (ax, threads, marker, outline, fixed_patches) in enumerate(panels):
            outline.set_segments(positions[i, boundary])
            threads.set_segments((state.orientation[i] >> fibres[i]).cast(core.ga.subspace("x y")).kernel)
            marker.set_offsets(frame_marks[i:i + 1])
            triangles = Triangulation(positions[i, :, 0], positions[i, :, 1], mesh.faces)
            surfaces.append(ax.tripcolor(triangles, speed[i], shading="gouraud",
                                          cmap=CURRENT_COLOURS, norm=current_scale, rasterized=True, zorder=1))
            colour = (0.29, 0.24, 0.20, alpha[i])
            paths.append(current_contours(ax, triangles, stream[i], vectors[i], levels[i, levels[i] != 0],
                                           ARROW_LENGTH * extent, colour))
        images.append(capture(figure))
        for surface in surfaces:
            surface.remove()
        for path, (ax, threads, marker, outline, fixed_patches) in zip(paths, panels):
            path.remove()
            for arrow in tuple(ax.patches)[fixed_patches:]:
                arrow.remove()
    plt.close(figure)
    return images


def inline(frames: list[np.ndarray], duration_ms: int) -> Shown:
    """Frames as a looping GIF to show in a notebook, kept in memory."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image

    images = [Image.fromarray(pixels) for pixels in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")


def readout(response: core.Response, labels: tuple[str, ...]) -> str:
    """Material, signed torque about the disc axis, and total dissipated power."""
    torque = response.torque.cast(core.ga.subspace("xy")).kernel[..., 0] * 1e3
    heating = response.heating.batch().sum(axis=-1).cast(core.ga.subspace.scalar()).kernel[..., 0] * 1e3
    label_width = max(len("material"), *(len(label) for label in labels))
    header = f"{'material':<{label_width}}  {'torque (mN m)':>14}  {'heating (mW)':>14}"
    rows = (f"{label:<{label_width}}  {moment:14.2f}  {power:14.2f}"
            for label, moment, power in zip(labels, torque, heating))
    return "\n".join((header, *rows))
