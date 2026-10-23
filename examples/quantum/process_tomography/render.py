"""Probe probabilities, reconstructed channels and predictions drawn from quantum states. The Bloch balls
are drawn on the GPU, the rest by matplotlib."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D

from numga import concatenate
from examples.util.animation import capture
from examples.quantum.process_tomography import core
from examples.util.wgsl import labels, meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene

sm = solids.mv
CHANNEL_COLOURS = ("#3275a8", "#d16a30", "#7b62a3")
PROBE_COLOURS = ("#3275a8", "#d16a30", "#438e65", "#7b62a3")
PROBE_NAMES = ("A", "B", "C", "D")
# The sphere of pure inputs, faint, and its grid; the edges joining the probes, and the axes.
SPHERE_COLOUR, SPHERE_OPACITY = np.full(3, 0.85), 0.06
GRID_COLOUR, GRID_OPACITY = np.full(3, 0.6), 0.6
EDGE_COLOUR, AXIS_COLOUR = np.full(3, 0.75), np.full(3, 0.55)
# The opacities of a channel's image of the sphere and of the probes as prepared.
IMAGE_OPACITY, PREPARED_OPACITY = 0.2, 0.2
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the square view
# about the centre, and the pixels across it.
ELEVATION, AZIMUTH = np.radians(24.0), np.radians(-54.0 + 90.0)
REACH, N_PIXELS = 1.45, 700
# The grid lines drawn, every few of the sampled latitudes and longitudes; the widths of the sphere's grid lines and
# an image's, the edges and the axes' shafts, the arrowheads' length and width, and the radii of the probes alone,
# after a channel and as prepared, in pixels; how far the axes reach and where they are named.
LATITUDE_STEP, LONGITUDE_STEP = 2, 3
GRID_WIDTH, IMAGE_WIDTH, EDGE_WIDTH, AXIS_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 2.0, 3.4, 4.0, 2.4, 18.0, 10.0
PROBE_RADIUS, AFTER_RADIUS, PREPARED_RADIUS = 12.0, 7.0, 11.0
AXIS_LENGTH, LABEL_DISTANCE = 1.3, 1.4


# --- plumbing -------------------------------------------------------------------------
def bloch(states: core.State) -> np.ndarray:
    """The Bloch vectors of normalized states, in an explicit spatial blade layout."""
    return 2 * states.cast(core.ga.subspace("x y z")).kernel                  # [..., 3]


def bloch_view() -> solids.View:
    """One view of the Bloch ball shared by all pictures."""
    return solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, REACH), N_PIXELS)


def bloch_axes(pixel: float) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The frame of the x, y and z axes."""
    shafts, heads = solids.frame(AXIS_LENGTH, AXIS_WIDTH / 2 * pixel, HEAD_LENGTH * pixel, HEAD_WIDTH / 2 * pixel)
    return [(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR)]


def show_ball(axes, image: np.ndarray, view: solids.View, title: str):
    """A traced picture in an axes, with the axes named past their tips. The image, for an animation to set its
    data."""
    shown = show(axes, image, view)
    labels(axes, view, sm.point(LABEL_DISTANCE * np.eye(3)), list("xyz"), ha="center", va="center", fontsize=11,
           color=AXIS_COLOUR)
    axes.set_title(title)
    return shown


def surface_faces(rows: int, columns: int) -> np.ndarray:
    """The triangles of a grid of points closed around its columns, from pole to pole, indices into the grid
    flattened: two to a cell, but one where a cell's row collapses into a pole."""
    index = np.arange(rows * columns).reshape(rows, columns)
    index = np.concatenate([index, index[:, :1]], axis=1)                      # [rows, columns + 1]
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], axis=-1)[:-1].reshape(-1, 3),
                           np.stack([a, c, d], axis=-1)[1:].reshape(-1, 3)])


def grid_lines(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every few latitudes, but the poles, `[lines, longitudes + 1, 3]`, and every few longitudes, `[lines,
    latitudes + 1, 3]`, of a grid of points `[latitudes + 1, longitudes + 1, 3]`."""
    return points[LATITUDE_STEP:-1:LATITUDE_STEP], np.moveaxis(points[:, :-1:LONGITUDE_STEP], 1, 0)


def grid(points: np.ndarray, colour: np.ndarray, width: float, view: solids.View) -> meshes.Samples:
    """The grid lines of a grid of points `[latitudes + 1, longitudes + 1, 3]`, as strokes `width` pixels across in
    a colour, at every sample."""
    lines = [line for batch in grid_lines(points) for line in batch]
    return strokes.polylines([(line, colour) for line in lines], width, view)


def grid_planes(points: np.ndarray) -> solids.Plane:
    """The planes of the grid lines of a grid of points on a sphere `[latitudes + 1, longitudes + 1, 3]`, each
    through its line's first point and those a third and two thirds along it."""
    planes = []
    for lines in grid_lines(points):
        count = lines.shape[1]
        first, second, third = (lines[:, index] for index in (0, count // 3, 2 * count // 3))   # [lines, 3] each
        normals = np.cross(second - first, third - first)
        planes.append(solids.plane(normals, np.sum(normals * first, axis=-1)))
    return concatenate(planes)


def draw_experiment(prepared: core.State, probabilities: core.Scalar,
                    labels: tuple[str, ...]) -> plt.Figure:
    """Four tetrahedral preparations and each channel's measured outcome probabilities."""
    probes = bloch(prepared)                                                   # [preparations, 3]
    measured = probabilities.to_array()                                      # [channels, preparations, outcomes]
    figure = plt.figure(figsize=(13, 3.7), layout="constrained")
    figure.get_layout_engine().set(wspace=0.12)
    probe_axes = figure.add_subplot(1, len(labels) + 1, 1)
    view = bloch_view()
    colours = np.array([to_rgb(colour) for colour in PROBE_COLOURS])
    # Joining the four probes makes their coverage of space visible.
    first, second = np.triu_indices(len(probes), k=1)
    edges = solids.rods(sm.point(probes[first]), sm.point(probes[second]), EDGE_WIDTH / 2 * view.pixel)
    scene = solids.draw([*bloch_axes(view.pixel), (edges, EDGE_COLOUR),
                         (solids.ball(sm.point(probes), PROBE_RADIUS * view.pixel)[:, None], colours)], view, np.ones(3))
    show_ball(probe_axes, scene, view, "Prepared states")
    for (x, y), name, colour in zip(view.across(sm.point(probes * 1.12)), PROBE_NAMES, PROBE_COLOURS):
        probe_axes.text(x, y, name, color=colour, fontsize=10)

    probability_scale = LogNorm(vmin=measured[measured > 0].min(), vmax=0.5)
    table_axes = []
    for index, (label, table) in enumerate(zip(labels, measured)):
        axes = figure.add_subplot(1, len(labels) + 1, index + 2)
        table_axes.append(axes)
        heatmap = axes.imshow(table, origin="upper", norm=probability_scale, cmap="turbo")
        axes.set(xticks=np.arange(len(PROBE_NAMES)), xticklabels=PROBE_NAMES,
                 yticks=np.arange(len(PROBE_NAMES)), yticklabels=PROBE_NAMES,
                 xlabel="Measurement outcome", ylabel="Prepared state", title=label)
        axes.tick_params(length=0)
    figure.colorbar(heatmap, ax=table_axes, label="Probability (log scale)", shrink=0.72, pad=0.02)
    return figure


def channel_scene(sphere: np.ndarray, probes: np.ndarray, view: solids.View) -> Scene:
    """What every picture of a channel shares, from the sphere of pure inputs `[latitudes + 1, longitudes + 1, 3]`
    and the probes as prepared `[preparations, 3]`: the axes; the sphere and the probes, translucent; and the sphere's
    grid inked where the planes of its lines cut it, the sphere the first part after the axes and the probes after
    the channel, which each picture adds."""
    colours = np.array([to_rgb(colour) for colour in PROBE_COLOURS])
    translucent = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR, SPHERE_OPACITY),
                   (solids.ball(sm.point(probes), PREPARED_RADIUS * view.pixel)[:, None], colours, PREPARED_OPACITY)]
    planes = grid_planes(sphere)                                               # [lines] Plane
    axes = bloch_axes(view.pixel)
    reference = solids.Ink(len(axes) + 1, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                           GRID_COLOUR, GRID_WIDTH, GRID_OPACITY)
    return Scene(view, np.ones(3), axes, [], inks=[reference], translucent=translucent)


def channel_picture(scene: Scene, faces: np.ndarray, image: np.ndarray, points: np.ndarray, colour: str) -> np.ndarray:
    """A channel's picture among the scene's shared parts: its image of the sphere `[latitudes + 1, longitudes + 1,
    3]`, a translucent surface over the same grid as the sphere, with its grid lines, and the probes after the
    channel `[preparations, 3]`, solid."""
    view = scene.view
    colours = np.array([to_rgb(colour) for colour in PROBE_COLOURS])
    # The reference grid and the channel's image use the same sampled directions.
    places = sm.point(image[:, :-1].reshape(-1, 3))                          # [(latitudes + 1) * longitudes] Point
    shell = meshes.samples(places, faces, np.array(to_rgb(colour)), meshes.smooth(places, faces), view, gloss=0.1)
    lines = grid(image, np.array(to_rgb(colour)), IMAGE_WIDTH, view)
    after = solids.ball(sm.point(points), AFTER_RADIUS * view.pixel)[:, None]
    return scene.draw([(after, colours)], meshes=[(shell, IMAGE_OPACITY), (lines, 1.0)])


def channels_figure(pictures: list[np.ndarray], view: solids.View, labels: tuple[str, ...]) -> tuple[plt.Figure, list]:
    """The channels' pictures side by side, each named, over the legend of the probes: the figure, and each
    picture's image, for an animation to set its data."""
    figure = plt.figure(figsize=(4.1 * len(labels), 4.3), layout="constrained")
    shown = [show_ball(figure.add_subplot(1, len(labels), index + 1), picture, view, label)
             for index, (label, picture) in enumerate(zip(labels, pictures))]
    # A prepared probe shows through both its surfaces.
    markers = (
        Line2D([], [], linestyle="none", marker="o", markersize=10, color="0.4", alpha=1 - (1 - PREPARED_OPACITY) ** 2,
               label="Prepared"),
        Line2D([], [], linestyle="none", marker="o", color="0.4", label="After channel"),
    )
    figure.legend(handles=markers, loc="outside lower center", ncol=2, frameon=False)
    return figure, shown


def draw_channels(surface: core.State, outputs: core.State, prepared: core.State,
                  transformed: core.State, labels: tuple[str, ...]) -> plt.Figure:
    """The sphere of pure inputs and its reconstructed images, with matching probe points: each image a translucent
    surface over the same grid as the sphere, the probes as prepared translucent and after the channel solid."""
    sphere = bloch(surface)                                                    # [latitudes + 1, longitudes + 1, 3]
    images = bloch(outputs)                                                    # [channels, latitudes + 1, longitudes + 1, 3]
    mapped = bloch(transformed)                                                # [channels, preparations, 3]
    view = bloch_view()
    scene = channel_scene(sphere, bloch(prepared), view)
    faces = surface_faces(sphere.shape[0], sphere.shape[1] - 1)
    pictures = [channel_picture(scene, faces, image, points, colour)
                for image, points, colour in zip(images, mapped, CHANNEL_COLOURS)]
    figure, _ = channels_figure(pictures, view, labels)
    return figure


def draw_predictions(observed: core.Scalar, predicted: core.Scalar,
                     labels: tuple[str, ...]) -> plt.Figure:
    """Predicted and measured probabilities for preparations absent from the reconstruction."""
    measured = observed.to_array()                                             # [channels, heldout, outcomes]
    inferred = predicted.to_array()                                            # [channels, heldout, outcomes]
    limit = 1.05 * max(float(measured.max()), float(inferred.max()))
    figure, axes = plt.subplots(figsize=(4.7, 4.5), layout="constrained")
    axes.plot([0, limit], [0, limit], color="0.7", linewidth=1, zorder=0)
    for label, actual, estimate, colour in zip(labels, measured, inferred, CHANNEL_COLOURS):
        axes.scatter(actual, estimate, s=22, color=colour, alpha=0.75,
                     linewidths=0, label=label)
    axes.set(xlim=(0, limit), ylim=(0, limit), aspect="equal", xticks=[], yticks=[],
             xlabel="Observed probability", ylabel="Predicted probability",
             title="Unseen preparations")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="upper left", frameon=False, fontsize=9)
    return figure


def print_completeness(spectra: core.Scalar, labels: tuple[str, ...]) -> None:
    """Print each probe set's singular values, including unobserved directions."""
    for label, spectrum in zip(labels, spectra.to_array()):
        print(f"{label}: {np.array2string(spectrum, precision=4, suppress_small=True)}")


def animate(surface: core.State, prepared: core.State,
            frames: Iterator[tuple[core.State, core.State]],
            labels: tuple[str, ...]) -> Iterator[np.ndarray]:
    """Successive applications of each reconstructed channel to the same input sphere, as `draw_channels` draws
    them: the figure, the axes, the sphere, its grid and the probes as prepared set up once, and each frame's images
    of the sphere and probes after the channels drawn among them."""
    sphere = bloch(surface)                                                    # [latitudes + 1, longitudes + 1, 3]
    view = bloch_view()
    scene = channel_scene(sphere, bloch(prepared), view)
    faces = surface_faces(sphere.shape[0], sphere.shape[1] - 1)
    blank = np.full((*view.shape, 3), 255, dtype=np.uint8)
    figure, shown = channels_figure([blank] * len(labels), view, labels)
    # The layout settled once, so that only the pictures change between frames.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    for output, transformed in frames:
        for image, picture, points, colour in zip(shown, bloch(output), bloch(transformed), CHANNEL_COLOURS):
            image.set_data(channel_picture(scene, faces, picture, points, colour))
        yield capture(figure)
    plt.close(figure)
