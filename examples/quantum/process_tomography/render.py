"""Probe probabilities, reconstructed channels and predictions drawn from quantum states, read either as points of
the projective Bloch ball or, for the conventional twin notebook, as states of the Pauli algebra; each drawing is
handed its reading. The Bloch balls are drawn on the GPU, the rest by matplotlib."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D

from numga import Extensor, concatenate
from examples.util.animation import capture
from examples.quantum.process_tomography import core
from examples.util import charts
from examples.util.wgsl import fitted, labels, meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene

CHANNEL_COLOURS = ("#3275a8", "#d16a30", "#7b62a3")
PROBE_COLOURS = ("#3275a8", "#d16a30", "#438e65", "#7b62a3")
PROBE_NAMES = ("A", "B", "C", "D")
# The sphere of pure inputs, faint, and its grid; the edges joining the probes, and the axes.
SPHERE_COLOUR, SPHERE_OPACITY = np.full(3, 0.85), 0.06
GRID_COLOUR, GRID_OPACITY = np.full(3, 0.6), 0.6
EDGE_COLOUR, AXIS_COLOUR = np.full(3, 0.75), np.full(3, 0.55)
# The opacities of a channel's image of the sphere and of the probes as prepared.
IMAGE_OPACITY, PREPARED_OPACITY = 0.2, 0.2
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the square view of
# the probes about the centre, the margin about a channel's ball and its axes as a share of the larger side, and the
# pixels across a view, until it is fitted to its axes.
ELEVATION, AZIMUTH = np.radians(24.0), np.radians(-54.0 + 90.0)
REACH, MARGIN, N_PIXELS = 1.45, 0.2, 700
# The width and height in pixels of each channel's panel, its title and its share of the legend; the width and height
# of the figure of the experiment, the whole column for the prepared states beside the three tables and their bar,
# and of the predictions, a few dozen points about a diagonal.
PANEL = (260, 305)
EXPERIMENT, PREDICTIONS = (charts.DESIGN_WIDTH, 250), (300, 300)
# Gallery picture: this makes plots/tomography.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
GALLERY_PANEL = (240, 285)
# The grid lines drawn, every few of the sampled latitudes and longitudes; the widths of the sphere's grid lines and
# an image's, the edges and the axes' shafts, the arrowheads' length and width, and the radii of the probes alone,
# after a channel and as prepared, in radii of the ball; how far the axes reach and where they are named.
LATITUDE_STEP, LONGITUDE_STEP = 2, 3
GRID_WIDTH, IMAGE_WIDTH, EDGE_WIDTH, AXIS_WIDTH = 0.00829, 0.0141, 0.0166, 0.00994
HEAD_LENGTH, HEAD_WIDTH = 0.0746, 0.0414
PROBE_RADIUS, AFTER_RADIUS, PREPARED_RADIUS = 0.0497, 0.029, 0.0456
AXIS_LENGTH, LABEL_DISTANCE = 1.3, 1.4


# --- plumbing -------------------------------------------------------------------------
def ball_positions(points: core.Point) -> solids.Point:
    """The points of the projective Bloch ball at unit weight, the renderer's points as they are: the ball is drawn in
    the algebra it lives in."""
    return points.normalized()


def bloch_positions(states: Extensor) -> solids.Point:
    """The points at the tips of the Bloch vectors of states of the Pauli algebra, a half plus half the Bloch vector:
    the dual of twice each state's vector part is the offset of that tip in the algebra of space, carried into the
    renderer's and placed from its origin."""
    return solids.origin + solids.context.include_from_algebra((2 * states.select.vector()).dual())


# How states are placed in the Bloch ball: `ball_positions` or `bloch_positions`.
Reading = Callable[[Extensor], solids.Point]
# Where the axes are named, just past their tips.
axis_names = solids.origin + solids.axes.dual() * LABEL_DISTANCE                  # [3] Point-


def bloch_view(axes) -> solids.View:
    """The view of the Bloch ball every picture shares, fitted to the matplotlib axes it is shown in."""
    return fitted(solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, REACH), N_PIXELS), axes)


def channel_view(sphere: solids.Point, axes) -> solids.View:
    """The view of a channel's pictures, framing the sphere of pure inputs `[latitudes + 1, longitudes + 1]` and the
    axes' names with a margin, fitted to the matplotlib axes it is shown in."""
    framed = solids.view([sphere, axis_names], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    return fitted(framed, axes)


def bloch_axes() -> list[tuple[solids.Quadric, np.ndarray]]:
    """The frame of the x, y and z axes."""
    shafts, heads = solids.frame(AXIS_LENGTH, AXIS_WIDTH / 2, HEAD_LENGTH, HEAD_WIDTH / 2)
    return [(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR)]


def show_ball(axes, image: np.ndarray, view: solids.View):
    """A traced picture in an axes, with the axes named past their tips. The image, for an animation to set its
    data."""
    shown = show(axes, image, view)
    labels(axes, view, axis_names, list("xyz"), ha="center", va="center", color=AXIS_COLOUR)
    return shown


def surface_faces(rows: int, columns: int) -> np.ndarray:
    """The triangles of a grid of points closed around its columns, from pole to pole, indices into the grid
    flattened: two to a cell, but one where a cell's row collapses into a pole."""
    index = np.arange(rows * columns).reshape(rows, columns)
    index = np.concatenate([index, index[:, :1]], axis=1)                      # [rows, columns + 1]
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], axis=-1)[:-1].reshape(-1, 3),
                           np.stack([a, c, d], axis=-1)[1:].reshape(-1, 3)])


def grid_lines(points: solids.Point) -> tuple[solids.Point, solids.Point]:
    """Every few latitudes, but the poles, `[lines, longitudes + 1]`, and every few longitudes, `[lines, latitudes +
    1]`, of a grid of points `[latitudes + 1, longitudes + 1]`."""
    # Each longitude drawn runs through every latitude, a row of the grid each.
    rows = np.arange(points.shape[0])
    columns = np.arange(0, points.shape[1] - 1, LONGITUDE_STEP)
    return points[LATITUDE_STEP:-1:LATITUDE_STEP], points[rows[None, :], columns[:, None]]


def grid(points: solids.Point, colour: np.ndarray, width: float, view: solids.View) -> meshes.Samples:
    """The grid lines of a grid of points `[latitudes + 1, longitudes + 1]`, as strokes `width` across in a colour,
    at every sample."""
    lines = [line for batch in grid_lines(points) for line in batch]
    return strokes.polylines([(line, colour) for line in lines], width / view.pixel, view)


def grid_planes(points: solids.Point) -> solids.Plane:
    """The planes of the grid lines of a grid of points on a sphere `[latitudes + 1, longitudes + 1]`, each through
    its line's first point and those a third and two thirds along it: the join of the three, made extrinsic by the
    intrinsic scalar one."""
    planes = []
    for lines in grid_lines(points):
        count = lines.shape[1]
        first, second, third = (lines[:, index] for index in (0, count // 3, 2 * count // 3))   # [lines] Point-, Point-, Point-
        planes.append((first & second & third) * solids.flip)
    return concatenate(planes)


@charts.style()
def draw_experiment(prepared: Extensor, probabilities: Extensor, labels: tuple[str, ...],
                    reading: Reading) -> plt.Figure:
    """Four tetrahedral preparations and each channel's measured outcome probabilities."""
    probes = reading(prepared)                                               # [preparations] Point-
    measured = probabilities.to_array()                                      # [channels, preparations, outcomes]
    figure = plt.figure(figsize=(EXPERIMENT[0] / charts.DPI, EXPERIMENT[1] / charts.DPI), layout="constrained")
    figure.get_layout_engine().set(wspace=0.12)
    probe_axes = figure.add_subplot(1, len(labels) + 1, 1)
    probe_axes.set_title("Prepared states")
    probe_axes.set_axis_off()
    probability_scale = LogNorm(vmin=measured[measured > 0].min(), vmax=0.5)
    table_axes = []
    # Each table's cells as patches, the first preparation at the top.
    centres = np.arange(len(PROBE_NAMES)) + 0.5
    for index, (label, table) in enumerate(zip(labels, measured)):
        axes = figure.add_subplot(1, len(labels) + 1, index + 2)
        table_axes.append(axes)
        heatmap = axes.pcolormesh(table, norm=probability_scale, cmap="turbo")
        axes.set(xticks=centres, xticklabels=PROBE_NAMES, yticks=centres, yticklabels=PROBE_NAMES, aspect="equal")
        axes.invert_yaxis()
        axes.set(xlabel="Measurement outcome", ylabel="Prepared state", title=label)
        axes.tick_params(length=0)
    colour_bar = figure.colorbar(heatmap, ax=table_axes, shrink=0.72, pad=0.02)
    colour_bar.set_label("Probability (log scale)")

    view = bloch_view(probe_axes)
    colours = np.array([to_rgb(colour) for colour in PROBE_COLOURS])
    # Joining the four probes along the edges of the tetrahedron they span makes their coverage of space visible.
    tetrahedron = core.Mesh(len(probes), np.arange(len(probes))[None])
    tails, heads = (tetrahedron.incidence[1, 0] * probes.field()).batch()     # [edges] Point-, [edges] Point-
    edges = solids.rods(tails, heads, EDGE_WIDTH / 2)
    scene = solids.draw([*bloch_axes(), (edges, EDGE_COLOUR),
                         (solids.ball(probes, PROBE_RADIUS)[:, None], colours)], view, np.ones(3), inks=[],
                         translucent=[], meshes=[])
    show_ball(probe_axes, scene, view)
    # Each probe named a little further out from the centre than the probe itself.
    names = solids.origin + (probes - solids.origin) * 1.12                   # [preparations] Point-
    for (x, y), name, colour in zip(view.across(names), PROBE_NAMES, PROBE_COLOURS):
        probe_axes.text(x, y, name, color=colour)
    return figure


def channel_scene(sphere: solids.Point, probes: solids.Point, view: solids.View) -> Scene:
    """What every picture of a channel shares, from the sphere of pure inputs `[latitudes + 1, longitudes + 1]` and
    the probes as prepared `[preparations]`: the axes; the sphere and the probes, translucent; and the sphere's
    grid inked where the planes of its lines cut it, the sphere the first part after the axes and the probes after
    the channel, which each picture adds."""
    colours = np.array([to_rgb(colour) for colour in PROBE_COLOURS])
    translucent = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR, SPHERE_OPACITY),
                   (solids.ball(probes, PREPARED_RADIUS)[:, None], colours, PREPARED_OPACITY)]
    planes = grid_planes(sphere)                                               # [lines] Plane+
    axes = bloch_axes()
    reference = solids.Ink(len(axes) + 1, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                           GRID_COLOUR, GRID_WIDTH / view.pixel, GRID_OPACITY)
    return Scene(view, np.ones(3), axes, [], inks=[reference], translucent=translucent)


def channel_picture(scene: Scene, faces: np.ndarray, image: solids.Point, points: solids.Point, colour: str
                    ) -> np.ndarray:
    """A channel's picture among the scene's shared parts: its image of the sphere `[latitudes + 1, longitudes + 1]`,
    a translucent surface over the same grid as the sphere, with its grid lines, and the probes after the channel
    `[preparations]`, solid."""
    view = scene.view
    colours = np.array([to_rgb(colour) for colour in PROBE_COLOURS])
    # The reference grid and the channel's image use the same sampled directions.
    places = image[:, :-1].reshape(-1)                                       # [(latitudes + 1) * longitudes] Point-
    shell = meshes.samples(places, faces, np.array(to_rgb(colour)), meshes.smooth(places, meshes.Mesh(len(places), faces)), view,
                           face_values=np.zeros(4), colour=meshes.plain, gloss=0.1)
    lines = grid(image, np.array(to_rgb(colour)), IMAGE_WIDTH, view)
    after = solids.ball(points, AFTER_RADIUS)[:, None]
    return scene.draw([(after, colours)], inks=[], translucent=[], meshes=[(shell, IMAGE_OPACITY), (lines, 1.0)])


def channels_figure(labels: tuple[str, ...], panel: tuple[float, float]) -> tuple[plt.Figure, list]:
    """The channels' axes side by side, each named, over the legend of the probes, each channel a panel of a width
    and height in pixels: the figure, and the axes."""
    figure = plt.figure(figsize=(panel[0] * len(labels) / charts.DPI, panel[1] / charts.DPI), layout="constrained")
    axes = [figure.add_subplot(1, len(labels), index + 1) for index in range(len(labels))]
    for ax, label in zip(axes, labels):
        ax.set_title(label)
        ax.set_axis_off()
    # A prepared probe shows through both its surfaces, and is as much larger than a probe after the channel as on the
    # balls.
    markers = (
        Line2D([], [], linestyle="none", marker="o", markersize=charts.MARKER_SIZE * PREPARED_RADIUS / AFTER_RADIUS,
               color="0.4", alpha=1 - (1 - PREPARED_OPACITY) ** 2, label="Prepared"),
        Line2D([], [], linestyle="none", marker="o", color="0.4", label="After channel"),
    )
    figure.legend(handles=markers, loc="outside lower center", ncol=2, frameon=False)
    return figure, axes


@charts.style()
def draw_channels(surface: Extensor, outputs: Extensor, prepared: Extensor, transformed: Extensor,
                  labels: tuple[str, ...], reading: Reading) -> plt.Figure:
    """The sphere of pure inputs and its reconstructed images, with matching probe points: each image a translucent
    surface over the same grid as the sphere, the probes as prepared translucent and after the channel solid."""
    sphere = reading(surface)                                                # [latitudes + 1, longitudes + 1] Point-
    images = reading(outputs)                                                # [channels, latitudes + 1, longitudes + 1] Point-
    probes = reading(prepared)                                               # [preparations] Point-
    mapped = reading(transformed)                                            # [channels, preparations] Point-
    figure, axes = channels_figure(labels, PANEL)
    view = channel_view(sphere, axes[0])
    scene = channel_scene(sphere, probes, view)
    faces = surface_faces(sphere.shape[0], sphere.shape[1] - 1)
    for ax, image, points, colour in zip(axes, images, mapped, CHANNEL_COLOURS):
        show_ball(ax, channel_picture(scene, faces, image, points, colour), view)
    return figure


@charts.style()
def draw_predictions(observed: Extensor, predicted: Extensor,
                     labels: tuple[str, ...]) -> plt.Figure:
    """Predicted and measured probabilities for preparations absent from the reconstruction."""
    measured = observed.to_array()                                             # [channels, heldout, outcomes]
    inferred = predicted.to_array()                                            # [channels, heldout, outcomes]
    limit = 1.05 * max(float(measured.max()), float(inferred.max()))
    figure, axes = charts.subplots(*PREDICTIONS, layout="constrained")
    axes.plot([0, limit], [0, limit], color="0.7", linewidth=charts.FRAME_WIDTH, zorder=0)
    for label, actual, estimate, colour in zip(labels, measured, inferred, CHANNEL_COLOURS):
        axes.scatter(actual, estimate, color=colour, alpha=0.75, linewidths=0, label=label)
    axes.set(xlim=(0, limit), ylim=(0, limit), aspect="equal", xticks=[], yticks=[], xlabel="Observed probability",
             ylabel="Predicted probability", title="Unseen preparations")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="upper left", frameon=False)
    return figure


def animate(surface: Extensor, prepared: Extensor, frames: Iterator[tuple[Extensor, Extensor]],
            labels: tuple[str, ...], reading: Reading, panel: tuple[float, float]) -> Iterator[np.ndarray]:
    """Successive applications of each reconstructed channel to the same input sphere, as `draw_channels` draws them,
    each channel a panel of a width and height in pixels: the figure, the axes, the sphere, its grid and the probes as
    prepared set up once, and each frame's images drawn among them."""
    with charts.style():
        sphere = reading(surface)                                                # [latitudes + 1, longitudes + 1] Point-
        figure, axes = channels_figure(labels, panel)
        view = channel_view(sphere, axes[0])
        scene = channel_scene(sphere, reading(prepared), view)
        faces = surface_faces(sphere.shape[0], sphere.shape[1] - 1)
        blank = np.full((*view.shape, 3), 255, dtype=np.uint8)
        shown = [show_ball(ax, blank, view) for ax in axes]
        # The layout settled once, so that only the pictures change between frames.
        figure.canvas.draw()
        figure.set_layout_engine("none")
        for outputs, transformed in frames:
            for image, picture, points, colour in zip(shown, reading(outputs), reading(transformed), CHANNEL_COLOURS):
                image.set_data(channel_picture(scene, faces, picture, points, colour))
            yield capture(figure)
        plt.close(figure)
