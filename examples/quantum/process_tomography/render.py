"""Ray-traced Bloch balls and matplotlib charts display probe probabilities, reconstructed channels, and predictions."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D

from numga import Extensor, concatenate
from numga.render import Scene, meshes, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology
from examples.quantum.process_tomography import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import fitted, labels, show

channel_colours = ("#3275a8", "#d16a30", "#7b62a3")
probe_colours = ("#3275a8", "#d16a30", "#438e65", "#7b62a3")
probe_names = ("A", "B", "C", "D")
sphere_colour, sphere_opacity = np.full(3, 0.85), 0.06
grid_colour, grid_opacity = np.full(3, 0.6), 0.6
edge_colour, axis_colour = np.full(3, 0.75), np.full(3, 0.55)
image_opacity, prepared_opacity = 0.2, 0.2
elevation, azimuth = np.radians(24.0), np.radians(-54.0 + 90.0)
reach, margin, n_pixels = 1.45, 0.2, 700
panel_size = (260, 305)
experiment_size, predictions_size = (charts.DESIGN_WIDTH, 250), (300, 300)
gallery_panel_size = (240, 285)
latitude_step, longitude_step = 2, 3
grid_width, image_width, edge_width, axis_width = 0.00829, 0.0141, 0.0166, 0.00994
tip_radius = 0.0207
probe_radius, output_radius, prepared_radius = 0.0497, 0.029, 0.0456
axis_length, label_distance = 1.3, 1.4


# --- plumbing -------------------------------------------------------------------------
def ball_positions(points: core.Point) -> solids.Point:
    """Unit-weight normalization places projective Bloch-ball states as renderer points (`[...] Point-`)."""
    return points.normalized()


def bloch_positions(states: Extensor) -> solids.Point:
    """Twice each Pauli state's dualized vector part offsets its Bloch-vector tip from the origin (`[...] Point-`)."""
    return solids.origin + solids.context.include_from_algebra((2 * states.select.vector()).dual())


Reading = Callable[[Extensor], solids.Point]
axis_names = solids.origin + solids.axes.dual() * label_distance             # [3] Point-


def bloch_view(axes) -> solids.View:
    """A square view around the origin fits the shared Bloch-ball camera to `axes`."""
    raw_view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, reach), n_pixels)
    return fitted(raw_view, axes)


def channel_view(sphere_points: solids.Point, axes) -> solids.View:
    """A padded bounding view frames the pure-state sphere and axis labels inside `axes`."""
    raw_view = solids.view([sphere_points, axis_names], 0.0, elevation, azimuth, margin, n_pixels)
    return fitted(raw_view, axes)


def bloch_axes() -> list[tuple[solids.Quadric, np.ndarray]]:
    """Three coordinate rods with tip balls mark the `x`, `y`, and `z` axes through the centre."""
    tips = solids.origin + solids.axes.dual() * axis_length                  # [3] Point-
    return [(solids.rods(solids.origin, tips, axis_width / 2), axis_colour),
            (solids.ball(tips, tip_radius)[:, None], axis_colour)]


def ball_panel(axes, image: np.ndarray, view: solids.View):
    """A panel artist displays a ray-traced Bloch-ball image on `axes` with coordinate labels past the tips."""
    artist = show(axes, image, view)
    labels(axes, view, axis_names, list("xyz"), ha="center", va="center", color=axis_colour)
    return artist


def grid_lines(points: solids.Point) -> tuple[solids.Point, solids.Point]:
    """Strided slicing selects interior latitude rings and longitude meridians from a sphere grid."""
    rows = np.arange(points.shape[0])
    columns = np.arange(0, points.shape[1], longitude_step)
    return points[latitude_step:-1:latitude_step], points[rows[None, :], columns[:, None]]


def grid_tubes(points: solids.Point, colour: np.ndarray, width: float, view: solids.View) -> list[Draw]:
    """Closed latitude loops and pole-to-pole longitude chains form tube strokes across a sphere grid."""
    latitudes, longitudes = grid_lines(points)                               # [...] Point-, [...] Point-
    loops = Topology.loops([latitudes.shape[1]] * latitudes.shape[0])
    chains = Topology.chains([longitudes.shape[1]] * longitudes.shape[0])
    return [simplices.tubes(latitudes.reshape(-1), loops, colour, width / view.pixel, view, gloss=solids.gloss),
            simplices.tubes(longitudes.reshape(-1), chains, colour, width / view.pixel, view, gloss=solids.gloss)]


def grid_planes(points: solids.Point) -> solids.Plane:
    """Three-point joins along each latitude and longitude curve define the sphere's inked cutting planes."""
    planes = []
    for curves in grid_lines(points):
        n_curve_points = curves.shape[1]
        indices = (0, n_curve_points // 3, 2 * n_curve_points // 3)
        first, second, third = (curves[:, index] for index in indices)       # [...] Point-, [...] Point-, [...] Point-
        planes.append((first & second & third) * solids.flip)
    return concatenate(planes)


@charts.style()
def experiment_figure(
    preparations: Extensor, probabilities: Extensor, channel_labels: tuple[str, ...], reading: Reading,
) -> plt.Figure:
    """Four tetrahedral probes in the Bloch ball sit beside log-scaled probability heatmaps for each channel."""
    probes = reading(preparations)                                           # [4] Point-
    probability_tables = probabilities.to_array()
    figure = plt.figure(
        figsize=(experiment_size[0] / charts.DPI, experiment_size[1] / charts.DPI), layout="constrained",
    )
    figure.get_layout_engine().set(wspace=0.12)
    probe_axes = figure.add_subplot(1, len(channel_labels) + 1, 1)
    probe_axes.set_title("Prepared states")
    probe_axes.set_axis_off()
    probability_scale = LogNorm(vmin=probability_tables[probability_tables > 0].min(), vmax=0.5)
    table_axes = []
    centres = np.arange(len(probe_names)) + 0.5
    for index, (channel_label, table) in enumerate(zip(channel_labels, probability_tables)):
        axes = figure.add_subplot(1, len(channel_labels) + 1, index + 2)
        table_axes.append(axes)
        heatmap = axes.pcolormesh(table, norm=probability_scale, cmap="turbo")
        axes.set(xticks=centres, xticklabels=probe_names, yticks=centres, yticklabels=probe_names, aspect="equal")
        axes.invert_yaxis()
        axes.set(xlabel="Measurement outcome", ylabel="Prepared state", title=channel_label)
        axes.tick_params(length=0)
    colour_bar = figure.colorbar(heatmap, ax=table_axes, shrink=0.72, pad=0.02)
    colour_bar.set_label("Probability (log scale)")

    view = bloch_view(probe_axes)
    colours = np.array([to_rgb(colour) for colour in probe_colours])
    # Joining all pairs of tetrahedral probes makes their three-dimensional span across the ball visible.
    tail_indices, head_indices = np.triu_indices(len(probes), k=1)
    edges = solids.rods(probes[tail_indices], probes[head_indices], edge_width / 2)
    scene = solids.draw(
        [*bloch_axes(), (edges, edge_colour), (solids.ball(probes, probe_radius)[:, None], colours)],
        view, np.ones(3), inks=[], translucent=[], meshes=[],
    )
    ball_panel(probe_axes, scene, view)
    name_points = solids.origin + (probes - solids.origin) * 1.12            # [4] Point-
    for (x, y), name, colour in zip(view.across(name_points), probe_names, probe_colours):
        probe_axes.text(x, y, name, color=colour)
    return figure


def channel_scene(sphere_points: solids.Point, probes: solids.Point, view: solids.View) -> Scene:
    """Coordinate axes, the inked translucent input sphere, and translucent prepared probes form the shared scene."""
    colours = np.array([to_rgb(colour) for colour in probe_colours])
    translucent = [(solids.ball(solids.origin, 1.0)[None, None], sphere_colour, sphere_opacity),
                   (solids.ball(probes, prepared_radius)[:, None], colours, prepared_opacity)]
    planes = grid_planes(sphere_points)                                      # [...] Plane+
    axes = bloch_axes()
    reference = solids.Ink(
        len(axes) + 1, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
        grid_colour, grid_width / view.pixel, grid_opacity,
    )
    return Scene(view, np.ones(3), axes, [], inks=[reference], translucent=translucent)


def channel_picture(
    scene: Scene, surface: Topology, image: solids.Point, probe_points: solids.Point, colour: str,
) -> np.ndarray:
    """A translucent output shell, deformed grid tubes, and solid output balls render one channel inside `scene`."""
    view = scene.view
    colours = np.array([to_rgb(probe_colour) for probe_colour in probe_colours])
    places = image.reshape(-1)                                               # [...] Point-
    shell = meshes.samples(
        places, surface.elements[2], np.array(to_rgb(colour)), meshes.smooth(places, surface), view,
        face_values=np.zeros(4), colour=meshes.plain, gloss=0.1,
    )
    tubes = grid_tubes(image, np.array(to_rgb(colour)), image_width, view)
    output_balls = solids.ball(probe_points, output_radius)[:, None]
    return scene.draw(
        [(output_balls, colours)], inks=[], translucent=[],
        meshes=[(shell, image_opacity)] + [(tube, 1.0) for tube in tubes],
    )


def channels_axes(channel_labels: tuple[str, ...], panel: tuple[float, float]) -> tuple[plt.Figure, list]:
    """Side-by-side channel panels sit above a shared legend comparing prepared and outgoing probe markers."""
    figure = plt.figure(
        figsize=(panel[0] * len(channel_labels) / charts.DPI, panel[1] / charts.DPI), layout="constrained",
    )
    axes = [figure.add_subplot(1, len(channel_labels), index + 1) for index in range(len(channel_labels))]
    for ax, channel_label in zip(axes, channel_labels):
        ax.set_title(channel_label)
        ax.set_axis_off()
    markers = (
        Line2D(
            [], [], linestyle="none", marker="o",
            markersize=charts.MARKER_SIZE * prepared_radius / output_radius,
            color="0.4", alpha=1 - (1 - prepared_opacity) ** 2, label="Prepared",
        ),
        Line2D([], [], linestyle="none", marker="o", color="0.4", label="After channel"),
    )
    figure.legend(handles=markers, loc="outside lower center", ncol=2, frameon=False)
    return figure, axes


@charts.style()
def channels_figure(
    surface: Extensor, outputs: Extensor, preparations: Extensor, probe_outputs: Extensor,
    channel_labels: tuple[str, ...], reading: Reading,
) -> plt.Figure:
    """Side-by-side Bloch-ball panels show each reconstructed channel's image of the sphere and probes."""
    sphere_points = reading(surface)                                         # [...] Point-
    images = reading(outputs)                                                # [...] Point-
    probes = reading(preparations)                                           # [4] Point-
    probe_images = reading(probe_outputs)                                    # [..., 4] Point-
    figure, axes = channels_axes(channel_labels, panel_size)
    view = channel_view(sphere_points, axes[0])
    scene = channel_scene(sphere_points, probes, view)
    grid_surface = Topology.loops([sphere_points.shape[1]]).product(Topology.chains([sphere_points.shape[0]]))
    for ax, image, points, colour in zip(axes, images, probe_images, channel_colours):
        ball_panel(ax, channel_picture(scene, grid_surface, image, points, colour), view)
    return figure


@charts.style()
def predictions_figure(
    observed_probabilities: Extensor, predicted_probabilities: Extensor, channel_labels: tuple[str, ...],
) -> plt.Figure:
    """A parity scatter chart compares predicted and exact outcome probabilities on unseen validation states."""
    observed_values = observed_probabilities.to_array()
    predicted_values = predicted_probabilities.to_array()
    limit = 1.05 * max(float(observed_values.max()), float(predicted_values.max()))
    figure, axes = charts.subplots(*predictions_size, layout="constrained")
    axes.plot([0, limit], [0, limit], color="0.7", linewidth=charts.FRAME_WIDTH, zorder=0)
    for channel_label, actual, estimate, colour in zip(channel_labels, observed_values, predicted_values, channel_colours):
        axes.scatter(actual, estimate, color=colour, alpha=0.75, linewidths=0, label=channel_label)
    axes.set(
        xlim=(0, limit), ylim=(0, limit), aspect="equal", xticks=[], yticks=[],
        xlabel="Observed probability", ylabel="Predicted probability", title="Unseen preparations",
    )
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="upper left", frameon=False)
    return figure


def channel_animation(
    surface: Extensor, preparations: Extensor, channel_labels: tuple[str, ...], reading: Reading,
    panel: tuple[float, float], frames: Iterator[tuple[Extensor, Extensor]],
) -> Iterator[np.ndarray]:
    """Yields rendered RGB frames showing successive applications of each reconstructed channel."""
    with charts.style():
        sphere_points = reading(surface)                                     # [...] Point-
        figure, axes = channels_axes(channel_labels, panel)
        view = channel_view(sphere_points, axes[0])
        scene = channel_scene(sphere_points, reading(preparations), view)
        grid_surface = Topology.loops([sphere_points.shape[1]]).product(Topology.chains([sphere_points.shape[0]]))
        blank = np.full((*view.shape, 3), 255, dtype=np.uint8)
        artists = [ball_panel(ax, blank, view) for ax in axes]
        figure.canvas.draw()
        figure.set_layout_engine("none")
        for outputs, probe_outputs in frames:
            for artist, picture, points, colour in zip(artists, reading(outputs), reading(probe_outputs), channel_colours):
                artist.set_data(channel_picture(scene, grid_surface, picture, points, colour))
            yield capture(figure)
        plt.close(figure)
