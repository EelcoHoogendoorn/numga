"""Rendering of spacetime simplex gradients, edge 1-forms, causal light-cone wakes, and moving-mirror Doppler figures."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle

from numga import concatenate, stack
from examples.relativity.moving_mirror import core
from examples.util import charts
from examples.util.animation import capture
from numga.topology import Topology
from numga.render import Camera, contours, drawing, simplices
from examples.util.panels import offset_panel, overlay

mv = core.mv
primal_colour = "#286580"
accent_colour = "#d77542"
ink_colour = "#173c4b"
teal_colour = "#148d88"
cone_colour = "#b5892e"
grid_colour = "#b8c4c9"
wall_colour = "#52616a"
paper_colour = "#f6f4ee"
wave_colours = LinearSegmentedColormap.from_list(
    "spacetime_wave",
    ("#184a63", "#5b9ab5", "#f6f4ee", "#df7a47", "#a6341b"),
)
n_contours = 7
n_streamlines = 13
n_stream_steps = 240
stream_step = 0.006
min_stream_length = 0.12
cone_width_px, cone_height_px = 780, 270
doppler_width_px, doppler_height_px = 800, 300
animation_width_px, animation_height_px = 680, 300
highlight_radius = charts.MARKER_PX
bold_px = 2 * charts.LINE_PX
bold_width = charts.points(bold_px)
cone_dashes = (5.5, 2.5)


# --- plumbing -------------------------------------------------------------------------
def frame_panel(ax: plt.Axes, events: core.Point, margin_ratio: float) -> None:
    """Frames `ax` as a square equal-aspect viewport around `events` (`[...] Point-`) without axes or ticks."""
    readings = stack((mv.x, mv.t))[:, None] & events                           # [2, ...] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                   # [2] Scalar+, Scalar+
    centre = (low + high) / 2                                                  # [2] Scalar+
    half_span = (1 + margin_ratio) * (high - low).max() / 2                    # [] Scalar+
    limits = stack((centre - half_span, centre + half_span), axis=-1).to_array()
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.set_axis_off()


def plain_axes(ax: plt.Axes) -> None:
    """Configures left and bottom spines with compact ticks on `ax`."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.78")
    ax.tick_params(color="0.78")


def ink(colour: str) -> np.ndarray:
    """Converts hex `colour` into a `[3]` float RGB array."""
    return np.array(to_rgb(colour))


def worldsheet(
    events: core.Point,
    topology: Topology,
    values: np.ndarray,
    limit: float,
    camera: Camera,
) -> np.ndarray:
    """Renders nodal `values` across `topology` at `events` with symmetric diverging colours."""
    colours = wave_colours(Normalize(-limit, limit)(values))[:, :3]
    return contours.picture(events, topology, values, colours, camera)


def diagram(ax: plt.Axes, pictures: list[np.ndarray]):
    """Composites `pictures` onto `ax` underneath the visible axis spines."""
    shown = overlay(ax, pictures)
    shown.set_zorder(1)
    ax.set_axis_on()
    return shown


def ringed_markers(
    events: core.Point,
    colour: str,
    radius: float,
    camera: Camera,
) -> list[np.ndarray]:
    """Draws markers of `radius` pixels at `events` with thin white halos."""
    return [
        simplices.markers(events, np.ones(3), radius + charts.FRAME_PX, camera),
        simplices.markers(events, ink(colour), radius, camera),
    ]


def null_rays(event: core.Point, before: float, after: float) -> core.Point:
    """Constructs the two 45-degree null segments (`[2, 2] Point-`) through `event` from `before` to `after`."""
    nulls = stack((core.time - core.space, core.time + core.space))            # [2] Point-
    return stack((event - before * nulls, event + after * nulls), axis=1)      # [2, 2] Point-


def picture_length(directions: core.Point) -> core.Scalar:
    """Euclidean display norm (`[...] Scalar+`) of `directions` at equal space and time plot scales."""
    return ((mv.x & directions).squared() + (mv.t & directions).squared()).square_root()  # [...] Scalar+


def light_cone_cross(origin: core.Point, arm_length: float, camera: Camera, opacity: float) -> np.ndarray:
    """Draws dashed 45-degree null rays of `arm_length` through `origin`."""
    return simplices.strokes(
        null_rays(origin, arm_length, arm_length).reshape(-1),
        Topology.chains([2, 2]),
        ink(cone_colour),
        charts.LINE_PX,
        camera,
        dashes=cone_dashes,
        cap=simplices.Cap.BUTT,
        opacity=opacity,
    )


def streamlines(
    corners: core.Point,
    gradients: core.Plane,
    directions: core.Point,
    seeds: core.Point,
) -> tuple[core.Point, Topology]:
    """Integrates midpoint streamlines from `seeds` along an affine direction field across `corners`."""
    distances = picture_length(corners[:, None] - corners[None, :]).reshape(-1)
    extent = distances.max()

    def velocity(points: core.Point) -> core.Point:
        weights = 1 + (gradients[:, None] & (points[None] - corners[:, None]))  # [3, ...] Scalar+
        field = (weights * directions[:, None]).sum(axis=0)                    # [...] Point-
        return field / picture_length(field)                                   # [...] Point-

    paths = []
    for step in (stream_step * extent, -stream_step * extent):
        points, path = seeds, [seeds]
        for _ in range(n_stream_steps):
            halfway = points + velocity(points) * (step / 2)                   # [...] Point-
            points = points + velocity(halfway) * step                         # [...] Point-
            path.append(points)
        paths.append(stack(path))                                              # [n_stream_steps + 1, ...] Point-
    path = concatenate((paths[1][::-1], paths[0][1:]))                         # [2 * n_stream_steps + 1, ...] Point-
    weights = 1 + (gradients[:, None, None] & (path[None] - corners[:, None, None]))  # [3, 2 * n_stream_steps + 1, ...] Scalar+
    inside = (weights >= 0).all(axis=0)
    forward = np.logical_and.accumulate(inside[n_stream_steps:], axis=0)
    backward = np.logical_and.accumulate(inside[n_stream_steps::-1], axis=0)[::-1]
    kept = np.concatenate((backward[:-1], forward))
    steps_kept = kept[1:] & kept[:-1]
    lengths = (picture_length(path[1:] - path[:-1]) * steps_kept).sum(axis=0)  # [...] Scalar+
    long = lengths >= min_stream_length * extent
    curves = [path[:, seed][keep] for seed, keep in enumerate(kept.T) if long[seed]]
    return concatenate(curves), Topology.chains([curve.shape[0] for curve in curves])


@charts.style()
def light_cone_figure(
    corners: core.Point,
    points: core.Point,
    gradients: core.Plane,
    edge_forms: core.Plane,
    topology: Topology,
    wake_vertices: core.Point,
    wake: core.Scalar,
    source_event: core.Point,
) -> plt.Figure:
    """Renders the three-panel figure of simplex gradient contours, edge 1-form streamlines, and causal light-cone wake."""
    corner_events = corners.batch()                                            # [3] Point-
    centroid = corner_events.mean(axis=0)                                      # [] Point-

    figure = plt.figure(figsize=(cone_width_px / charts.DPI, cone_height_px / charts.DPI))
    gradient_ax = figure.add_axes((0.0, 0.02, 0.30, 0.96))
    edge_ax = figure.add_axes((0.31, 0.02, 0.30, 0.96))
    wake_ax = figure.add_axes((0.69, 0.17, 0.29, 0.79))

    # Panel 1: Simplex gradient level lines, light cone, Euclidean normal, and Lorentzian dual.
    tails, heads = core.triangle.corners[2, 1]
    index = 1
    others = [tails[index], heads[index]]
    levels = np.linspace(0, 1, n_contours + 2)[1:-1]
    level_ends = corner_events[index] + (1 - levels[:, None]) * (corner_events[others] - corner_events[index])  # [n_contours, 2] Point-
    frame_panel(gradient_ax, corner_events, 0.22)
    camera = offset_panel(gradient_ax, mv.x, mv.t)
    lorentz_dual = gradients[index].dual()                                     # [] Point-
    euclidean_normal = -(mv.t >> lorentz_dual)                                 # [] Point-
    scale = 0.28 / picture_length(lorentz_dual)                                # [] Scalar+
    tips = centroid + scale * stack((euclidean_normal, lorentz_dual))          # [2] Point-
    overlay(
        gradient_ax,
        [
            contours.picture(corner_events, core.triangle, np.zeros(3), ink(paper_colour), camera),
            simplices.strokes(
                level_ends.reshape(-1),
                Topology.chains([2] * n_contours),
                ink(primal_colour),
                charts.LINE_PX,
                camera,
                opacity=0.55,
            ),
            simplices.strokes(corner_events, core.triangle, ink(wall_colour), charts.LINE_PX, camera),
            simplices.strokes(corner_events[others], Topology.chains([2]), ink(wall_colour), bold_px, camera),
            light_cone_cross(centroid, 0.38, camera, 0.82),
            simplices.markers(corner_events, ink(wall_colour), charts.MARKER_PX / 2, camera),
            *ringed_markers(corner_events[index][None], accent_colour, highlight_radius, camera),
            simplices.strokes(
                stack((centroid.broadcast_to((2,)), tips), axis=-1).reshape(-1),
                Topology.chains([2, 2]),
                np.stack((ink(primal_colour), ink(accent_colour))),
                bold_px,
                camera,
                cap=simplices.Cap.BUTT,
            ),
            *ringed_markers(tips[:1], primal_colour, charts.MARKER_PX / 2, camera),
            *ringed_markers(tips[1:], accent_colour, charts.MARKER_PX / 2, camera),
            simplices.markers(centroid[None], ink(ink_colour), charts.MARKER_PX / 2, camera),
        ],
    )

    # Panel 2: Spacetime edge Whitney 1-form circulating around its opposite corner.
    edge_index = 0
    frame_panel(edge_ax, corner_events, 0.22)
    camera = offset_panel(edge_ax, mv.x, mv.t)
    sample_weights = 1 + (gradients[:, None] & (points[None, :] - corner_events[:, None]))  # [3, ...] Scalar+
    corner_field = edge_forms[edge_index, sample_weights.argmax(axis=-1)].dual()  # [3] Point-
    start, end = corner_events[tails[edge_index]], corner_events[heads[edge_index]]  # [] Point-, Point-
    fractions = np.linspace(0.06, 0.94, n_streamlines)
    seeds = (start * (1 - fractions) + end * fractions) * 0.85 + centroid * 0.15  # [n_streamlines] Point-
    curves, chains = streamlines(corner_events, gradients, corner_field, seeds)
    overlay(
        edge_ax,
        [
            contours.picture(corner_events, core.triangle, np.zeros(3), ink(paper_colour), camera),
            simplices.strokes(corner_events, core.triangle, ink(wall_colour), charts.LINE_PX, camera),
            simplices.strokes(curves, chains, ink(primal_colour), charts.LINE_PX, camera),
            light_cone_cross(corner_events[edge_index], 0.42, camera, 0.82),
            simplices.strokes(stack((start, end)), Topology.chains([2]), ink(accent_colour), bold_px, camera),
            *ringed_markers(corner_events[edge_index][None], primal_colour, highlight_radius, camera),
        ],
    )

    # Panel 3: Causal forward light-cone wake from a localized spacetime source event.
    wake_vals = np.asarray(wake.batch().to_array())
    wake_limit = float(np.percentile(np.abs(wake_vals), 99.2))
    duration = float((mv.t & wake_vertices.batch()).max().to_array())
    wake_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, duration),
        aspect="equal",
        xticks=[0.0, 0.5, 1.0],
        yticks=[0.0, round(duration, 2)],
        xlabel="space x",
        ylabel="time t",
    )
    plain_axes(wake_ax)
    camera = offset_panel(wake_ax, mv.x, mv.t)
    rays = null_rays(source_event, 0.0, 0.50).reshape(-1)                      # [4] Point-
    diagram(
        wake_ax,
        [
            worldsheet(wake_vertices.batch(), topology, wake_vals, wake_limit, camera),
            simplices.strokes(
                rays,
                Topology.chains([2, 2]),
                ink(cone_colour),
                charts.LINE_PX,
                camera,
                dashes=cone_dashes,
                cap=simplices.Cap.BUTT,
                opacity=0.85,
            ),
            *ringed_markers(source_event[None], accent_colour, highlight_radius, camera),
        ],
    )
    return figure


def strip_grid(vertices: core.Point, n_space: int, n_time: int) -> core.Point:
    """Reshapes strip `vertices` into a `[n_time, n_space] Point-` grid of events."""
    return vertices.batch().reshape(n_time, n_space)                           # [n_time, n_space] Point-


def coarse_grid(grid: core.Point, step_t: int, step_x: int) -> tuple[core.Point, Topology]:
    """Subsamples rows, columns, and diagonals of `grid` into line chains showing strip deformation."""
    n_times, n_space = grid.shape
    rows = grid[::step_t]                                                      # [..., n_space] Point-
    columns = grid[np.arange(n_times)[None, :], np.arange(0, n_space, step_x)[:, None]]  # [..., n_times] Point-
    diagonals = stack((grid[:-step_t:step_t, :-step_x:step_x], grid[step_t::step_t, step_x::step_x]), axis=-1)  # [..., 2] Point-
    lines = concatenate((rows.reshape(-1), columns.reshape(-1), diagonals.reshape(-1)))  # [...] Point-
    return lines, Topology.chains(
        [n_space] * rows.shape[0]
        + [n_times] * columns.shape[0]
        + [2] * (diagonals.shape[0] * diagonals.shape[1])
    )


def world_lines(grid: core.Point, opacity: float, camera: Camera) -> list[np.ndarray]:
    """Draws faint coarse strip lines and bold boundary worldlines from `grid`."""
    lines, chains = coarse_grid(grid, 18, 7)
    n_times = grid.shape[0]
    return [
        simplices.strokes(lines, chains, ink(wall_colour), charts.FRAME_PX, camera, opacity=opacity),
        simplices.strokes(grid[:, 0], Topology.chains([n_times]), ink(wall_colour), charts.LINE_PX, camera),
        simplices.strokes(grid[:, -1], Topology.chains([n_times]), ink(ink_colour), bold_px, camera),
    ]


@charts.style()
def doppler_figure(
    topology: Topology,
    vertices: core.Point,
    n_space: int,
    n_time: int,
    wave_rate: core.Scalar,
    slice_energies: core.Scalar,
    before_row: int,
    after_row: int,
) -> plt.Figure:
    """Renders the moving-mirror worldsheet, incident and Doppler-compressed waveforms, and slab energy curve."""
    grid = strip_grid(vertices, n_space, n_time)                               # [n_time, n_space] Point-
    wave_rate_vals = np.asarray(wave_rate.batch().to_array())
    wave_rate_grid = wave_rate_vals.reshape(n_time, n_space)
    energies = np.asarray(slice_energies.to_array())
    field_limit = float(np.percentile(np.abs(wave_rate_vals), 99.4))
    row_times = (mv.t & grid[:, 0]).to_array()                                 # [n_time]
    duration = float(row_times[-1])
    final_mirror = float((mv.x & grid[-1, -1]).to_array())

    figure = plt.figure(figsize=(doppler_width_px / charts.DPI, doppler_height_px / charts.DPI))
    world_ax = figure.add_axes((0.07, 0.17, 0.24, 0.80))
    wave_ax = figure.add_axes((0.42, 0.17, 0.24, 0.80))
    energy_ax = figure.add_axes((0.77, 0.17, 0.21, 0.80))

    world_ax.set(
        xlim=(0.0, 1.02),
        ylim=(0.0, duration),
        aspect="equal",
        xticks=[0.0, round(final_mirror, 2), 1.0],
        yticks=[
            0.0,
            round(float(row_times[before_row]), 2),
            round(float(row_times[after_row]), 2),
            round(duration, 2),
        ],
        xlabel="space x",
        ylabel="time t",
    )
    plain_axes(world_ax)
    camera = offset_panel(world_ax, mv.x, mv.t)
    diagram(
        world_ax,
        [worldsheet(vertices.batch(), topology, wave_rate_vals, field_limit, camera)]
        + world_lines(grid, 0.14, camera)
        + [
            simplices.strokes(
                grid[[before_row, after_row]].reshape(-1),
                Topology.chains([n_space, n_space]),
                np.repeat([ink(primal_colour), ink(accent_colour)], n_space - 1, axis=0),
                charts.LINE_PX,
                camera,
                dashes=cone_dashes,
                cap=simplices.Cap.BUTT,
            )
        ],
    )

    peak_before = float(np.abs(wave_rate_grid[before_row]).max())
    before_profile = wave_rate_grid[before_row] / peak_before
    after_profile = wave_rate_grid[after_row] / peak_before
    x_before = (mv.x & grid[before_row]).to_array()
    x_after = (mv.x & grid[after_row]).to_array()
    wave_ax.axhline(0.0, color="0.82", linewidth=charts.FRAME_WIDTH, zorder=1)
    wave_ax.fill_between(x_before, 0.0, before_profile, color=primal_colour, alpha=0.14, zorder=2)
    wave_ax.plot(x_before, before_profile, color=primal_colour, zorder=3)
    wave_ax.fill_between(x_after, 0.0, after_profile, color=accent_colour, alpha=0.18, zorder=4)
    wave_ax.plot(x_after, after_profile, color=accent_colour, zorder=5)
    wave_ax.axvline(float(x_after[-1]), color=ink_colour, linestyle=":", zorder=6)
    max_amp = float(np.ceil(np.abs(after_profile).max() * 10) / 10)
    wave_ax.set(
        xlim=(0.0, 1.0),
        ylim=(-max_amp * 1.08, max_amp * 1.08),
        xticks=[0.0, round(float(x_after[-1]), 2), 1.0],
        yticks=[-1.0, 0.0, 1.0, round(float(after_profile.max()), 2)],
        xlabel="space x",
        ylabel="normalized wave rate",
    )
    plain_axes(wave_ax)

    layer_times = (mv.t & (grid[:-1, 0] + grid[1:, 0]) / 2).to_array()
    relative_energy = energies / energies[before_row]
    final_ratio = round(float(relative_energy[after_row]), 2)
    energy_ax.axhline(1.0, color="0.84", linestyle=":", linewidth=charts.FRAME_WIDTH, zorder=1)
    energy_ax.axhline(final_ratio, color=accent_colour, linestyle=":", linewidth=charts.FRAME_WIDTH, alpha=0.7, zorder=1)
    energy_ax.plot(layer_times, relative_energy, color=ink_colour, zorder=3)
    energy_ax.scatter(
        [row_times[before_row], row_times[after_row]],
        [1.0, relative_energy[after_row]],
        c=[primal_colour, accent_colour],
        zorder=4,
    )
    energy_ax.set(
        xlim=(0.0, duration),
        ylim=(0.0, final_ratio * 1.12),
        xticks=[0.0, round(duration, 2)],
        yticks=[0.0, 1.0, final_ratio],
        xlabel="time t",
        ylabel="relative wave energy",
    )
    plain_axes(energy_ax)
    return figure


def cavity_animation(
    topology: Topology,
    vertices: core.Point,
    n_space: int,
    n_time: int,
    wave_rate: core.Scalar,
    frame_rows: Iterable[int],
) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the sweeping worldsheet simultaneity slice and moving-mirror cavity."""
    grid = strip_grid(vertices, n_space, n_time)                               # [n_time, n_space] Point-
    wave_rate_vals = np.asarray(wave_rate.batch().to_array())
    wave_rate_grid = wave_rate_vals.reshape(n_time, n_space)
    places = (mv.x & grid).to_array()                                          # [n_time, n_space]

    field_limit = float(np.percentile(np.abs(wave_rate_vals), 99.4))
    peak_initial = float(np.abs(wave_rate_grid[20]).max())
    normalized_grid = wave_rate_grid / peak_initial
    y_limit = float(np.abs(normalized_grid).max() * 1.12)
    duration = float((mv.t & grid[-1, 0]).to_array())

    with charts.style():
        figure = plt.figure(figsize=(animation_width_px / charts.DPI, animation_height_px / charts.DPI))
        world_ax = figure.add_axes((0.07, 0.17, 0.36, 0.79))
        cavity_ax = figure.add_axes((0.54, 0.17, 0.44, 0.79))

        world_ax.set(
            xlim=(0.0, 1.02),
            ylim=(0.0, duration),
            aspect="equal",
            xticks=[0.0, 1.0],
            yticks=[0.0, round(duration, 2)],
            xlabel="space x",
            ylabel="time t",
        )
        plain_axes(world_ax)
        camera = offset_panel(world_ax, mv.x, mv.t)
        sheet = drawing.composited(
            [worldsheet(vertices.batch(), topology, wave_rate_vals, field_limit, camera)]
            + world_lines(grid, 0.1, camera)
        )
        world = diagram(world_ax, [sheet])

        cavity_ax.axhline(0.0, color="0.82", linewidth=charts.FRAME_WIDTH, zorder=1)
        beyond = cavity_ax.add_patch(
            Rectangle((1.02, -y_limit), 0.0, 2 * y_limit, facecolor="#e8e5dd", edgecolor="none", zorder=2)
        )
        cavity_ax.axvline(0.0, color=wall_colour, linewidth=bold_width, zorder=4)
        mirror = cavity_ax.axvline(1.02, color=ink_colour, linewidth=bold_width, zorder=5)
        (waveform_line,) = cavity_ax.plot([], [], color=ink_colour, zorder=4)
        cavity_ax.set(
            xlim=(0.0, 1.02),
            ylim=(-y_limit, y_limit),
            xticks=[0.0, round(float(places[-1, -1]), 2), 1.0],
            yticks=[-2.0, -1.0, 0.0, 1.0, 2.0],
            xlabel="space x",
            ylabel="normalized wave rate",
        )
        plain_axes(cavity_ax)
        figure.canvas.draw()

        fill = None
        for row in frame_rows:
            x_slice, waveform = places[row], normalized_grid[row]
            wall_x = float(x_slice[-1])
            world.set_data(
                drawing.composited(
                    [
                        sheet,
                        simplices.strokes(grid[row], Topology.chains([n_space]), ink(accent_colour), charts.LINE_PX, camera),
                        *ringed_markers(grid[row, -1:], accent_colour, charts.MARKER_PX / 2, camera),
                    ]
                )
            )
            beyond.set_x(wall_x)
            beyond.set_width(1.02 - wall_x)
            mirror.set_xdata([wall_x, wall_x])
            waveform_line.set_data(x_slice, waveform)
            if fill is not None:
                fill.remove()
            fill = cavity_ax.fill_between(x_slice, 0.0, waveform, color=primal_colour, alpha=0.18, zorder=3)
            yield capture(figure)
        plt.close(figure)
