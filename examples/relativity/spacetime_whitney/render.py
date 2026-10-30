"""Spacetime Whitney basis fields, causal light-cone wakes, and relativistic moving-mirror Doppler figures."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.tri import Triangulation

from numga import concatenate, stack
from examples.relativity.spacetime_whitney import core
from examples.util import charts
from examples.util.animation import capture

primal_colour = "#286580"
accent_colour = "#d77542"
ink_colour = "#173c4b"
teal_colour = "#148d88"
cone_colour = "#b5892e"
mesh_colour = "#b8c4c9"
wall_colour = "#52616a"
wave_colours = LinearSegmentedColormap.from_list(
    "spacetime_wave", ("#184a63", "#5b9ab5", "#f6f4ee", "#df7a47", "#a6341b"))
n_contours = 7
n_streamlines = 13
# The steps along each streamline, either way from its seed, each this share of the triangle's extent; streamlines
# shorter than this share of it are left out.
n_stream_steps = 240
stream_step = 0.006
min_stream_length = 0.12
# Two triangles and the wake beside them, in nearly the full column.
cone_width_px, cone_height_px = 780, 270
# Three quantity panels side by side, each with its own labelled axes, across the full column.
doppler_width_px, doppler_height_px = 800, 300
animation_width_px, animation_height_px = 680, 300
# A highlighted event or corner, twice a chart's marker across, as a scatter's area in points squared.
highlight_size = charts.points(2 * charts.MARKER_PX) ** 2
# Walls, mirrors and the highlighted edge drawn twice a chart's line; the faint mesh and the markers' white halos as
# thin as a chart's frame.
bold_width = charts.points(2 * charts.LINE_PX)


# --- plumbing -------------------------------------------------------------------------
def coordinates(events: core.Point) -> np.ndarray:
    """Spacetime coordinates (x, t) of intrinsic events and directions."""
    mv = events.context.multivector
    return np.asarray((stack((mv.x, mv.t)) & events.batch()[..., None]).to_array())


def frame_panel(ax: plt.Axes, places: np.ndarray, margin_ratio: float) -> None:
    """Square equal-aspect viewport around `places` with no axes or ticks."""
    low, high = places.min(axis=0), places.max(axis=0)
    centre = (low + high) / 2
    half_span = (1 + margin_ratio) * (high - low).max() / 2
    ax.set(xlim=(centre[0] - half_span, centre[0] + half_span),
           ylim=(centre[1] - half_span, centre[1] + half_span), aspect="equal")
    ax.set_axis_off()


def plain_axes(ax: plt.Axes) -> None:
    """Minimal left and bottom spines with compact tick typography for quantity plots."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.78")
    ax.tick_params(color="0.78")


def null_rays(event: core.Point, before: float, after: float) -> core.Point:
    """The two null rays through an event, at 45 degrees, from a reach `before` into its past to a reach `after` into
    its future, `[rays, 2]`: future time plus or minus space."""
    mv = event.context.multivector
    space, time = mv.x.dual(), -mv.t.dual()                                  # [] Point-, Point-
    nulls = stack((time - space, time + space))                              # [rays] Point-
    return stack((event - before * nulls, event + after * nulls), axis=1)    # [rays, 2] Point-


def picture_length(directions: core.Point) -> core.Scalar:
    """The length a direction is drawn at, with space and time at equal scales: the root of the sum of the squares of
    its readings by `mv.x` and `mv.t`, the picture's own Euclidean measure, not spacetime's."""
    mv = directions.context.multivector
    return ((mv.x & directions).squared() + (mv.t & directions).squared()).square_root()


def draw_light_cone_cross(ax: plt.Axes, origin: core.Point, arm_length: float) -> None:
    """Dashed null rays at 45 degrees through the event `origin`."""
    for ray in coordinates(null_rays(origin, arm_length, arm_length)):
        ax.plot(*ray.T, color=cone_colour, linestyle="--", alpha=0.82, zorder=4)


def streamlines(corners: core.Point, gradients: core.Plane, directions: core.Point, seeds: core.Point) -> list[np.ndarray]:
    """The curves through the seeds `[seeds]` along a direction field affine across the triangle, its values at the
    corners `[corners]`, read out in (x, t) up to where each first leaves the triangle, the short ones left out: each
    step taken along the field at its own midpoint, the field there interpolated by the corners' hats."""
    distances = picture_length(corners[:, None] - corners[None, :]).reshape(-1)
    extent = distances.max()

    def velocity(points: core.Point) -> core.Point:
        # The corner hats at the points weight the field's values at the corners; drawn at unit picture speed.
        hats = 1 + (gradients[:, None] & (points[None] - corners[:, None]))    # [corners, seeds] Scalar+
        field = (hats * directions[:, None]).sum(axis=0)                     # [seeds] Point-
        return field / picture_length(field)

    paths = []
    for step in (stream_step * extent, -stream_step * extent):
        points, path = seeds, [seeds]
        for _ in range(n_stream_steps):
            halfway = points + velocity(points) * (step / 2)                  # [seeds] Point-
            points = points + velocity(halfway) * step
            path.append(points)
        paths.append(stack(path))                                            # [steps + 1, seeds] Point-
    # Both ways from the seed, backward first, so that each streamline runs along itself.
    path = concatenate((paths[1][::-1], paths[0][1:]))                       # [2 * steps + 1, seeds] Point-
    hats = 1 + (gradients[:, None, None] & (path[None] - corners[:, None, None]))   # [corners, 2 * steps + 1, seeds] Scalar+
    inside = (hats >= 0).all(axis=0)                                         # [2 * steps + 1, seeds]
    # Each streamline up to where it first leaves the triangle, either way from its seed.
    forward = np.logical_and.accumulate(inside[n_stream_steps:], axis=0)
    backward = np.logical_and.accumulate(inside[n_stream_steps::-1], axis=0)[::-1]
    kept = np.concatenate((backward[:-1], forward))                          # [2 * steps + 1, seeds]
    steps_kept = kept[1:] & kept[:-1]                                        # [2 * steps, seeds]
    lengths = (picture_length(path[1:] - path[:-1]) * steps_kept).sum(axis=0)   # [seeds]
    lines = np.swapaxes(coordinates(path), 0, 1)                             # [seeds, 2 * steps + 1, 2]
    return [line[keep] for line, keep, length in zip(lines, kept.T, lengths) if length >= min_stream_length * extent]


@charts.style()
def draw_light_cone_whitney(corners: core.Point, points: core.Point,
                            gradients: core.Plane,
                            edge_forms: core.Plane, wake_mesh: core.Mesh,
                            wake_vertices: core.Point, wake: core.Scalar,
                            source_event: core.Point) -> plt.Figure:
    """Corner hat gradient and its Lorentzian reflection, edge 1-form, and causal light-cone wake."""
    mv = corners.context.multivector
    corner_xt = coordinates(corners)
    centroid = corners.mean(axis=0)                                          # [] Point-

    figure = plt.figure(figsize=(cone_width_px / charts.DPI, cone_height_px / charts.DPI))
    hat_ax = figure.add_axes((0.0, 0.02, 0.30, 0.96))
    edge_ax = figure.add_axes((0.31, 0.02, 0.30, 0.96))
    wake_ax = figure.add_axes((0.69, 0.17, 0.29, 0.79))

    # Panel 1: Corner hat contours, light cone, Euclidean normal, and Lorentzian dual.
    index = 1
    others = [(index + 1) % 3, (index + 2) % 3]
    # The hat falls evenly from one at its corner to zero along the opposite side: each level runs parallel to that
    # side, across the two sides from the corner, a share of one less the level of the way along them.
    levels = np.linspace(0.0, 1.0, n_contours + 2)[1:-1]
    level_ends = corners[index] + (1 - levels[:, None]) * (corners[others] - corners[index])   # [levels, 2] Point-
    frame_panel(hat_ax, corner_xt, 0.22)
    hat_ax.add_patch(Polygon(corner_xt, facecolor="#f6f4ee", edgecolor="none", zorder=1))
    hat_ax.add_collection(LineCollection(coordinates(level_ends), colors=primal_colour, alpha=0.55, zorder=2))
    hat_ax.add_patch(Polygon(corner_xt, fill=False, edgecolor=wall_colour, zorder=3))
    hat_ax.plot(*corner_xt[others].T, color=wall_colour, linewidth=bold_width, solid_capstyle="round", zorder=4)
    hat_ax.scatter(*corner_xt.T, color=wall_colour, zorder=5)
    hat_ax.scatter(*corner_xt[index], s=highlight_size, color=accent_colour, edgecolor="white",
                   linewidth=charts.FRAME_WIDTH, zorder=6)
    draw_light_cone_cross(hat_ax, centroid, 0.38)

    # The Euclidean normal -(mv.t >> gradients.dual()) beside the Lorentzian dual gradients.dual(), both drawn from
    # the centroid at the scale that draws the Lorentzian dual a fixed length.
    lorentz_dual = gradients[index].dual()                                   # [] Point-
    euclidean_normal = -(mv.t >> lorentz_dual)                               # [] Point-
    scale = 0.28 / picture_length(lorentz_dual)                              # [] Scalar+
    for direction, colour in ((euclidean_normal, primal_colour), (lorentz_dual, accent_colour)):
        segment = coordinates(stack((centroid, centroid + scale * direction)))   # [2, 2]
        hat_ax.plot(*segment.T, color=colour, linewidth=bold_width, zorder=7)
        hat_ax.scatter(*segment[1], color=colour,
                       edgecolor="white", linewidth=charts.FRAME_WIDTH, zorder=8)
    hat_ax.scatter(*coordinates(centroid), color=ink_colour, zorder=8)

    # Panel 2: Spacetime edge Whitney 1-form circulating around its opposite corner.
    edge_index = 0
    frame_panel(edge_ax, corner_xt, 0.22)
    edge_ax.add_patch(Polygon(corner_xt, facecolor="#f6f4ee", edgecolor=wall_colour, zorder=1))
    draw_light_cone_cross(edge_ax, corners[edge_index], 0.42)
    # The edge form is affine across the triangle; its dual at the corners, the samples where their own hats reach
    # one, sets it everywhere.
    sample_hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+
    corner_field = edge_forms[edge_index, sample_hats.argmax(axis=-1)].dual()   # [corners] Point-
    # Seeds along the edge, drawn a little in towards the centroid.
    start, end = corners[(edge_index + 1) % 3], corners[(edge_index + 2) % 3]   # [] Point-, Point-
    fractions = np.linspace(0.06, 0.94, n_streamlines)
    seeds = (start * (1 - fractions) + end * fractions) * 0.85 + centroid * 0.15   # [seeds] Point-
    edge_ax.add_collection(LineCollection(streamlines(corners, gradients, corner_field, seeds),
                                          colors=primal_colour, zorder=3))
    edge_ax.plot(*coordinates(stack((start, end))).T, color=accent_colour, linewidth=bold_width, zorder=5)
    edge_ax.scatter(*corner_xt[edge_index], s=highlight_size, color=primal_colour, edgecolor="white",
                    linewidth=charts.FRAME_WIDTH, zorder=7)

    # Panel 3: Causal forward light-cone wake from a localized spacetime source event.
    wake_xt = coordinates(wake_vertices)
    wake_tri = Triangulation(*wake_xt.T, triangles=wake_mesh.faces)
    wake_vals = np.asarray(wake.batch().to_array())
    wake_limit = float(np.percentile(np.abs(wake_vals), 99.2))
    wake_ax.tripcolor(wake_tri, wake_vals, shading="gouraud", cmap=wave_colours,
                      norm=Normalize(-wake_limit, wake_limit), zorder=1)
    for ray in coordinates(null_rays(source_event, 0.0, 0.50)):
        wake_ax.plot(*ray.T, color=cone_colour, linestyle="--", alpha=0.85, zorder=3)
    wake_ax.scatter(*coordinates(source_event), s=highlight_size, color=accent_colour, edgecolor="white",
                    linewidth=charts.FRAME_WIDTH, zorder=4)
    duration = float(wake_xt[:, 1].max())
    wake_ax.set(xlim=(0.0, 1.0), ylim=(0.0, duration), aspect="equal",
                xticks=[0.0, 0.5, 1.0], yticks=[0.0, round(duration, 2)],
                xlabel="space $x$", ylabel="time $t$")
    plain_axes(wake_ax)
    return figure


def strip_grid(vertices: core.Point, n_space: int, n_time: int) -> core.Point:
    """The strip's vertices by their time row and their place across it, `[times, space]`, as its constructor lays
    them out, row after row."""
    return vertices.batch().reshape(n_time, n_space)                          # [times, space] Point-


def decimated_mesh_segments(grid: core.Point, step_t: int, step_x: int) -> np.ndarray:
    """Coarse subset of spacetime mesh lines so the deforming moving-mirror grid is visible without clutter: every
    few rows and columns of the grid of events `[times, space]`, and the diagonals across their cells, read out."""
    rows, columns = grid[::step_t], grid[:, ::step_x]                        # [rows, space] Point-, [times, columns] Point-
    # Each diagonal joins a corner of a coarse cell to the corner opposite it.
    diagonals = stack((grid[:-step_t:step_t, :-step_x:step_x], grid[step_t::step_t, step_x::step_x]), axis=-1)   # [rows, columns, 2] Point-
    return np.concatenate([
        np.stack((coordinates(rows[:, :-1]), coordinates(rows[:, 1:])), axis=-2).reshape(-1, 2, 2),
        np.stack((coordinates(columns[:-1]), coordinates(columns[1:])), axis=-2).reshape(-1, 2, 2),
        coordinates(diagonals).reshape(-1, 2, 2),
    ])


@charts.style()
def draw_moving_mirror_doppler(mesh: core.Mesh, vertices: core.Point,
                               n_space: int, n_time: int,
                               wave_rate: core.Scalar, slice_energies: core.Scalar,
                               before_row: int, after_row: int) -> plt.Figure:
    """Spacetime moving-mirror worldsheet, incident vs. Doppler-compressed wave packets, and wave energy gain."""
    mv = vertices.context.multivector
    grid = strip_grid(vertices, n_space, n_time)                             # [times, space] Point-
    wave_rate_vals = np.asarray(wave_rate.batch().to_array())
    wave_rate_grid = wave_rate_vals.reshape(n_time, n_space)
    energies = np.asarray(slice_energies.to_array())
    triangulation = Triangulation(*coordinates(vertices).T, triangles=mesh.faces)
    field_limit = float(np.percentile(np.abs(wave_rate_vals), 99.4))
    # The times of the rows, read on the wall at rest, and the mirror's place at the end.
    row_times = (mv.t & grid[:, 0]).to_array()                               # [times]
    duration = float(row_times[-1])
    final_mirror = float((mv.x & grid[-1, -1]).to_array())

    figure = plt.figure(figsize=(doppler_width_px / charts.DPI, doppler_height_px / charts.DPI))
    world_ax = figure.add_axes((0.07, 0.17, 0.24, 0.80))
    wave_ax = figure.add_axes((0.42, 0.17, 0.24, 0.80))
    energy_ax = figure.add_axes((0.77, 0.17, 0.21, 0.80))

    # Panel 1: 2D (x, t) spacetime worldsheet with deforming mesh and moving mirror worldline.
    world_ax.tripcolor(triangulation, wave_rate_vals, shading="gouraud", cmap=wave_colours,
                       norm=Normalize(-field_limit, field_limit), zorder=1)
    coarse_lines = decimated_mesh_segments(grid, 18, 7)
    world_ax.add_collection(LineCollection(coarse_lines, colors=wall_colour, linewidths=charts.FRAME_WIDTH,
                                           alpha=0.14, zorder=2))
    world_ax.plot(*coordinates(grid[:, 0]).T, color=wall_colour, zorder=4)
    world_ax.plot(*coordinates(grid[:, -1]).T, color=ink_colour, linewidth=bold_width, zorder=5)
    for row, colour in ((before_row, primal_colour), (after_row, accent_colour)):
        world_ax.plot(*coordinates(grid[row]).T, color=colour, linestyle="--", zorder=6)
    world_ax.set(xlim=(0.0, 1.02), ylim=(0.0, duration), aspect="equal",
                 xticks=[0.0, round(final_mirror, 2), 1.0],
                 yticks=[0.0, round(float(row_times[before_row]), 2),
                         round(float(row_times[after_row]), 2), round(duration, 2)],
                 xlabel="space $x$", ylabel="time $t$")
    plain_axes(world_ax)

    # Panel 2: Spatial wave-rate slices before and after relativistic mirror reflection.
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
    wave_ax.set(xlim=(0.0, 1.0), ylim=(-max_amp * 1.08, max_amp * 1.08),
                xticks=[0.0, round(float(x_after[-1]), 2), 1.0],
                yticks=[-1.0, 0.0, 1.0, round(float(after_profile.max()), 2)],
                xlabel="space $x$", ylabel=r"wave rate $\partial_t\phi\,/\,r_0$")
    plain_axes(wave_ax)

    # Panel 3: Total wave energy across spatial slices rising by the Einstein Doppler factor.
    # Each slab's energy at the time halfway between its two rows.
    layer_times = (mv.t & (grid[:-1, 0] + grid[1:, 0]) / 2).to_array()
    relative_energy = energies / energies[before_row]
    final_ratio = round(float(relative_energy[after_row]), 2)
    energy_ax.axhline(1.0, color="0.84", linestyle=":", linewidth=charts.FRAME_WIDTH, zorder=1)
    energy_ax.axhline(final_ratio, color=accent_colour, linestyle=":", linewidth=charts.FRAME_WIDTH, alpha=0.7, zorder=1)
    energy_ax.plot(layer_times, relative_energy, color=ink_colour, zorder=3)
    energy_ax.scatter([row_times[before_row], row_times[after_row]],
                      [1.0, relative_energy[after_row]],
                      c=[primal_colour, accent_colour], zorder=4)
    energy_ax.set(xlim=(0.0, duration), ylim=(0.0, final_ratio * 1.12),
                  xticks=[0.0, round(duration, 2)],
                  yticks=[0.0, 1.0, final_ratio],
                  xlabel="time $t$", ylabel=r"relative wave energy $\mathcal{E}(t) / \mathcal{E}_0$")
    plain_axes(energy_ax)
    return figure


def animate_moving_mirror(mesh: core.Mesh, vertices: core.Point,
                          n_space: int, n_time: int,
                          wave_rate: core.Scalar, frame_rows: Iterable[int]
                          ) -> Iterator[np.ndarray]:
    """Side-by-side spacetime worldsheet slice bar and live 1D relativistic moving-mirror cavity."""
    grid = strip_grid(vertices, n_space, n_time)                             # [times, space] Point-
    wave_rate_vals = np.asarray(wave_rate.batch().to_array())
    wave_rate_grid = wave_rate_vals.reshape(n_time, n_space)
    triangulation = Triangulation(*coordinates(vertices).T, triangles=mesh.faces)
    coarse_lines = decimated_mesh_segments(grid, 18, 7)
    # Each row of events read out once, for the slice and the cavity at its moment.
    grid_xt = coordinates(grid)                                              # [times, space, 2]

    field_limit = float(np.percentile(np.abs(wave_rate_vals), 99.4))
    peak_initial = float(np.abs(wave_rate_grid[20]).max())
    normalized_grid = wave_rate_grid / peak_initial
    y_limit = float(np.abs(normalized_grid).max() * 1.12)
    duration = float(grid_xt[-1, 0, 1])

    with charts.style():
        # The figure, its layout and everything that stays put are drawn once; each frame moves the slice, the mirror
        # and the waveform.
        figure = plt.figure(figsize=(animation_width_px / charts.DPI, animation_height_px / charts.DPI))
        world_ax = figure.add_axes((0.07, 0.17, 0.36, 0.79))
        cavity_ax = figure.add_axes((0.54, 0.17, 0.44, 0.79))

        # Left: the spacetime worldsheet, with the horizontal slice of simultaneity sweeping up it.
        world_ax.tripcolor(triangulation, wave_rate_vals, shading="gouraud", cmap=wave_colours,
                           norm=Normalize(-field_limit, field_limit), zorder=1)
        world_ax.add_collection(LineCollection(coarse_lines, colors=wall_colour,
                                               linewidths=charts.FRAME_WIDTH, alpha=0.1, zorder=2))
        world_ax.plot(*grid_xt[:, 0].T, color=wall_colour, zorder=4)
        world_ax.plot(*grid_xt[:, -1].T, color=ink_colour, linewidth=bold_width, zorder=5)
        slice_line, = world_ax.plot([], [], color=accent_colour, zorder=6)
        slice_end = world_ax.scatter([], [], color=accent_colour, edgecolor="white", linewidth=charts.FRAME_WIDTH,
                                     zorder=7)
        world_ax.set(xlim=(0.0, 1.02), ylim=(0.0, duration), aspect="equal",
                     xticks=[0.0, 1.0], yticks=[0.0, round(duration, 2)],
                     xlabel="space $x$", ylabel="time $t$")
        plain_axes(world_ax)

        # Right: the cavity at that moment, the advancing mirror and the Doppler-compressed waveform.
        cavity_ax.axhline(0.0, color="0.82", linewidth=charts.FRAME_WIDTH, zorder=1)
        beyond = cavity_ax.add_patch(Rectangle((1.02, -y_limit), 0.0, 2 * y_limit,
                                               facecolor="#e8e5dd", edgecolor="none", zorder=2))
        cavity_ax.axvline(0.0, color=wall_colour, linewidth=bold_width, zorder=4)
        mirror = cavity_ax.axvline(1.02, color=ink_colour, linewidth=bold_width, zorder=5)
        waveform_line, = cavity_ax.plot([], [], color=ink_colour, zorder=4)
        cavity_ax.set(xlim=(0.0, 1.02), ylim=(-y_limit, y_limit),
                      xticks=[0.0, round(float(grid_xt[-1, -1, 0]), 2), 1.0],
                      yticks=[-2.0, -1.0, 0.0, 1.0, 2.0],
                      xlabel="space $x$", ylabel=r"wave rate $\partial_t\phi\,/\,r_0$")
        plain_axes(cavity_ax)
        figure.canvas.draw()

        fill = None
        for row in frame_rows:
            x_slice, waveform = grid_xt[row, :, 0], normalized_grid[row]
            wall_x = float(x_slice[-1])
            slice_line.set_data(*grid_xt[row].T)
            slice_end.set_offsets(grid_xt[row, -1][None])
            beyond.set_x(wall_x)
            beyond.set_width(1.02 - wall_x)
            mirror.set_xdata([wall_x, wall_x])
            waveform_line.set_data(x_slice, waveform)
            if fill is not None:
                fill.remove()
            fill = cavity_ax.fill_between(x_slice, 0.0, waveform, color=primal_colour, alpha=0.18, zorder=3)
            yield capture(figure)
        plt.close(figure)
