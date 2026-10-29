"""Spacetime Whitney basis fields, causal light-cone wakes, and relativistic moving-mirror Doppler figures."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.tri import LinearTriInterpolator, Triangulation

from numga import stack
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
n_stream_grid = 80
n_streamlines = 13
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


def draw_light_cone_cross(ax: plt.Axes, origin: np.ndarray, arm_length: float) -> None:
    """Dashed null rays at 45 degrees through `origin`."""
    for sign in (-1.0, 1.0):
        ray = np.array([[origin[0] - sign * arm_length, origin[1] - arm_length],
                        [origin[0] + sign * arm_length, origin[1] + arm_length]])
        ax.plot(*ray.T, color=cone_colour, linestyle="--", alpha=0.82, zorder=4)


@charts.style()
def draw_light_cone_whitney(corners: core.Point, points: core.Point,
                            gradients: core.Plane,
                            edge_forms: core.Plane, wake_mesh: core.Mesh,
                            wake_vertices: core.Point, wake: core.Scalar,
                            source_event: core.Point) -> plt.Figure:
    """Corner hat gradient and its Lorentzian reflection, edge 1-form, and causal light-cone wake."""
    mv = corners.context.multivector
    corner_xt = coordinates(corners)
    sample_xt = coordinates(points)
    centroid = corner_xt.mean(axis=0)

    # The Euclidean normal -(mv.t >> gradients.dual()) beside the Lorentzian dual gradients.dual().
    lorentz_dual_xt = coordinates(gradients.dual())
    euclidean_normal_xt = coordinates(-(mv.t >> gradients.dual()))
    edge_field_xt = coordinates(edge_forms.dual())

    figure = plt.figure(figsize=(cone_width_px / charts.DPI, cone_height_px / charts.DPI))
    hat_ax = figure.add_axes((0.0, 0.02, 0.30, 0.96))
    edge_ax = figure.add_axes((0.31, 0.02, 0.30, 0.96))
    wake_ax = figure.add_axes((0.69, 0.17, 0.29, 0.79))

    # Panel 1: Corner hat contours, light cone, Euclidean normal, and Lorentzian dual.
    index = 1
    single_tri = Triangulation(*corner_xt.T, triangles=np.array([[0, 1, 2]]))
    levels = np.linspace(0.0, 1.0, n_contours + 2)[1:-1]
    frame_panel(hat_ax, corner_xt, 0.22)
    values = np.eye(3)[index]
    hat_ax.add_patch(Polygon(corner_xt, facecolor="#f6f4ee", edgecolor="none", zorder=1))
    hat_ax.tricontour(single_tri, values, levels=levels, colors=primal_colour, alpha=0.55, zorder=2)
    hat_ax.add_patch(Polygon(corner_xt, fill=False, edgecolor=wall_colour, zorder=3))
    opposite = corner_xt[[(index + 1) % 3, (index + 2) % 3]]
    hat_ax.plot(*opposite.T, color=wall_colour, linewidth=bold_width, solid_capstyle="round", zorder=4)
    hat_ax.scatter(*corner_xt.T, color=wall_colour, zorder=5)
    hat_ax.scatter(*corner_xt[index], s=highlight_size, color=accent_colour, edgecolor="white",
                   linewidth=charts.FRAME_WIDTH, zorder=6)
    draw_light_cone_cross(hat_ax, centroid, 0.38)

    scale = 0.28 / np.linalg.norm(lorentz_dual_xt[index])
    euclid_vec = scale * euclidean_normal_xt[index]
    lorentz_vec = scale * lorentz_dual_xt[index]
    for displacement, colour in ((euclid_vec, primal_colour), (lorentz_vec, accent_colour)):
        segment = np.stack((centroid, centroid + displacement))
        hat_ax.plot(*segment.T, color=colour, linewidth=bold_width, zorder=7)
        hat_ax.scatter(*(centroid + displacement), color=colour,
                       edgecolor="white", linewidth=charts.FRAME_WIDTH, zorder=8)
    hat_ax.scatter(*centroid, color=ink_colour, zorder=8)

    # Panel 2: Spacetime edge Whitney 1-form circulating around its opposite corner.
    edge_index = 0
    frame_panel(edge_ax, corner_xt, 0.22)
    edge_ax.add_patch(Polygon(corner_xt, facecolor="#f6f4ee", edgecolor=wall_colour, zorder=1))
    draw_light_cone_cross(edge_ax, corner_xt[edge_index], 0.42)
    # Trace the sampled dual field continuously; the mask clips each curve to the triangle.
    sample_tri = Triangulation(*sample_xt.T)
    grid_x = np.linspace(corner_xt[:, 0].min(), corner_xt[:, 0].max(), n_stream_grid)
    grid_t = np.linspace(corner_xt[:, 1].min(), corner_xt[:, 1].max(), n_stream_grid)
    stream_x, stream_t = np.meshgrid(grid_x, grid_t)
    component_x = LinearTriInterpolator(sample_tri, edge_field_xt[edge_index, :, 0])(stream_x, stream_t)
    component_t = LinearTriInterpolator(sample_tri, edge_field_xt[edge_index, :, 1])(stream_x, stream_t)
    start, end = corner_xt[(edge_index + 1) % 3], corner_xt[(edge_index + 2) % 3]
    fractions = np.linspace(0.06, 0.94, n_streamlines)[:, None]
    seeds = (start * (1 - fractions) + end * fractions) * 0.85 + centroid * 0.15
    edge_ax.streamplot(grid_x, grid_t, component_x, component_t,
                       color=primal_colour, start_points=seeds,
                       arrowstyle="-", broken_streamlines=False, minlength=0.12, zorder=3)
    edge_ax.plot([start[0], end[0]], [start[1], end[1]], color=accent_colour, linewidth=bold_width, zorder=5)
    edge_ax.scatter(*corner_xt[edge_index], s=highlight_size, color=primal_colour, edgecolor="white",
                    linewidth=charts.FRAME_WIDTH, zorder=7)

    # Panel 3: Causal forward light-cone wake from a localized spacetime source event.
    wake_xt = coordinates(wake_vertices)
    wake_tri = Triangulation(*wake_xt.T, triangles=wake_mesh.faces)
    wake_vals = np.asarray(wake.batch().to_array())
    source_xt = coordinates(source_event)
    wake_limit = float(np.percentile(np.abs(wake_vals), 99.2))
    wake_ax.tripcolor(wake_tri, wake_vals, shading="gouraud", cmap=wave_colours,
                      norm=Normalize(-wake_limit, wake_limit), zorder=1)
    ray_span = 0.50
    for sign in (-1.0, 1.0):
        wake_ax.plot([source_xt[0], source_xt[0] + sign * ray_span],
                     [source_xt[1], source_xt[1] + ray_span],
                     color=cone_colour, linestyle="--", alpha=0.85, zorder=3)
    wake_ax.scatter(*source_xt, s=highlight_size, color=accent_colour, edgecolor="white",
                    linewidth=charts.FRAME_WIDTH, zorder=4)
    duration = float(wake_xt[:, 1].max())
    wake_ax.set(xlim=(0.0, 1.0), ylim=(0.0, duration), aspect="equal",
                xticks=[0.0, 0.5, 1.0], yticks=[0.0, round(duration, 2)],
                xlabel="space $x$", ylabel="time $t$")
    plain_axes(wake_ax)
    return figure


def decimated_mesh_segments(xt_grid: np.ndarray, step_t: int, step_x: int) -> np.ndarray:
    """Coarse subset of spacetime mesh lines so the deforming moving-mirror grid is visible without clutter."""
    segments = []
    for row in range(0, xt_grid.shape[0], step_t):
        segments.append(np.stack((xt_grid[row, :-1], xt_grid[row, 1:]), axis=1))
    for col in range(0, xt_grid.shape[1], step_x):
        segments.append(np.stack((xt_grid[:-1, col], xt_grid[1:, col]), axis=1))
    rows = np.arange(0, xt_grid.shape[0] - step_t, step_t)
    cols = np.arange(0, xt_grid.shape[1] - step_x, step_x)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    diags = np.stack((xt_grid[rr, cc], xt_grid[rr + step_t, cc + step_x]), axis=-2).reshape(-1, 2, 2)
    segments.append(diags)
    return np.concatenate(segments, axis=0)


@charts.style()
def draw_moving_mirror_doppler(mesh: core.Mesh, vertices: core.Point,
                               n_space: int, n_time: int,
                               wave_rate: core.Scalar, slice_energies: core.Scalar,
                               before_row: int, after_row: int) -> plt.Figure:
    """Spacetime moving-mirror worldsheet, incident vs. Doppler-compressed wave packets, and wave energy gain."""
    places = coordinates(vertices)
    xt_grid = places.reshape(n_time, n_space, 2)
    wave_rate_vals = np.asarray(wave_rate.batch().to_array())
    wave_rate_grid = wave_rate_vals.reshape(n_time, n_space)
    energies = np.asarray(slice_energies.to_array())
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    field_limit = float(np.percentile(np.abs(wave_rate_vals), 99.4))
    duration = float(xt_grid[-1, 0, 1])

    figure = plt.figure(figsize=(doppler_width_px / charts.DPI, doppler_height_px / charts.DPI))
    world_ax = figure.add_axes((0.07, 0.17, 0.24, 0.80))
    wave_ax = figure.add_axes((0.42, 0.17, 0.24, 0.80))
    energy_ax = figure.add_axes((0.77, 0.17, 0.21, 0.80))

    # Panel 1: 2D (x, t) spacetime worldsheet with deforming mesh and moving mirror worldline.
    world_ax.tripcolor(triangulation, wave_rate_vals, shading="gouraud", cmap=wave_colours,
                       norm=Normalize(-field_limit, field_limit), zorder=1)
    coarse_lines = decimated_mesh_segments(xt_grid, 18, 7)
    world_ax.add_collection(LineCollection(coarse_lines, colors=wall_colour, linewidths=charts.FRAME_WIDTH,
                                           alpha=0.14, zorder=2))
    world_ax.plot(xt_grid[:, 0, 0], xt_grid[:, 0, 1], color=wall_colour, zorder=4)
    world_ax.plot(xt_grid[:, -1, 0], xt_grid[:, -1, 1], color=ink_colour, linewidth=bold_width, zorder=5)
    for row, colour in ((before_row, primal_colour), (after_row, accent_colour)):
        world_ax.plot(xt_grid[row, :, 0], xt_grid[row, :, 1], color=colour, linestyle="--", zorder=6)
    world_ax.set(xlim=(0.0, 1.02), ylim=(0.0, duration), aspect="equal",
                 xticks=[0.0, round(float(xt_grid[-1, -1, 0]), 2), 1.0],
                 yticks=[0.0, round(float(xt_grid[before_row, 0, 1]), 2),
                         round(float(xt_grid[after_row, 0, 1]), 2), round(duration, 2)],
                 xlabel="space $x$", ylabel="time $t$")
    plain_axes(world_ax)

    # Panel 2: Spatial wave-rate slices before and after relativistic mirror reflection.
    peak_before = float(np.abs(wave_rate_grid[before_row]).max())
    before_profile = wave_rate_grid[before_row] / peak_before
    after_profile = wave_rate_grid[after_row] / peak_before
    x_before = xt_grid[before_row, :, 0]
    x_after = xt_grid[after_row, :, 0]
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
    layer_times = 0.5 * (xt_grid[:-1, 0, 1] + xt_grid[1:, 0, 1])
    relative_energy = energies / energies[before_row]
    final_ratio = round(float(relative_energy[after_row]), 2)
    energy_ax.axhline(1.0, color="0.84", linestyle=":", linewidth=charts.FRAME_WIDTH, zorder=1)
    energy_ax.axhline(final_ratio, color=accent_colour, linestyle=":", linewidth=charts.FRAME_WIDTH, alpha=0.7, zorder=1)
    energy_ax.plot(layer_times, relative_energy, color=ink_colour, zorder=3)
    energy_ax.scatter([xt_grid[before_row, 0, 1], xt_grid[after_row, 0, 1]],
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
    places = coordinates(vertices)
    xt_grid = places.reshape(n_time, n_space, 2)
    wave_rate_vals = np.asarray(wave_rate.batch().to_array())
    wave_rate_grid = wave_rate_vals.reshape(n_time, n_space)
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    coarse_lines = decimated_mesh_segments(xt_grid, 18, 7)

    field_limit = float(np.percentile(np.abs(wave_rate_vals), 99.4))
    peak_initial = float(np.abs(wave_rate_grid[20]).max())
    normalized_grid = wave_rate_grid / peak_initial
    y_limit = float(np.abs(normalized_grid).max() * 1.12)
    duration = float(xt_grid[-1, 0, 1])

    with charts.style():
        for row in frame_rows:
            figure = plt.figure(figsize=(animation_width_px / charts.DPI, animation_height_px / charts.DPI))
            world_ax = figure.add_axes((0.07, 0.17, 0.36, 0.79))
            cavity_ax = figure.add_axes((0.54, 0.17, 0.44, 0.79))

            # Left: Spacetime worldsheet with sweeping horizontal simultaneity slice.
            world_ax.tripcolor(triangulation, wave_rate_vals, shading="gouraud", cmap=wave_colours,
                               norm=Normalize(-field_limit, field_limit), zorder=1)
            world_ax.add_collection(LineCollection(coarse_lines, colors=wall_colour,
                                                   linewidths=charts.FRAME_WIDTH, alpha=0.1, zorder=2))
            world_ax.plot(xt_grid[:, 0, 0], xt_grid[:, 0, 1], color=wall_colour, zorder=4)
            world_ax.plot(xt_grid[:, -1, 0], xt_grid[:, -1, 1], color=ink_colour, linewidth=bold_width, zorder=5)
            world_ax.plot(xt_grid[row, :, 0], xt_grid[row, :, 1], color=accent_colour, zorder=6)
            world_ax.scatter(xt_grid[row, -1, 0], xt_grid[row, -1, 1],
                             color=accent_colour, edgecolor="white", linewidth=charts.FRAME_WIDTH, zorder=7)
            world_ax.set(xlim=(0.0, 1.02), ylim=(0.0, duration), aspect="equal",
                         xticks=[0.0, 1.0], yticks=[0.0, round(duration, 2)],
                         xlabel="space $x$", ylabel="time $t$")
            plain_axes(world_ax)

            # Right: Live 1D cavity with advancing mirror piston and Doppler-compressed waveform.
            x_slice = xt_grid[row, :, 0]
            wall_x = float(x_slice[-1])
            waveform = normalized_grid[row]
            cavity_ax.axhline(0.0, color="0.82", linewidth=charts.FRAME_WIDTH, zorder=1)
            cavity_ax.add_patch(Rectangle((wall_x, -y_limit), 1.02 - wall_x, 2 * y_limit,
                                          facecolor="#e8e5dd", edgecolor="none", zorder=2))
            cavity_ax.axvline(0.0, color=wall_colour, linewidth=bold_width, zorder=4)
            cavity_ax.axvline(wall_x, color=ink_colour, linewidth=bold_width, zorder=5)
            cavity_ax.fill_between(x_slice, 0.0, waveform, color=primal_colour, alpha=0.18, zorder=3)
            cavity_ax.plot(x_slice, waveform, color=ink_colour, zorder=4)
            cavity_ax.set(xlim=(0.0, 1.02), ylim=(-y_limit, y_limit),
                          xticks=[0.0, round(float(xt_grid[-1, -1, 0]), 2), 1.0],
                          yticks=[-2.0, -1.0, 0.0, 1.0, 2.0],
                          xlabel="space $x$", ylabel=r"wave rate $\partial_t\phi\,/\,r_0$")
            plain_axes(cavity_ax)

            image = capture(figure)
            plt.close(figure)
            yield image
