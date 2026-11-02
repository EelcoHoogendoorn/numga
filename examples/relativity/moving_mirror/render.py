"""Corner hats and edge forms on spacetime, causal light-cone wakes, and relativistic moving-mirror Doppler figures."""

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
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera, panel

primal_colour = "#286580"
accent_colour = "#d77542"
ink_colour = "#173c4b"
teal_colour = "#148d88"
cone_colour = "#b5892e"
mesh_colour = "#b8c4c9"
wall_colour = "#52616a"
paper_colour = "#f6f4ee"
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
# A highlighted event or corner, twice a chart's marker across, as a radius in pixels.
highlight_radius = charts.MARKER_PX
# Walls, mirrors and the highlighted edge drawn twice a chart's line, in pixels, and in points for the charts; the faint
# mesh and the markers' white halos as thin as a chart's frame.
bold_px = 2 * charts.LINE_PX
bold_width = charts.points(bold_px)
# The light cone's dashes, a dash and a gap in pixels.
cone_dashes = (5.5, 2.5)


# --- plumbing -------------------------------------------------------------------------
def frame_panel(ax: plt.Axes, events: core.Point, margin_ratio: float) -> None:
    """Square equal-aspect viewport around the events `[events]`, read across by the plane x and up by t, with no axes
    or ticks."""
    mv = events.context.multivector
    readings = stack((mv.x, mv.t))[:, None] & events                          # [2, events] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                  # [2] Scalar+, Scalar+
    centre = (low + high) / 2
    half_span = (1 + margin_ratio) * (high - low).max() / 2
    limits = stack((centre - half_span, centre + half_span), axis=-1).to_array()
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.set_axis_off()


def plain_axes(ax: plt.Axes) -> None:
    """Minimal left and bottom spines with compact tick typography for quantity plots."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.78")
    ax.tick_params(color="0.78")


def ink(colour: str) -> np.ndarray:
    """A colour's red, green and blue."""
    return np.array(to_rgb(colour))


def worldsheet(events: core.Point, topology: core.Topology, values: np.ndarray, limit: float, camera: Camera) -> np.ndarray:
    """A field on a world sheet's events coloured, as a spacetime diagram's camera sees it: its picture."""
    colours = wave_colours(Normalize(-limit, limit)(values))[:, :3]
    return contours.picture(events, topology, values, colours, camera)


def diagram(ax: plt.Axes, pictures: list[np.ndarray]):
    """A spacetime diagram's pictures laid over each other into its panel, under the panel's frame, which stays; the
    image, for an animation to set its data."""
    shown = simplices.overlay(ax, pictures)
    shown.set_zorder(1)
    ax.set_axis_on()
    return shown


def ringed(events: core.Point, colour: str, radius: float, camera: Camera) -> list[np.ndarray]:
    """Dots at events, `radius` pixels, ringed in white as thin as a chart's frame: their pictures."""
    return [simplices.markers(events, np.ones(3), radius + charts.FRAME_PX, camera),
            simplices.markers(events, ink(colour), radius, camera)]


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


def light_cone_cross(origin: core.Point, arm_length: float, camera: Camera, opacity: float) -> np.ndarray:
    """Dashed null rays at 45 degrees through the event `origin`: their picture."""
    return simplices.strokes(null_rays(origin, arm_length, arm_length).reshape(-1), simplices.Topology.chains([2, 2]),
                             ink(cone_colour), charts.LINE_PX, camera, dashes=cone_dashes, cap=simplices.Cap.BUTT,
                             opacity=opacity)


def streamlines(corners: core.Point, gradients: core.Plane, directions: core.Point,
                seeds: core.Point) -> tuple[core.Point, core.Topology]:
    """The curves through the seeds `[seeds]` along a direction field affine across the triangle, its values at the
    corners `[corners]`, up to where each first leaves the triangle, the short ones left out: their events, one curve
    after another, and the chains of edges through them. Each step is taken along the field at its own midpoint, the
    field there interpolated by the corners' hats."""
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
            halfway = points + velocity(points) * (step / 2)                 # [seeds] Point-
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
    lengths = (picture_length(path[1:] - path[:-1]) * steps_kept).sum(axis=0)   # [seeds] Scalar+
    long = lengths >= min_stream_length * extent                             # [seeds]
    curves = [path[:, seed][keep] for seed, keep in enumerate(kept.T) if long[seed]]
    return concatenate(curves), simplices.Topology.chains([curve.shape[0] for curve in curves])


@charts.style()
def draw_light_cone_whitney(corners: core.Point, points: core.Point,
                            gradients: core.Plane,
                            edge_forms: core.Plane, topology: core.Topology,
                            wake_vertices: core.Point, wake: core.Scalar,
                            source_event: core.Point) -> plt.Figure:
    """Corner hat gradient and its Lorentzian reflection, edge 1-form, and causal light-cone wake."""
    mv = corners.context.multivector
    centroid = corners.mean(axis=0)                                          # [] Point-

    figure = plt.figure(figsize=(cone_width_px / charts.DPI, cone_height_px / charts.DPI))
    hat_ax = figure.add_axes((0.0, 0.02, 0.30, 0.96))
    edge_ax = figure.add_axes((0.31, 0.02, 0.30, 0.96))
    wake_ax = figure.add_axes((0.69, 0.17, 0.29, 0.79))

    # Panel 1: Corner hat contours, light cone, Euclidean normal, and Lorentzian dual.
    tails, heads = core.triangle.corners[2, 1]
    index = 1
    others = [tails[index], heads[index]]
    # The hat falls evenly from one at its corner to zero along the opposite side: each level runs parallel to that
    # side, across the two sides from the corner, a share of one less the level of the way along them.
    levels = np.linspace(0.0, 1.0, n_contours + 2)[1:-1]
    level_ends = corners[index] + (1 - levels[:, None]) * (corners[others] - corners[index])   # [levels, 2] Point-
    frame_panel(hat_ax, corners, 0.22)
    camera = panel(hat_ax, mv.x, mv.t)
    # The Euclidean normal -(mv.t >> gradients.dual()) beside the Lorentzian dual gradients.dual(), both drawn from
    # the centroid at the scale that draws the Lorentzian dual a fixed length.
    lorentz_dual = gradients[index].dual()                                   # [] Point-
    euclidean_normal = -(mv.t >> lorentz_dual)                               # [] Point-
    scale = 0.28 / picture_length(lorentz_dual)                              # [] Scalar+
    tips = centroid + scale * stack((euclidean_normal, lorentz_dual))        # [2] Point-
    simplices.overlay(hat_ax, [
        contours.picture(corners, core.triangle, np.zeros(3), ink(paper_colour), camera),
        simplices.strokes(level_ends.reshape(-1), simplices.Topology.chains([2] * n_contours), ink(primal_colour),
                          charts.LINE_PX, camera, opacity=0.55),
        simplices.strokes(corners, core.triangle, ink(wall_colour), charts.LINE_PX, camera),
        simplices.strokes(corners[others], simplices.Topology.chains([2]), ink(wall_colour), bold_px, camera),
        light_cone_cross(centroid, 0.38, camera, 0.82),
        simplices.markers(corners, ink(wall_colour), charts.MARKER_PX / 2, camera),
        *ringed(corners[index][None], accent_colour, highlight_radius, camera),
        simplices.strokes(stack((centroid.broadcast_to((2,)), tips), axis=-1).reshape(-1),
                          simplices.Topology.chains([2, 2]), np.stack((ink(primal_colour), ink(accent_colour))), bold_px,
                          camera, cap=simplices.Cap.BUTT),
        *ringed(tips[:1], primal_colour, charts.MARKER_PX / 2, camera),
        *ringed(tips[1:], accent_colour, charts.MARKER_PX / 2, camera),
        simplices.markers(centroid[None], ink(ink_colour), charts.MARKER_PX / 2, camera)])

    # Panel 2: Spacetime edge Whitney 1-form circulating around its opposite corner.
    edge_index = 0
    frame_panel(edge_ax, corners, 0.22)
    camera = panel(edge_ax, mv.x, mv.t)
    # The edge form is affine across the triangle; its dual at the corners, the samples where their own hats reach
    # one, sets it everywhere.
    sample_hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+
    corner_field = edge_forms[edge_index, sample_hats.argmax(axis=-1)].dual()   # [corners] Point-
    # Seeds along the edge, drawn a little in towards the centroid.
    start, end = corners[tails[edge_index]], corners[heads[edge_index]]      # [] Point-, Point-
    fractions = np.linspace(0.06, 0.94, n_streamlines)
    seeds = (start * (1 - fractions) + end * fractions) * 0.85 + centroid * 0.15  # [seeds] Point-
    curves, chains = streamlines(corners, gradients, corner_field, seeds)
    simplices.overlay(edge_ax, [
        contours.picture(corners, core.triangle, np.zeros(3), ink(paper_colour), camera),
        simplices.strokes(corners, core.triangle, ink(wall_colour), charts.LINE_PX, camera),
        simplices.strokes(curves, chains, ink(primal_colour), charts.LINE_PX, camera),
        light_cone_cross(corners[edge_index], 0.42, camera, 0.82),
        simplices.strokes(stack((start, end)), simplices.Topology.chains([2]), ink(accent_colour), bold_px, camera),
        *ringed(corners[edge_index][None], primal_colour, highlight_radius, camera)])

    # Panel 3: Causal forward light-cone wake from a localized spacetime source event.
    wake_vals = np.asarray(wake.batch().to_array())
    wake_limit = float(np.percentile(np.abs(wake_vals), 99.2))
    # The wake's last moment, read by the plane t.
    duration = float((mv.t & wake_vertices.batch()).max().to_array())
    wake_ax.set(xlim=(0.0, 1.0), ylim=(0.0, duration), aspect="equal",
                xticks=[0.0, 0.5, 1.0], yticks=[0.0, round(duration, 2)],
                xlabel="space $x$", ylabel="time $t$")
    plain_axes(wake_ax)
    camera = panel(wake_ax, mv.x, mv.t)
    rays = null_rays(source_event, 0.0, 0.50).reshape(-1)                   # [rays * 2] Point-
    diagram(wake_ax, [
        worldsheet(wake_vertices.batch(), topology, wake_vals, wake_limit, camera),
        simplices.strokes(rays, simplices.Topology.chains([2, 2]), ink(cone_colour), charts.LINE_PX, camera,
                          dashes=cone_dashes, cap=simplices.Cap.BUTT, opacity=0.85),
        *ringed(source_event[None], accent_colour, highlight_radius, camera)])
    return figure


def strip_grid(vertices: core.Point, n_space: int, n_time: int) -> core.Point:
    """The strip's vertices by their time row and their place across it, `[times, space]`, as its constructor lays
    them out, row after row."""
    return vertices.batch().reshape(n_time, n_space)                          # [times, space] Point-


def coarse_grid(grid: core.Point, step_t: int, step_x: int) -> tuple[core.Point, core.Topology]:
    """Coarse subset of spacetime mesh lines so the deforming moving-mirror grid is visible without clutter: every
    few rows and columns of the grid of events `[times, space]`, and the diagonals across their cells; their events,
    one line after another, and the chains of edges along them."""
    n_times, n_space = grid.shape
    rows = grid[::step_t]                                                    # [rows, space] Point-
    # Every few columns, each with its events in their order up it.
    columns = grid[np.arange(n_times)[None, :], np.arange(0, n_space, step_x)[:, None]]   # [columns, times] Point-
    # Each diagonal joins a corner of a coarse cell to the corner opposite it.
    diagonals = stack((grid[:-step_t:step_t, :-step_x:step_x], grid[step_t::step_t, step_x::step_x]), axis=-1)   # [rows, columns, 2] Point-
    lines = concatenate((rows.reshape(-1), columns.reshape(-1), diagonals.reshape(-1)))
    return lines, simplices.Topology.chains([n_space] * rows.shape[0] + [n_times] * columns.shape[0]
                                        + [2] * (diagonals.shape[0] * diagonals.shape[1]))


def world_lines(grid: core.Point, opacity: float, camera: Camera) -> list[np.ndarray]:
    """The world sheet's coarse lines, faint, and the worldlines of the wall at rest and of the mirror, bold, read from
    the grid of events `[times, space]`: their pictures."""
    lines, chains = coarse_grid(grid, 18, 7)
    n_times = grid.shape[0]
    return [simplices.strokes(lines, chains, ink(wall_colour), charts.FRAME_PX, camera, opacity=opacity),
            simplices.strokes(grid[:, 0], simplices.Topology.chains([n_times]), ink(wall_colour), charts.LINE_PX, camera),
            simplices.strokes(grid[:, -1], simplices.Topology.chains([n_times]), ink(ink_colour), bold_px, camera)]


@charts.style()
def draw_moving_mirror_doppler(topology: core.Topology, vertices: core.Point,
                               n_space: int, n_time: int,
                               wave_rate: core.Scalar, slice_energies: core.Scalar,
                               before_row: int, after_row: int) -> plt.Figure:
    """Spacetime moving-mirror worldsheet, incident vs. Doppler-compressed wave packets, and wave energy gain."""
    mv = vertices.context.multivector
    grid = strip_grid(vertices, n_space, n_time)                              # [times, space] Point-
    wave_rate_vals = np.asarray(wave_rate.batch().to_array())
    wave_rate_grid = wave_rate_vals.reshape(n_time, n_space)
    energies = np.asarray(slice_energies.to_array())
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
    world_ax.set(xlim=(0.0, 1.02), ylim=(0.0, duration), aspect="equal",
                 xticks=[0.0, round(final_mirror, 2), 1.0],
                 yticks=[0.0, round(float(row_times[before_row]), 2),
                         round(float(row_times[after_row]), 2), round(duration, 2)],
                 xlabel="space $x$", ylabel="time $t$")
    plain_axes(world_ax)
    camera = panel(world_ax, mv.x, mv.t)
    diagram(world_ax, [worldsheet(vertices.batch(), topology, wave_rate_vals, field_limit, camera)]
            + world_lines(grid, 0.14, camera)
            + [simplices.strokes(grid[[before_row, after_row]].reshape(-1), simplices.Topology.chains([n_space, n_space]),
                                 np.repeat([ink(primal_colour), ink(accent_colour)], n_space - 1, axis=0),
                                 charts.LINE_PX, camera, dashes=cone_dashes, cap=simplices.Cap.BUTT)])

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


def animate_moving_mirror(topology: core.Topology, vertices: core.Point,
                          n_space: int, n_time: int,
                          wave_rate: core.Scalar, frame_rows: Iterable[int]
                          ) -> Iterator[np.ndarray]:
    """Side-by-side spacetime worldsheet slice bar and live 1D relativistic moving-mirror cavity."""
    mv = vertices.context.multivector
    grid = strip_grid(vertices, n_space, n_time)                              # [times, space] Point-
    wave_rate_vals = np.asarray(wave_rate.batch().to_array())
    wave_rate_grid = wave_rate_vals.reshape(n_time, n_space)
    # Each row of events' places across the cavity, read by the plane x once, for the cavity at its moment.
    places = (mv.x & grid).to_array()                                        # [times, space]

    field_limit = float(np.percentile(np.abs(wave_rate_vals), 99.4))
    peak_initial = float(np.abs(wave_rate_grid[20]).max())
    normalized_grid = wave_rate_grid / peak_initial
    y_limit = float(np.abs(normalized_grid).max() * 1.12)
    duration = float((mv.t & grid[-1, 0]).to_array())

    with charts.style():
        # The figure, its layout and everything that stays put are drawn once; each frame moves the slice, the mirror
        # and the waveform.
        figure = plt.figure(figsize=(animation_width_px / charts.DPI, animation_height_px / charts.DPI))
        world_ax = figure.add_axes((0.07, 0.17, 0.36, 0.79))
        cavity_ax = figure.add_axes((0.54, 0.17, 0.44, 0.79))

        # Left: the spacetime worldsheet, with the horizontal slice of simultaneity sweeping up it.
        world_ax.set(xlim=(0.0, 1.02), ylim=(0.0, duration), aspect="equal",
                     xticks=[0.0, 1.0], yticks=[0.0, round(duration, 2)],
                     xlabel="space $x$", ylabel="time $t$")
        plain_axes(world_ax)
        camera = panel(world_ax, mv.x, mv.t)
        sheet = simplices.composited([worldsheet(vertices.batch(), topology, wave_rate_vals, field_limit, camera)]
                                     + world_lines(grid, 0.1, camera))
        world = diagram(world_ax, [sheet])

        # Right: the cavity at that moment, the advancing mirror and the Doppler-compressed waveform.
        cavity_ax.axhline(0.0, color="0.82", linewidth=charts.FRAME_WIDTH, zorder=1)
        beyond = cavity_ax.add_patch(Rectangle((1.02, -y_limit), 0.0, 2 * y_limit,
                                               facecolor="#e8e5dd", edgecolor="none", zorder=2))
        cavity_ax.axvline(0.0, color=wall_colour, linewidth=bold_width, zorder=4)
        mirror = cavity_ax.axvline(1.02, color=ink_colour, linewidth=bold_width, zorder=5)
        waveform_line, = cavity_ax.plot([], [], color=ink_colour, zorder=4)
        cavity_ax.set(xlim=(0.0, 1.02), ylim=(-y_limit, y_limit),
                      xticks=[0.0, round(float(places[-1, -1]), 2), 1.0],
                      yticks=[-2.0, -1.0, 0.0, 1.0, 2.0],
                      xlabel="space $x$", ylabel=r"wave rate $\partial_t\phi\,/\,r_0$")
        plain_axes(cavity_ax)
        figure.canvas.draw()

        fill = None
        for row in frame_rows:
            x_slice, waveform = places[row], normalized_grid[row]
            wall_x = float(x_slice[-1])
            world.set_data(simplices.composited([
                sheet, simplices.strokes(grid[row], simplices.Topology.chains([n_space]), ink(accent_colour),
                                         charts.LINE_PX, camera),
                *ringed(grid[row, -1:], accent_colour, charts.MARKER_PX / 2, camera)]))
            beyond.set_x(wall_x)
            beyond.set_width(1.02 - wall_x)
            mirror.set_xdata([wall_x, wall_x])
            waveform_line.set_data(x_slice, waveform)
            if fill is not None:
                fill.remove()
            fill = cavity_ax.fill_between(x_slice, 0.0, waveform, color=primal_colour, alpha=0.18, zorder=3)
            yield capture(figure)
        plt.close(figure)
