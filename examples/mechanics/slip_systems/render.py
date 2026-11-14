"""Stereographic views display slip-system regions and tensile-axis trajectories in a face-centred cubic crystal."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from numga import stack
from numga.render import Camera, drawing, simplices
from numga.topology import Topology

from examples.mechanics.slip_systems import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import overlay, panel, pinned, room

system_cmap = "tab20"
# Leading Schmid factors mapped to white and to full system saturation.
faint_factor, full_factor = 0.25, 0.5
path_colour = "0.1"
normal_colour = "0.15"
edge_colour = "0.3"
corner_names = ("[001]", "[011]", "[111]")
triangle_corners = core.mv.vector([[0, 0, 1], [0, 1, 1], [1, 1, 1]]).normalized().dual().field()   # Direction-[3]
goal_directions = core.mv.vector([[1, 0, 1], [1, 1, 2]]).normalized().dual()                       # [2] Direction-
goal_names = ("[101]", "[112]")
path_shade = 0.55
corner_lifts = (-10, 8, 8)
n_edge_samples = 59
n_rim_samples = 180
disk_window = (-1.02, 1.02)
triangle_window = (-0.04, 0.46)
panel_pixels = 300
normal_reach = 0.75 * charts.MARKER_PX * np.sqrt(2)
goal_reach = charts.MARKER_PX


# --- plumbing -------------------------------------------------------------------------
def stereographic_bisectors(directions: core.Direction) -> core.Direction:
    """Upper-hemisphere bisectors with `[001]` (`[...] Direction-`) project to the stereographic disk."""
    unit = directions.normalized()                                            # [...] Direction-
    upper = unit * np.where((core.mv.z & unit) < 0, -1, 1)                    # [...] Direction-
    return core.mv.z.dual() + upper                                           # [...] Direction-


def disk_coordinates(directions: core.Direction) -> np.ndarray:
    """Central projection of upper-hemisphere bisectors gives 2D disk coordinates (`[..., 2]`) for text labels."""
    bisectors = stereographic_bisectors(directions)                           # [...] Direction-
    return ((stack((core.mv.x, core.mv.y)) & bisectors[..., None]) / (core.mv.z & bisectors)[..., None]).to_array()


def marker_squares(directions: core.Direction, reach: float, turn: float, camera: Camera) -> core.Direction:
    """Rotated offsets around projected directions form four-cornered marker polygons (`[..., 4] Direction-`)."""
    angles = turn + np.pi / 2 * np.arange(4)                                  # [4]
    rotors = (core.mv.xy * (-angles / 2)).exp()                               # [4] Rotor+
    square_offsets = rotors >> core.mv.x.dual()                               # [4] Direction-
    bisectors = stereographic_bisectors(directions)                           # [...] Direction-
    places = bisectors / (core.mv.z & bisectors)                              # [...] Direction-
    return places[:, None] + square_offsets * (reach * camera.pixel)          # [..., 4] Direction-


def disk_camera(ax: plt.Axes) -> Camera:
    """Central projection from the origin pairs directions with `x`, `y`, and weight plane `z` across `[001]`."""
    return panel(ax, core.mv.x, core.mv.y, core.mv.z)


def window_directions(window: tuple[float, float], n_pixels: int) -> core.Direction:
    """Stereographic reflection lifts pixel centres of `window` to pull directions (`[n_pixels, n_pixels] Direction-`)."""
    edges = np.linspace(*window, n_pixels + 1)                                # [n_pixels + 1]
    across = (edges[:-1] + edges[1:]) / 2                                     # [n_pixels]
    offsets = core.mv.x * across + core.mv.y * across[::-1, None]             # [n_pixels, n_pixels] Plane+
    mirrors = (core.mv.z + offsets).normalized()                              # [n_pixels, n_pixels] Plane+
    return mirrors >> core.mv.z.dual()                                        # [n_pixels, n_pixels] Direction-


def system_colours(n_systems: int) -> np.ndarray:
    """Tabulated colormap entries assign RGB colours (`[n_systems, 3]`) to the slip systems."""
    return plt.get_cmap(system_cmap)(np.arange(n_systems) % 20)[:, :3]        # [n_systems, 3]


def region_shades(factors: core.Scalar) -> np.ndarray:
    """Leading slip-system hues (`[..., 3]`) lighten toward white where the largest Schmid factor falls toward `faint_factor`."""
    values = factors.abs().to_array().reshape(*factors.shape[:2], -1)         # [..., 12]
    hue = system_colours(values.shape[-1])[values.argmax(axis=-1)]            # [..., 3]
    strength = np.clip((values.max(axis=-1) - faint_factor) / (full_factor - faint_factor), 0, 1)[..., None]
    return 1 - strength * (1 - hue)                                           # [..., 3]


def region_image(ax: plt.Axes, factors: Callable[[core.Direction], core.Scalar], window: tuple[float, float]):
    """Pixel-aligned region shading colours `window` by the leading slip system and masks outside the unit disk in white."""
    n_pixels = min(room(ax))
    directions = window_directions(window, n_pixels)                          # [n_pixels, n_pixels] Direction-
    # Reflection sends pixels inside the unit disk into the lower hemisphere and pixels outside the rim into the upper hemisphere.
    beyond = ((core.mv.z & directions) > 0)[..., None]                        # [n_pixels, n_pixels, 1]
    return pinned(ax, np.where(beyond, 1.0, region_shades(factors(directions))), (*window, *window))


def standard_triangle(camera: Camera) -> np.ndarray:
    """Rotor interpolation along the loop edges renders the `[001]`, `[011]`, `[111]` standard triangle onto `camera`."""
    shares = np.linspace(0, 1, n_edge_samples, endpoint=False)                # [n_edge_samples]
    tails, heads = Topology.loops([3]).incidence[1, 0] * triangle_corners     # [] Direction-[3], Direction-[3]
    turns = ((heads * tails.inverse()).log() * (shares / 2)).exp()            # [n_edge_samples] Rotor+[3]
    points = (turns >> tails).batch()                                         # [n_edge_samples, 3] Direction-
    edge_points = points[np.arange(n_edge_samples), np.arange(3)[:, None]]    # [3, n_edge_samples] Direction-
    arcs = stereographic_bisectors(edge_points).reshape(-1)                   # [3 * n_edge_samples] Direction-
    return simplices.strokes(arcs, Topology.loops([3 * n_edge_samples]), to_rgb(edge_colour),
                             charts.LINE_PX, camera)


def draw_disk(ax: plt.Axes, factors: Callable[[core.Direction], core.Scalar], planes: core.Plane,
              slips: core.Direction) -> None:
    """Draws the full stereographic hemisphere with slip-system regions, plane-normal squares, slip-direction dots, and standard triangle."""
    ax.set(xlim=disk_window, ylim=disk_window)
    region_image(ax, factors, disk_window)
    camera = disk_camera(ax)
    angles = np.linspace(0, 2 * np.pi, n_rim_samples, endpoint=False)         # [n_rim_samples]
    rim_rotors = (core.mv.xy * (-angles / 2)).exp()                           # [n_rim_samples] Rotor+
    rim = stereographic_bisectors(rim_rotors >> core.mv.x.dual())             # [n_rim_samples] Direction-
    normals = marker_squares(planes.dual(), normal_reach, np.pi / 4, camera)  # [4, 4] Direction-
    overlay(ax, [
        simplices.strokes(rim, Topology.loops([n_rim_samples]), to_rgb(edge_colour), charts.LINE_PX, camera),
        simplices.strokes(normals.reshape(-1), Topology.loops([4] * 4),
                          to_rgb(normal_colour), charts.LINE_PX, camera),
        simplices.markers(stereographic_bisectors(slips.reshape(-1)), to_rgb(normal_colour), charts.MARKER_PX / 2, camera),
        standard_triangle(camera),
    ])


def draw_close(ax: plt.Axes, factors: Callable[[core.Direction], core.Scalar]) -> list[np.ndarray]:
    """Draws the enlarged standard-triangle background and corner labels, returning triangle and goal-diamond layers."""
    ax.set(xlim=triangle_window, ylim=triangle_window)
    region_image(ax, factors, triangle_window)
    for corner, name, lift in zip(disk_coordinates(triangle_corners.batch()), corner_names, corner_lifts):
        ax.annotate(name, corner, xytext=(0, lift), textcoords="offset points", ha="center", va="center",
                    color=edge_colour)
    for goal, name in zip(disk_coordinates(goal_directions), goal_names):
        ax.annotate(name, goal, xytext=(0, 9), textcoords="offset points", ha="center", va="center",
                    color=edge_colour)
    camera = disk_camera(ax)
    diamonds = marker_squares(goal_directions, goal_reach, 0.0, camera)       # [2, 4] Direction-
    return [
        standard_triangle(camera),
        simplices.strokes(diamonds.reshape(-1), Topology.loops([4] * 2), to_rgb(edge_colour), charts.LINE_PX, camera),
    ]


def trail_layers(camera: Camera, axes: core.Direction, amounts: core.Scalar) -> list[np.ndarray]:
    """Blends active slip-system colours along each trajectory chain and marks each trajectory's current endpoint."""
    n_points, n_starts = axes.shape
    weights = amounts.abs().to_array().reshape(n_points - 1, n_starts, 12)    # [n_points - 1, n_starts, 12]
    colour_mix = weights @ system_colours(12) / weights.sum(axis=-1, keepdims=True)   # [n_points - 1, n_starts, 3]
    tracks = stereographic_bisectors(
        axes[np.arange(n_points), np.arange(n_starts)[:, None]],
    )                                                                         # [n_starts, n_points] Direction-
    return [
        simplices.strokes(tracks.reshape(-1), Topology.chains([n_points] * n_starts),
                          np.moveaxis(colour_mix, 1, 0).reshape(-1, 3) * path_shade, charts.LINE_PX, camera),
        simplices.markers(tracks[:, -1], to_rgb(path_colour), charts.MARKER_PX / 2, camera),
    ]


@charts.style()
def draw_chart(factors: Callable[[core.Direction], core.Scalar], planes: core.Plane, slips: core.Direction,
               axes: core.Direction, amounts: core.Scalar) -> plt.Figure:
    """Draws the full stereographic disk beside the enlarged standard triangle with completed tensile-axis trajectories."""
    figure, (disk, close) = charts.subplots(760, 380, ncols=2, layout="constrained")
    disk.set_axis_off()
    close.set_axis_off()
    draw_disk(disk, factors, planes, slips)
    triangle_and_goals = draw_close(close, factors)
    overlay(close, triangle_and_goals[:1] + trail_layers(disk_camera(close), axes, amounts) + triangle_and_goals[1:])
    return figure


def animate_paths(
    factors: Callable[[core.Direction], core.Scalar],
    start_axes: core.Direction,
    n_steps: int,
    n_frames: int,
    states: Iterable[tuple[core.Direction, core.Scalar]],
) -> Iterator[np.ndarray]:
    """Yields `n_frames` RGB frames of tensile-axis trajectories growing across the standard triangle at `panel_pixels` width."""
    return animate_paths_sized(factors, start_axes, n_steps, n_frames, panel_pixels, states)


def animate_paths_sized(
    factors: Callable[[core.Direction], core.Scalar],
    start_axes: core.Direction,
    n_steps: int,
    n_frames: int,
    pixels: int,
    states: Iterable[tuple[core.Direction, core.Scalar]],
) -> Iterator[np.ndarray]:
    """Yields `n_frames` RGB frames (`[pixels, pixels, 3]`) streamed from `states` over a static standard-triangle background."""
    frame_steps = set(np.linspace(1, n_steps, n_frames).round().astype(int))
    with charts.style():
        figure, ax = charts.subplots(pixels, pixels, layout="constrained")
        ax.set_axis_off()
        triangle_and_goals = draw_close(ax, factors)
        figure.canvas.draw()
        figure.set_layout_engine("none")
    camera = disk_camera(ax)
    image = overlay(ax, triangle_and_goals)
    axis_history = [start_axes]
    amount_history = []
    try:
        for step, (axis, step_amounts) in enumerate(states, start=1):
            axis_history.append(axis)
            amount_history.append(step_amounts)
            if step in frame_steps:
                image.set_data(drawing.composited(
                    triangle_and_goals[:1]
                    + trail_layers(camera, stack(axis_history), stack(amount_history))
                    + triangle_and_goals[1:]
                ))
                yield capture(figure)
    finally:
        plt.close(figure)
