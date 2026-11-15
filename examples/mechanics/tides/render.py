"""Two-panel frames display the star cluster over the Plummer density beside a zoomed view of the predicted tidal rim."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, to_rgb, to_rgba

from numga import stack
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from examples.mechanics.tides import core
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay, pinned

star_colour = to_rgb("#0f172a")
prediction_colour = "#ea580c"
zoom_factor = 1.6
n_rim_pixels = 160
# Width of the predicted rim contour in pixels.
rim_width = 1.6
# Star marker radii in pixels for the full field and close-up panel.
wide_star_radius, near_star_radius = 1.0, 1.5
# Faintest density drawn as a fraction of the peak.
density_floor = 1e-4


def density_picture(ax: Axes, points: core.Point, mass_density: core.Scalar) -> np.ndarray:
    """Logarithmic blue shading maps `mass_density` across the cell grid `points` into an RGBA image."""
    corners = points[[0, -1], [0, -1]]                                       # [2] Point-
    values = mass_density.to_array()
    colours = plt.get_cmap("Blues")(LogNorm(values.max() * density_floor, values.max())(values))[..., :3]
    ax.set(xlim=(core.mv.x & corners).to_array(), ylim=(core.mv.y & corners).to_array(), xticks=[], yticks=[])
    ax.set_aspect("equal")
    places = points.reshape(-1)                                              # [...] Point-
    return contours.picture(
        places,
        Topology.grid(*values.shape),
        values.reshape(-1),
        colours.reshape(-1, 3),
        offset_panel(ax, core.mv.x, core.mv.y),
    )


def rim_picture(
    centre: core.Point,
    predicted_spread: core.Spread,
    half_size: core.Scalar,
    grid: Topology,
    view: Camera,
) -> np.ndarray:
    """Implicit contouring draws the unit level curve `predicted_spread.solve(offsets) & offsets == 1` around `centre`."""
    samples = half_size * np.linspace(-1, 1, n_rim_pixels)                   # [n_rim_pixels] Scalar+
    offsets = (core.mv.x * samples[None, :] + core.mv.y * samples[:, None]).dual()  # [n_rim_pixels, n_rim_pixels] Point-
    levels = (predicted_spread.solve(offsets) & offsets).to_array()
    places = (centre + offsets).reshape(-1)                                  # [n_rim_pixels * n_rim_pixels] Point-
    prediction_level = contours.Level(1.0, np.array(to_rgba(prediction_colour)), rim_width)
    return contours.picture(places, grid, levels.reshape(-1), np.ones(3), view, levels=(prediction_level,))


def flyby_frame(
    points: core.Point,
    mass_density: core.Scalar,
    positions: core.Point,
    centre: core.Point,
    predicted_spread: core.Spread,
) -> np.ndarray:
    """Two-panel snapshot pairs the full Plummer density view with the zoomed tidal rim around `centre`."""
    (rendered_frame,) = flyby_frames(points, mass_density, [(positions, centre, predicted_spread)])
    return rendered_frame


def flyby_frames(
    points: core.Point,
    mass_density: core.Scalar,
    states: Iterable[tuple[core.Point, core.Point, core.Spread]],
) -> Iterator[np.ndarray]:
    """Yields two-panel RGB frames showing the star cluster over `mass_density` beside the zoomed predicted rim."""
    figure = plt.figure(figsize=(11, 4.4), dpi=80, layout="constrained")
    whole_axes, near_axes = figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.5, 1)})
    # Both panels start bare of ticks before either image is fitted to its layout box.
    near_axes.set(xlim=(-1, 1), ylim=(-1, 1), xticks=[], yticks=[])
    near_axes.set_aspect("equal")
    background = density_picture(whole_axes, points, mass_density)
    whole_camera = offset_panel(whole_axes, core.mv.x, core.mv.y)
    star_overlay = overlay(whole_axes, [background])
    whole_axes.set_axis_on()
    rim_overlay = pinned(near_axes, np.zeros(offset_panel(near_axes, core.mv.x, core.mv.y).shape + (4,)), (-1, 1, -1, 1))
    near_axes.set_axis_on()
    rim_grid = Topology.grid(n_rim_pixels, n_rim_pixels)
    planes = stack((core.mv.x, core.mv.y))                                   # [2] Plane+
    for positions, centre, predicted_spread in states:
        star_overlay.set_data(
            drawing.composited([background, simplices.markers(positions, star_colour, wide_star_radius, whole_camera)])
        )
        # Frame the close-up around the cluster's centre of mass using the widest horizontal or vertical star offset.
        spans = (planes & (positions - centre)[..., None]).abs().reshape(-1)
        half_size = spans.max() * zoom_factor                                # [] Scalar+
        middle = planes & centre                                             # [2] Scalar+
        window = stack([middle - half_size, middle + half_size], axis=-1).to_array().reshape(-1)
        rim_overlay.set_extent(window)
        near_axes.set(xlim=window[:2], ylim=window[2:])
        near_camera = offset_panel(near_axes, core.mv.x, core.mv.y)
        rim_overlay.set_data(
            drawing.composited([
                rim_picture(centre, predicted_spread, half_size, rim_grid, near_camera),
                simplices.markers(positions, star_colour, near_star_radius, near_camera),
            ])
        )
        yield capture(figure)
    plt.close(figure)
