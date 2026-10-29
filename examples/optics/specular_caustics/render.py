"""Finite-footprint irradiance and the folded ray map on the receiving plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from contourpy import contour_generator
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from numga import stack
from examples.optics.specular_caustics import core

dpi = 100
colours = LinearSegmentedColormap.from_list(
    "caustics", ["#03101e", "#07334b", "#176b81", "#71bbbf", "#fff4d8"], N=1024,
)


# The planes that read places on the receiver, across x and y, and on a vertical section, across x and up z.
receiver_planes = stack((core.mv.x, core.mv.y))                             # [2] Plane+
section_planes = stack((core.mv.x, core.mv.z))                              # [2] Plane+
# The four samples of a chart around a place between them, as steps along its rows and columns.
cell_corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])                  # [corners, 2]


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Direction, planes: core.Plane) -> np.ndarray:
    """Positions read by pairing with the planes across which they are drawn."""
    return np.asarray((planes & points[..., None]).to_array())


def between_samples(chart_landings: core.Horizontal, places: np.ndarray) -> core.Horizontal:
    """The landings at places between the samples of the chart over the water, `places` `[vertices, 2]` in fractional
    rows and columns: the landings of the four samples around each place, each weighted by its bilinear share."""
    cells = np.clip(np.floor(places).astype(int), 0, np.array(chart_landings.shape) - 2)   # [vertices, 2]
    fractions = places - cells                                               # [vertices, 2]
    samples = cells[None] + cell_corners[:, None]                            # [corners, vertices, 2]
    # A sample's share falls off linearly along the rows and along the columns from one at its place.
    shares = np.where(cell_corners[:, None], fractions[None], 1 - fractions[None]).prod(axis=-1)   # [corners, vertices]
    return (chart_landings[samples[..., 0], samples[..., 1]] * shares).sum(axis=0)   # [vertices] Horizontal-


def irradiance(
    landings: core.Horizontal, halfwidth: float, ray_area: float, n_pixels: int, blur_pixels: float,
) -> np.ndarray:
    """Power per receiver area for equal-power rays, with a Gaussian pixel footprint."""
    places = coordinates(landings, receiver_planes)
    pixel_width = 2 * halfwidth / n_pixels
    pixel_area = pixel_width ** 2
    n_padding = int(np.ceil(4 * blur_pixels))
    window_halfwidth = halfwidth + n_padding * pixel_width
    n_bins = n_pixels + 2 * n_padding
    counts, _, _ = np.histogram2d(
        places[..., 1].ravel(), places[..., 0].ravel(), bins=n_bins,
        range=((-window_halfwidth, window_halfwidth),) * 2,
    )
    # Trace beyond the visible window so a footprint crossing its edge still contributes.
    density = gaussian_filter(counts * (ray_area / pixel_area), blur_pixels, mode="constant")
    return density[n_padding:n_padding + n_pixels, n_padding:n_padding + n_pixels]


def pixels(density: np.ndarray, exposure: float) -> np.ndarray:
    """The colours of a density, its bright folds compressed by one exposure shared by the whole sequence."""
    brightness = -np.expm1(-exposure * density)
    return colours(brightness, bytes=True)[..., :3]


def draw(
    landings: core.Horizontal, halfwidth: float, ray_area: float,
    n_pixels: int, blur_pixels: float, exposure: float,
) -> plt.Figure:
    """Irradiance on a square receiver, from dark water blue to warm bright folds."""
    density = irradiance(landings, halfwidth, ray_area, n_pixels, blur_pixels)
    figure = plt.figure(figsize=(n_pixels / dpi,) * 2, dpi=dpi, facecolor=colours(0))
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(pixels(density, exposure), origin="lower", interpolation="nearest",
                extent=(-halfwidth, halfwidth, -halfwidth, halfwidth))
    axes.set_axis_off()
    return figure


def draw_critical(
    landings: core.Horizontal, chart_landings: core.Horizontal, area: core.Scalar,
    halfwidth: float, ray_area: float, n_pixels: int, blur_pixels: float, exposure: float,
) -> plt.Figure:
    """The zero-area curves carried through the ray map, over the receiver's irradiance."""
    figure = draw(landings, halfwidth, ray_area, n_pixels, blur_pixels, exposure)
    areas = np.asarray(area.to_array())
    # The zero-area curves over the water, in its chart's columns and rows, between its samples.
    source_curves = contour_generator(z=areas).lines(0)
    receiver_curves = [coordinates(between_samples(chart_landings, curve[:, ::-1]), receiver_planes)
                       for curve in source_curves]
    figure.axes[0].add_collection(LineCollection(receiver_curves, colors="#ff907a", linewidths=0.45))
    return figure


def animate(
    landings: Iterable[core.Horizontal], halfwidth: float, ray_area: float,
    n_pixels: int, blur_pixels: float, exposure: float,
) -> Iterator[np.ndarray]:
    """Yields the coloured irradiance of one ray map at a time, on a fixed receiver at a fixed exposure."""
    for frame in landings:
        density = irradiance(frame, halfwidth, ray_area, n_pixels, blur_pixels)
        yield np.ascontiguousarray(pixels(density, exposure)[::-1])


def draw_paths(
    surface: core.Direction, ray_surface: core.Direction, ray_landing: core.Direction,
    solution_surface: core.Direction, target: core.Direction, incident_height: float,
) -> plt.Figure:
    """A ray fan in the x-z plane, with the paths reaching one target highlighted."""
    # Each ray comes straight down from the incident height to where it crosses the surface, and on to the receiver.
    incoming = ray_surface + core.up * (incident_height - (core.mv.z & ray_surface))              # [rays] Direction-
    solution_incoming = solution_surface + core.up * (incident_height - (core.mv.z & solution_surface))   # [paths] Direction-
    destinations = target.broadcast_to(solution_surface.shape)                                    # [paths] Direction-
    fan_paths = stack((incoming, ray_surface, ray_landing), axis=1)                               # [rays, stages] Direction-
    solved_paths = stack((solution_incoming, solution_surface, destinations), axis=1)             # [paths, stages] Direction-
    curve, paths, solved, landings, destination = (coordinates(points, section_planes)
                                                   for points in (surface, fan_paths, solved_paths, ray_landing, target))
    lower = landings[:, 1].min()
    left, right = curve[:, 0].min(), curve[:, 0].max()
    vertical_span = incident_height - lower
    figure = plt.figure(figsize=(3 * (right - left) / vertical_span, 3), dpi=dpi, facecolor="white")
    axes = figure.add_axes((0, 0, 1, 1))
    axes.fill_between(curve[:, 0], lower, curve[:, 1], color="#e4f2f6")
    axes.add_collection(LineCollection(paths, colors="#7798a6", linewidths=0.35, alpha=0.35))
    axes.plot(curve[:, 0], curve[:, 1], color="#428698", linewidth=0.7)
    axes.add_collection(LineCollection(solved, colors="#df8a35", linewidths=0.9))
    axes.plot((left, right), (lower, lower), color="#578390", linewidth=0.65)
    axes.scatter(destination[0], destination[1], s=6, color="#b25422", zorder=3, clip_on=False)
    axes.set(xlim=(left, right), ylim=(lower, incident_height), aspect="equal")
    axes.set_axis_off()
    return figure
