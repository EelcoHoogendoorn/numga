"""Finite-footprint irradiance and the folded ray map on the receiving plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from contourpy import contour_generator
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, map_coordinates

from numga import stack
from examples.optics.specular_caustics import core

DPI = 100
COLOURS = LinearSegmentedColormap.from_list(
    "caustics", ["#03101e", "#07334b", "#176b81", "#71bbbf", "#fff4d8"], N=1024,
)


# --- plumbing -------------------------------------------------------------------------
def _coordinates(points: core.Direction, planes: tuple[str, ...]) -> np.ndarray:
    """Positions read by pairing with the planes across the named axes."""
    mv = points.context.multivector
    return np.asarray((stack(tuple(getattr(mv, plane) for plane in planes)) & points[..., None]).to_array())


def irradiance(
    landings: core.Horizontal, halfwidth: float, ray_area: float, n_pixels: int, blur_pixels: float,
) -> np.ndarray:
    """Power per receiver area for equal-power rays, with a Gaussian pixel footprint."""
    coordinates = _coordinates(landings, ("x", "y"))
    pixel_width = 2 * halfwidth / n_pixels
    pixel_area = pixel_width ** 2
    n_padding = int(np.ceil(4 * blur_pixels))
    window_halfwidth = halfwidth + n_padding * pixel_width
    n_bins = n_pixels + 2 * n_padding
    counts, _, _ = np.histogram2d(
        coordinates[..., 1].ravel(), coordinates[..., 0].ravel(), bins=n_bins,
        range=((-window_halfwidth, window_halfwidth),) * 2,
    )
    # Trace beyond the visible window so a footprint crossing its edge still contributes.
    density = gaussian_filter(counts * (ray_area / pixel_area), blur_pixels, mode="constant")
    return density[n_padding:n_padding + n_pixels, n_padding:n_padding + n_pixels]


def pixels(density: np.ndarray, exposure: float) -> np.ndarray:
    """Compress bright folds with one exposure shared by the whole sequence."""
    brightness = -np.expm1(-exposure * density)
    return COLOURS(brightness, bytes=True)[..., :3]


def draw(
    landings: core.Horizontal, halfwidth: float, ray_area: float,
    n_pixels: int, blur_pixels: float, exposure: float,
) -> plt.Figure:
    """Irradiance on a square receiver, from dark water blue to warm bright folds."""
    density = irradiance(landings, halfwidth, ray_area, n_pixels, blur_pixels)
    figure = plt.figure(figsize=(n_pixels / DPI,) * 2, dpi=DPI, facecolor=COLOURS(0))
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
    coordinates = _coordinates(chart_landings, ("x", "y"))
    areas = np.asarray(area.to_array())
    source_curves = contour_generator(z=areas).lines(0)
    # Contours live between source samples; interpolate their receiver positions there.
    receiver_curves = [
        np.stack([map_coordinates(coordinates[..., axis], curve[:, ::-1].T, order=1, mode="nearest")
                  for axis in range(2)], axis=-1)
        for curve in source_curves
    ]
    figure.axes[0].add_collection(LineCollection(receiver_curves, colors="#ff907a", linewidths=0.65))
    return figure


def animate(
    landings: Iterable[core.Horizontal], halfwidth: float, ray_area: float,
    n_pixels: int, blur_pixels: float, exposure: float,
) -> Iterator[np.ndarray]:
    """Read and colour one ray map at a time, keeping the receiver and exposure fixed."""
    for frame in landings:
        density = irradiance(frame, halfwidth, ray_area, n_pixels, blur_pixels)
        yield np.ascontiguousarray(pixels(density, exposure)[::-1])


def draw_paths(
    surface: core.Direction, ray_surface: core.Direction, ray_landing: core.Direction,
    solution_surface: core.Direction, target: core.Direction, incident_height: float,
) -> plt.Figure:
    """A ray fan in the x-z plane, with the paths reaching one target highlighted."""
    curve = _coordinates(surface, ("x", "z"))
    crossings = _coordinates(ray_surface, ("x", "z"))
    landings = _coordinates(ray_landing, ("x", "z"))
    solutions = _coordinates(solution_surface, ("x", "z"))
    destination = _coordinates(target, ("x", "z"))
    incoming = np.column_stack((crossings[:, 0], np.full(len(crossings), incident_height)))
    solution_incoming = np.column_stack((solutions[:, 0], np.full(len(solutions), incident_height)))
    paths = np.stack((incoming, crossings, landings), axis=1)
    solved_paths = np.stack((solution_incoming, solutions, np.broadcast_to(destination, solutions.shape)), axis=1)
    lower = landings[:, 1].min()
    left, right = curve[:, 0].min(), curve[:, 0].max()
    vertical_span = incident_height - lower
    figure = plt.figure(figsize=(5.5 * (right - left) / vertical_span, 5.5), facecolor="white")
    axes = figure.add_axes((0, 0, 1, 1))
    axes.fill_between(curve[:, 0], lower, curve[:, 1], color="#e4f2f6")
    axes.add_collection(LineCollection(paths, colors="#7798a6", linewidths=0.65, alpha=0.35))
    axes.plot(curve[:, 0], curve[:, 1], color="#428698", linewidth=1.3)
    axes.add_collection(LineCollection(solved_paths, colors="#df8a35", linewidths=1.65))
    axes.plot((left, right), (lower, lower), color="#578390", linewidth=1.2)
    axes.scatter(destination[0], destination[1], s=20, color="#b25422", zorder=3, clip_on=False)
    axes.set(xlim=(left, right), ylim=(lower, incident_height), aspect="equal")
    axes.set_axis_off()
    return figure
