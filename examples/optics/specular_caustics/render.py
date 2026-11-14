"""Finite-footprint receiver irradiance maps, caustic fold contours, and vertical ray-fan cross-sections."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from contourpy import contour_generator
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.ndimage import gaussian_filter

from numga import concatenate, stack
from numga.topology import Topology
from numga.render import contours, simplices
from examples.util.panels import offset_panel, overlay
from examples.optics.specular_caustics import core

dpi = 100
colours = LinearSegmentedColormap.from_list(
    "caustics", ["#03101e", "#07334b", "#176b81", "#71bbbf", "#fff4d8"], N=1024,
)

# Coordinate readout planes spanning the horizontal receiver.
receiver_planes = stack((core.mv.x, core.mv.y))                                   # [2] Plane+
# Corner index offsets of a unit grid square for bilinear chart interpolation.
quad_offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])                         # [4, 2]
# Fold-curve stroke style and vertical cross-section palette and stroke widths in pixels.
fold_colour, fold_width = "#ff907a", 1.0
water_colour, fan_colour, surface_colour, solved_colour, receiver_colour, target_colour = (
    "#e4f2f6", "#7798a6", "#428698", "#df8a35", "#578390", "#b25422")
fan_width, surface_width, solved_width, target_radius = 1.0, 1.0, 1.5, 2.0
# Fractional vertical margin below the receiver plane in the side-view cross-section.
section_room = 0.03


# --- plumbing -------------------------------------------------------------------------
def plane_coordinates(points: core.Direction, planes: core.Plane) -> np.ndarray:
    """Pairing `points` (`[...] Direction-`) against `planes` (`[2] Plane+`) extracts `[..., 2]` coordinates."""
    return np.asarray((planes & points[..., None]).to_array())


def interpolated_landings(chart_landings: core.Horizontal, grid_places: np.ndarray) -> core.Horizontal:
    """Bilinearly weighting the four surrounding chart samples of `chart_landings` at fractional row-column
    `grid_places` (`[n_points, 2]`) maps source contour vertices (`[n_points] Horizontal-`) to the receiver."""
    base_indices = np.clip(np.floor(grid_places).astype(int), 0, np.array(chart_landings.shape) - 2)  # [n_points, 2]
    fractions = grid_places - base_indices                                        # [n_points, 2]
    corner_indices = base_indices[None] + quad_offsets[:, None]                   # [4, n_points, 2]
    weights = np.where(quad_offsets[:, None], fractions[None], 1 - fractions[None]).prod(axis=-1)  # [4, n_points]
    return (chart_landings[corner_indices[..., 0], corner_indices[..., 1]] * weights).sum(axis=0)  # [n_points] Horizontal-


def receiver_irradiance(
    landings: core.Horizontal, halfwidth: float, ray_area: float, n_pixels: int, blur_pixels: float,
) -> np.ndarray:
    """Binning equal-power `landings` onto `[n_pixels, n_pixels]` with Gaussian `blur_pixels` estimates receiver irradiance."""
    places = plane_coordinates(landings, receiver_planes)
    pixel_width = 2 * halfwidth / n_pixels
    pixel_area = pixel_width ** 2
    n_padding = int(np.ceil(4 * blur_pixels))
    window_halfwidth = halfwidth + n_padding * pixel_width
    n_bins = n_pixels + 2 * n_padding
    counts, _, _ = np.histogram2d(
        places[..., 1].ravel(), places[..., 0].ravel(), bins=n_bins,
        range=((-window_halfwidth, window_halfwidth),) * 2,
    )
    # Pad beyond the viewport before smoothing so rays landing just outside still contribute across the border.
    density = gaussian_filter(counts * (ray_area / pixel_area), blur_pixels, mode="constant")
    return density[n_padding:n_padding + n_pixels, n_padding:n_padding + n_pixels]


def exposed_rgb(density: np.ndarray, exposure: float) -> np.ndarray:
    """Exponential tone compression maps `density` at `exposure` into `uint8` RGB colours."""
    brightness = -np.expm1(-exposure * density)
    return colours(brightness, bytes=True)[..., :3]


def irradiance_figure(
    landings: core.Horizontal, halfwidth: float, ray_area: float,
    n_pixels: int, blur_pixels: float, exposure: float,
) -> plt.Figure:
    """Tone-mapped irradiance of `landings` on the square receiver `[-halfwidth, halfwidth]^2`."""
    density = receiver_irradiance(landings, halfwidth, ray_area, n_pixels, blur_pixels)
    figure = plt.figure(figsize=(n_pixels / dpi,) * 2, dpi=dpi, facecolor=colours(0))
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(exposed_rgb(density, exposure), origin="lower", interpolation="nearest",
                extent=(-halfwidth, halfwidth, -halfwidth, halfwidth))
    axes.set_axis_off()
    return figure


def folds_figure(
    landings: core.Horizontal, chart_landings: core.Horizontal, jacobian_determinant: core.Scalar,
    halfwidth: float, ray_area: float, n_pixels: int, blur_pixels: float, exposure: float,
) -> plt.Figure:
    """Zero contours of `jacobian_determinant` mapped through `chart_landings` overlay coral caustic folds on the irradiance."""
    figure = irradiance_figure(landings, halfwidth, ray_area, n_pixels, blur_pixels, exposure)
    axes = figure.axes[0]
    determinants = np.asarray(jacobian_determinant.to_array())
    source_curves = contour_generator(z=determinants).lines(0)
    receiver_curves = concatenate([interpolated_landings(chart_landings, curve[:, ::-1]) for curve in source_curves])
    chains = Topology.chains([len(curve) for curve in source_curves])
    overlay(axes, [simplices.strokes(receiver_curves, chains, to_rgb(fold_colour), fold_width,
                                     offset_panel(axes, core.mv.x, core.mv.y))])
    return figure


def caustics_animation(
    halfwidth: float, ray_area: float, n_pixels: int, blur_pixels: float, exposure: float,
    landing_states: Iterable[core.Horizontal],
) -> Iterator[np.ndarray]:
    """Yields `[n_pixels, n_pixels, 3]` RGB irradiance frames for each ray-landing field in `landing_states`."""
    for landings in landing_states:
        density = receiver_irradiance(landings, halfwidth, ray_area, n_pixels, blur_pixels)
        yield np.ascontiguousarray(exposed_rgb(density, exposure)[::-1])


def paths_figure(
    surface_profile: core.Direction, fan_surface: core.Direction, fan_landing: core.Direction,
    solution_surface: core.Direction, target_point: core.Direction, incident_height: float,
) -> plt.Figure:
    """Vertical `xz` cross-section displaying a refracted ray fan and the Newton-solved paths reaching `target_point`."""
    incoming_fan = fan_surface + core.up * (incident_height - (core.mv.z & fan_surface))  # [n_fan] Direction-
    incoming_solutions = solution_surface + core.up * (incident_height - (core.mv.z & solution_surface))  # [n_paths] Direction-
    destinations = target_point.broadcast_to(solution_surface.shape)              # [n_paths] Direction-
    fan_paths = stack((incoming_fan, fan_surface, fan_landing), axis=1)           # [n_fan, 3] Direction-
    solved_paths = stack((incoming_solutions, solution_surface, destinations), axis=1)  # [n_paths, 3] Direction-

    lower = float((core.mv.z & fan_landing).min().to_array())
    horizontal_coords = core.mv.x & surface_profile                               # [n_section] Scalar+
    left, right = float(horizontal_coords.min().to_array()), float(horizontal_coords.max().to_array())
    bottom = lower - section_room * (incident_height - lower)
    vertical_span = incident_height - bottom
    figure = plt.figure(figsize=(3 * (right - left) / vertical_span, 3), dpi=dpi, facecolor="white")
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set(xlim=(left, right), ylim=(bottom, incident_height), aspect="equal")
    axes.set_axis_off()
    camera = offset_panel(axes, core.mv.x, core.mv.z)

    receiver_feet = surface_profile - core.up * ((core.mv.z & surface_profile) - lower)  # [n_section] Direction-
    water_strip = stack((receiver_feet, surface_profile)).reshape(-1)             # [2 * n_section] Direction-
    n_fan_rays, n_paths, n_stages = fan_paths.shape[0], solved_paths.shape[0], fan_paths.shape[1]
    overlay(axes, [
        contours.picture(water_strip, Topology.grid(2, len(surface_profile)), np.zeros(len(water_strip)),
                         to_rgb(water_colour), camera),
        simplices.strokes(fan_paths.reshape(-1), Topology.chains([n_stages] * n_fan_rays), to_rgb(fan_colour),
                          fan_width, camera, opacity=0.35),
        simplices.strokes(surface_profile, Topology.chains([len(surface_profile)]), to_rgb(surface_colour),
                          surface_width, camera),
        simplices.strokes(solved_paths.reshape(-1), Topology.chains([n_stages] * n_paths), to_rgb(solved_colour),
                          solved_width, camera),
        simplices.strokes(receiver_feet[[0, -1]], Topology.chains([2]), to_rgb(receiver_colour), surface_width, camera),
        simplices.markers(target_point[None], to_rgb(target_colour), target_radius, camera)])
    return figure
