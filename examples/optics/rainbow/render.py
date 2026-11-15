"""Renderings of the primary and secondary rainbows pair a swept-ray cross-section of one raindrop with
the double bow seen across the sky opposite the sun."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from numga.render import contours, drawing, simplices
from numga.topology import Topology
from examples.optics.rainbow import core
from examples.util import charts, shapes
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay, pinned, room

mv = core.mv
sunlight = core.forward                                                       # [] Direction-
# Palette for the background, drop, incident sunlight, primary and secondary paths, sky, ground, and axes.
background = "#0b1020"
drop_colour = "#1e3a5f"
sunlight_colour = "#fde68a"
primary_colour = "#fb923c"
secondary_colour = "#60a5fa"
sky_colour = np.array([0.04, 0.06, 0.12])
ground_colour = "#14532d"
axis_colour, label_colour = "#475569", "#94a3b8"
# Exterior ray length in drop radii, angular histogram bin width and sky half-width in degrees, and tone exponent.
ray_length = 2.4
bin_width = 0.05
view_half_width = 62.0
tone_exponent = 0.65
# Figure dimensions, stroke widths, bundle blend fraction, and drop rim sector count.
sky_width, sweep_size = 6.0, (600, 270)
light_width, bundle_width, curve_width = 1.5, 1.0, charts.points(1.0)
bundle_share = 0.4
drop_sectors = 96


# --- plumbing -------------------------------------------------------------------------
def pixel_grid(
    view: tuple[float, float, float, float], n_columns_max: int, n_rows_max: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Square-pixel centres inside `view` fitting within `n_columns_max` by `n_rows_max` give the
    horizontal and vertical coordinate grids and outer pixel extent."""
    left, right, bottom, top = view
    pixel = max((right - left) / n_columns_max, (top - bottom) / n_rows_max)
    n_columns, n_rows = round((right - left) / pixel), round((top - bottom) / pixel)
    middle = np.array([left + right, bottom + top]) / 2
    half = np.array([n_columns, n_rows]) * pixel / 2
    across = middle[0] - half[0] + pixel * (np.arange(n_columns) + 0.5)
    up = middle[1] + half[1] - pixel * (np.arange(n_rows) + 0.5)
    return across, up, (middle[0] - half[0], middle[0] + half[0], middle[1] - half[1], middle[1] + half[1])


def viewing_degrees(sight_cosines: core.Scalar) -> np.ndarray:
    """Antisolar viewing cosines (`[...] Scalar+`) convert to viewing angles in degrees (`[...]`)."""
    return np.degrees(np.arccos(np.clip(sight_cosines.to_array(), -1, 1)))


def spectral_colours(wavelengths: np.ndarray) -> np.ndarray:
    """Piecewise linear RGB approximations at `wavelengths` (`[n_colours]`) normalize so equal energy
    across all wavelengths sums to white (`[n_colours, 3]`)."""
    red = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0.4, 0, 0, 0, 1, 1, 0.5])
    green = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0, 0, 1, 1, 1, 0, 0])
    blue = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0.6, 1, 1, 0, 0, 0, 0])
    colours = np.stack([red, green, blue], axis=-1)
    return colours / colours.sum(axis=0)


def sky_brightness(
    sight_cosines: core.Scalar, flux: core.Scalar, sun_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogramming `sight_cosines` weighted by `flux` (`[n_reflections, n_colours, n_rays] Scalar+` each)
    and convolving across the solar disc of width `sun_width` gives ring angles and radiance per solid angle."""
    edges = np.arange(0, 90 + bin_width, bin_width)
    angles = (edges[1:] + edges[:-1]) / 2
    ray_angles, weights = viewing_degrees(sight_cosines), flux.to_array()
    ring_light = np.stack([
        np.histogram(ray_angles[:, colour].ravel(), edges, weights=weights[:, colour].ravel())[0]
        for colour in range(ray_angles.shape[1])
    ])
    # Convolving across the solar disc accounts for rays arriving across the sun's angular diameter.
    across = np.arange(-sun_width / 2, sun_width / 2 + bin_width, bin_width)
    disc = np.sqrt(np.clip(1 - (2 * across / sun_width) ** 2, 0, None))
    disc_light = np.stack([np.convolve(ring, disc / disc.sum(), mode="same") for ring in ring_light])
    return angles, disc_light / np.sin(np.radians(angles))


def sky_figure(
    sight_cosines: core.Scalar,
    flux: core.Scalar,
    wavelengths: np.ndarray,
    sun_elevation: float,
    sun_width: float,
) -> plt.Figure:
    """Renders the antisolar sky with the antisolar point `sun_elevation` degrees below the horizon
    and solar disc blur `sun_width`."""
    angles, light = sky_brightness(sight_cosines, flux, sun_width)
    colour = light.T @ spectral_colours(wavelengths)
    colour = colour / colour.max()
    view = (-view_half_width, view_half_width, sun_elevation - 6, sun_elevation + view_half_width * 0.8)
    figure = plt.figure(
        figsize=(sky_width, sky_width * (view[3] - view[2]) / (view[1] - view[0])),
        dpi=100,
        facecolor=background,
    )
    ax = figure.add_axes((0, 0, 1, 1))
    # Angular distance from the antisolar point matches Euclidean radius on the chart around the origin.
    across, height, extent = pixel_grid(view, *room(ax))
    n_rows, n_columns = len(height), len(across)
    chart = core.forward * across[None, :] + core.up * height[:, None]        # [n_rows, n_columns] Direction-
    distance = chart.antinorm().to_array()
    image = np.stack([np.interp(distance, angles, colour[:, channel], right=0) for channel in range(3)], axis=-1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    camera = offset_panel(ax, mv.x, mv.y)._replace(pixels=n_columns)
    ground = (
        core.forward * np.array(extent[:2])[None, :]
        + core.up * np.array([extent[2], sun_elevation])[:, None]
    ).reshape(-1)                                                             # [4] Direction-
    grid = Topology.grid(2, 2)
    V, E, F = grid.n_elements
    seen = drawing.composited([
        contours.picture(ground, grid, np.zeros(V), to_rgb(ground_colour), camera),
    ])
    sky = np.clip(sky_colour + image**tone_exponent, 0, 1)
    pinned(ax, sky * (1 - seen[..., 3:]) + seen[..., :3] * seen[..., 3:], extent)
    return figure


def sweep_animation(
    bundle: tuple[core.Direction, core.Direction, core.Direction],
    offsets: core.Direction,
    sight_cosines: core.Scalar,
    wavelengths: np.ndarray,
    states: Iterable[tuple[core.Direction, core.Direction, core.Direction, core.Scalar]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames sweeping one ray across the raindrop alongside its primary and secondary
    viewing-angle curves."""
    def ray_polylines(
        entered: core.Direction, vertices: core.Direction, exits: core.Direction,
    ) -> list[core.Direction]:
        """Polylines from incoming sunlight through `order` internal bounces and out past the drop."""
        sunward = entered - sunlight * ray_length                             # [...] Direction-
        ahead = vertices + exits * ray_length                                 # [..., ...] Direction-
        return [
            concatenate([stack([sunward, entered]), vertices[:order + 1], ahead[order][None]])
            for order in (1, 2)
        ]

    def exit_strokes(paths: core.Direction) -> tuple[core.Direction, Topology]:
        """Final exit segments of `paths` (`[n_points, n_rays] Direction-`) pair endpoints with edges."""
        n_rays = paths.shape[1]
        ends = concatenate([paths[-2], paths[-1]])                            # [2 * n_rays] Direction-
        edges = np.stack([np.arange(n_rays), n_rays + np.arange(n_rays)], axis=-1)
        return ends, Topology(2 * n_rays, edges)

    bundle_rays = ray_polylines(*bundle)
    heights = (mv.y & offsets).to_array()
    raw_colours = spectral_colours(wavelengths)
    curves = viewing_degrees(sight_cosines)
    colours = raw_colours / raw_colours.max(axis=-1, keepdims=True)
    with charts.style():
        figure = plt.figure(
            figsize=(sweep_size[0] / charts.DPI, sweep_size[1] / charts.DPI),
            dpi=charts.DPI,
            facecolor=background,
        )
        try:
            drop = figure.add_axes((0, 0, 0.5, 1))
            drop.set_facecolor(background)
            drop.set_xlim(-ray_length, ray_length)
            drop.set_ylim(-ray_length * 0.9, ray_length * 0.9)
            drop.set_aspect("equal")
            drop.set_axis_off()
            camera = offset_panel(drop, mv.x, mv.y)
            disk, coordinates = shapes.disk(1.0, 1, drop_sectors)
            V, E, F = disk.n_elements
            places = mv(core.Plane, coordinates).dual()                       # [V] Direction-
            still = [contours.picture(places, disk, np.zeros(V), to_rgb(drop_colour), camera)]
            # Blending the static bundle into the background leaves palette slots free for spectral curves.
            still += [
                simplices.strokes(
                    *exit_strokes(paths),
                    bundle_share * np.array(to_rgb(colour)) + (1 - bundle_share) * np.array(to_rgb(background)),
                    bundle_width,
                    camera,
                )
                for paths, colour in zip(bundle_rays, (primary_colour, secondary_colour))
            ]
            drop_seen = overlay(drop, still)
            angles = figure.add_axes((0.57, 0.1, 0.4, 0.85))
            angles.set_facecolor(background)
            markers = []
            for order, mark in ((0, primary_colour), (1, secondary_colour)):
                for curve, colour in zip(curves[order], colours):
                    angles.plot(heights, curve, color=colour, linewidth=curve_width)
                markers += angles.plot([], [], "o", color=mark)
            angles.set_xlim(0, 1)
            angles.set_ylim(0, 90)
            angles.set_xticks([0, 1])
            angles.set_yticks([0, round(curves[0].max()), round(curves[1].min()), 90])
            angles.tick_params(colors=label_colour)
            for side in ("top", "right"):
                angles.spines[side].set_visible(False)
            for spine in angles.spines.values():
                spine.set_color(axis_colour)
            for entered, vertices, exits, state_cosines in states:
                swept_paths = ray_polylines(entered, vertices, exits)
                lights = [
                    (swept_paths[1][1:], secondary_colour),
                    (swept_paths[0][1:], primary_colour),
                    (swept_paths[0][:2], sunlight_colour),
                ]
                drop_seen.set_data(drawing.composited(still + [
                    simplices.strokes(path, Topology.chains([len(path)]), to_rgb(colour), light_width, camera)
                    for path, colour in lights
                ]))
                swept_height = float((mv.y & entered).to_array())
                marks = viewing_degrees(state_cosines)
                for order, marker in enumerate(markers):
                    marker.set_data([swept_height], [marks[order + 1]])
                yield capture(figure)
        finally:
            plt.close(figure)
