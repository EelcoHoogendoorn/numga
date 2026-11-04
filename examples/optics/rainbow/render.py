"""Drawing for the rainbow: light swept across a raindrop beside the angles it leaves at, and the double
bow on the sky opposite the sun, seen from the ground."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Algebra, concatenate, stack
from examples.util import charts, shapes
from examples.util.animation import capture
from examples.util.wgsl import contours, pinned, room, simplices
from examples.util.wgsl.camera import offset_panel
from examples.optics.rainbow import core

mv = core.mv
# The direction sunlight travels, across the page.
sunlight = mv.x.dual()                                                        # [] Direction-
# The planes of x and of y, upright to the page, whose duals are the directions along the sunlight and up the page.
Upright = core.ga.gatype.vector().derive.within(Algebra("x+y+"))              # Vector+
# The colours of the background, the drop, the sunlight, light reflected once and twice inside the drop, the sky
# and the ground; the colours of the angles' axes and tick labels.
background = "#0b1020"
drop_colour = "#1e3a5f"
sunlight_colour = "#fde68a"
primary_colour = "#fb923c"
secondary_colour = "#60a5fa"
sky_colour = np.array([0.04, 0.06, 0.12])
ground_colour = "#14532d"
axis_colour, label_colour = "#475569", "#94a3b8"
# How far light is drawn outside the drop, in drop radii; the width of the angles the sky's light is
# gathered in, and the half width of the view of the sky, in degrees.
ray_length = 2.4
bin_width = 0.05
view_half_width = 62.0
# The power the sky's light is raised to for display, brightening faint light more than bright.
tone_exponent = 0.65
# The sky's width, in inches at 100 dpi, and the sweep's size, in pixels; the widths of the swept light and the
# bundle, in pixels, and of the angle curves, in points, a colour's each, thin for the dense fan of colours to stay
# apart; the sides of the drop's outline.
sky_width, sweep_size = 6.0, (600, 270)
light_width, bundle_width, curve_width = 1.5, 1.0, charts.points(1.0)
# How much of the bundle's colour shows over the background.
bundle_share = 0.4
drop_sectors = 96


# --- plumbing -------------------------------------------------------------------------
def pixel_grid(view: tuple[float, float, float, float], columns: int, rows: int
               ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """The middles of the square pixels of the view, left, right, bottom and top, that fit in `columns` by `rows`:
    across, and up with the rows down from the top; and the grid's extent, the outer edges of its pixels."""
    left, right, bottom, top = view
    pixel = max((right - left) / columns, (top - bottom) / rows)
    n_columns, n_rows = round((right - left) / pixel), round((top - bottom) / pixel)
    middle, half = np.array([left + right, bottom + top]) / 2, np.array([n_columns, n_rows]) * pixel / 2
    across = middle[0] - half[0] + pixel * (np.arange(n_columns) + 0.5)
    up = middle[1] + half[1] - pixel * (np.arange(n_rows) + 0.5)
    return across, up, (middle[0] - half[0], middle[0] + half[0], middle[1] - half[1], middle[1] + half[1])


def degrees(sight_cosines: core.Scalar) -> np.ndarray:
    """The angle from the point opposite the sun, in degrees, from its cosine."""
    return np.degrees(np.arccos(np.clip(sight_cosines.to_array(), -1, 1)))


def spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """The colour of light of each wavelength `[colours, 3]`, an approximation of the spectral colours,
    scaled so that light of every wavelength together is white."""
    red = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0.4, 0, 0, 0, 1, 1, 0.5])
    green = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0, 0, 1, 1, 1, 0, 0])
    blue = np.interp(wavelengths, [380, 440, 490, 510, 580, 645, 780], [0.6, 1, 1, 0, 0, 0, 0])
    colours = np.stack([red, green, blue], axis=-1)
    return colours / colours.sum(axis=0)


def brightness(sight_cosines: core.Scalar, flux: core.Scalar, sun_width: float) -> tuple[np.ndarray, np.ndarray]:
    """The angles from the point opposite the sun, and the light per solid angle seen at each, per
    colour `[colours, angles]`: the light gathered into narrow rings, spread over the sun's disc."""
    edges = np.arange(0, 90 + bin_width, bin_width)
    angles = (edges[1:] + edges[:-1]) / 2
    ray_angles, weights = degrees(sight_cosines), flux.to_array()            # [orders, colours, rays]
    ring_light = np.stack([np.histogram(ray_angles[:, colour].ravel(), edges, weights=weights[:, colour].ravel())[0]
                           for colour in range(ray_angles.shape[1])])         # [colours, angles]
    # Light from each point of the sun's disc arrives from a slightly different direction.
    across = np.arange(-sun_width / 2, sun_width / 2 + bin_width, bin_width)
    disc = np.sqrt(np.clip(1 - (2 * across / sun_width) ** 2, 0, None))
    disc_light = np.stack([np.convolve(ring, disc / disc.sum(), mode="same") for ring in ring_light])
    return angles, disc_light / np.sin(np.radians(angles))


def draw_sky(sight_cosines: core.Scalar, flux: core.Scalar, wavelengths: np.ndarray, sun_elevation: float,
             sun_width: float) -> plt.Figure:
    """The sky opposite the sun, with the point opposite the sun below the horizon by the sun's height."""
    angles, light = brightness(sight_cosines, flux, sun_width)
    colour = light.T @ spectrum(wavelengths)                                  # [angles, 3]
    colour = colour / colour.max()
    # The view of the sky centred above the point opposite the sun, from below the horizon, in degrees.
    view = (-view_half_width, view_half_width, sun_elevation - 6, sun_elevation + view_half_width * 0.8)
    figure = plt.figure(figsize=(sky_width, sky_width * (view[3] - view[2]) / (view[1] - view[0])), dpi=100,
                        facecolor=background)
    ax = figure.add_axes((0, 0, 1, 1))
    # Each pixel the figure shows at its angle from the point opposite the sun: the sky drawn as a chart about that
    # point, at the chart's middle, in which the angle from it is the distance from it.
    across, height, extent = pixel_grid(view, *room(ax))
    chart = mv.x.dual() * across[None, :] + mv.y.dual() * height[:, None]     # [rows, columns] Direction-
    distance = chart.norm().to_array()                                        # [rows, columns]
    image = np.stack([np.interp(distance, angles, colour[:, channel], right=0) for channel in range(3)], axis=-1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    # The ground, below the horizon at the sun's height, over the sky, the picture's own pixels read by the planes of
    # the chart.
    camera = offset_panel(ax, mv.x, mv.y)._replace(pixels=len(across))
    ground = (mv.x.dual() * np.array(extent[:2])[None, :]
              + mv.y.dual() * np.array([extent[2], sun_elevation])[:, None]).reshape(-1)   # [corners] Direction-
    seen = simplices.composited([
        contours.picture(ground, simplices.Topology.grid(2, 2), np.zeros(len(ground)), to_rgb(ground_colour), camera)])
    sky = np.clip(sky_colour + image ** tone_exponent, 0, 1)
    pinned(ax, sky * (1 - seen[..., 3:]) + seen[..., :3] * seen[..., 3:], extent)
    return figure


def animate_sweep(swept_light: tuple[core.Direction, core.Direction, core.Direction, core.Scalar],
                  bundle: tuple[core.Direction, core.Direction, core.Direction, core.Scalar],
                  offsets: core.Direction, sight_cosines: core.Scalar, wavelengths: np.ndarray) -> Iterator[np.ndarray]:
    """Light swept across the drop, with a bundle of light behind it, its paths after one and two
    reflections; beside it, the angle each colour is seen at against the offset of the light."""
    def rays(entry: core.Direction, vertices: core.Direction, exits: core.Direction) -> list[core.Direction]:
        """Per order drawn, the path of each ray from the sun to past the drop `[points, rays]`."""
        sunward = entry - sunlight * ray_length                               # [rays] Direction-
        ahead = vertices + exits * ray_length                                 # [orders + 1, rays] Direction-
        return [concatenate([stack([sunward, entry]), vertices[:order + 1], ahead[order][None]])
                for order in (1, 2)]                                          # [orders][points, rays] Direction-

    def exits(paths: core.Direction) -> tuple[core.Direction, simplices.Topology]:
        """The light leaving the drop along each path `[points, rays]`: the ends of its last stretch, and an edge
        between them."""
        n_rays = paths.shape[1]
        ends = concatenate([paths[-2], paths[-1]])                            # [2 * rays] Direction-
        return ends, simplices.Topology(2 * n_rays, np.stack([np.arange(n_rays), n_rays + np.arange(n_rays)], axis=-1))

    swept_rays, bundle_rays = rays(*swept_light[:3]), rays(*bundle[:3])
    heights = (mv.y & offsets).to_array()                                     # [rays]
    curves, colours = degrees(sight_cosines), spectrum(wavelengths) / spectrum(wavelengths).max(axis=-1, keepdims=True)
    marks, swept_heights = degrees(swept_light[3]), (mv.y & swept_light[0]).to_array()   # [orders + 1, frames], [frames]
    # The charts' style held while the frames are drawn, for the artists that drawing makes.
    with charts.style():
        # The drop, the bundle of light behind it and each colour's angles stay put; the swept light and its marks
        # move.
        figure = plt.figure(figsize=(sweep_size[0] / charts.DPI, sweep_size[1] / charts.DPI), dpi=charts.DPI,
                            facecolor=background)
        drop = figure.add_axes((0, 0, 0.5, 1))
        drop.set_facecolor(background)
        drop.set_xlim(-ray_length, ray_length)
        drop.set_ylim(-ray_length * 0.9, ray_length * 0.9)
        drop.set_aspect("equal")
        drop.set_axis_off()
        camera = offset_panel(drop, mv.x, mv.y)
        disk, coordinates = shapes.disk(1.0, 1, drop_sectors)
        places = mv(Upright, coordinates).dual()                                # [vertices] Direction-
        still = [contours.picture(places, disk, np.zeros(disk.n_vertices), to_rgb(drop_colour), camera)]
        # The bundle faint, its colours dimmed toward the background, and opaque, so that its few shades leave the
        # animation's palette to the angles' colours.
        still += [simplices.strokes(*exits(paths), bundle_share * np.array(to_rgb(colour))
                                    + (1 - bundle_share) * np.array(to_rgb(background)), bundle_width, camera)
                  for paths, colour in zip(bundle_rays, (primary_colour, secondary_colour))]
        drop_seen = simplices.overlay(drop, still)
        angles = figure.add_axes((0.57, 0.1, 0.4, 0.85))
        angles.set_facecolor(background)
        markers = []
        for order, mark in ((0, primary_colour), (1, secondary_colour)):
            for curve, colour in zip(curves[order], colours):
                angles.plot(heights, curve, color=colour, linewidth=curve_width)
            markers += angles.plot([], [], "o", color=mark)
        angles.set_xlim(0, 1)
        angles.set_ylim(0, 90)
        # Ticks only at the ends of the drop's radius and of the right angle, and at the bows: the largest angle
        # after one reflection and the smallest after two.
        angles.set_xticks([0, 1])
        angles.set_yticks([0, round(curves[0].max()), round(curves[1].min()), 90])
        angles.tick_params(colors=label_colour)
        for side in ("top", "right"):
            angles.spines[side].set_visible(False)
        for spine in angles.spines.values():
            spine.set_color(axis_colour)
        for frame in range(len(swept_heights)):
            # The swept light's paths after two reflections and after one, and the sunlight reaching the drop.
            lights = [(swept_rays[1][1:, frame], secondary_colour), (swept_rays[0][1:, frame], primary_colour),
                      (swept_rays[0][:2, frame], sunlight_colour)]
            drop_seen.set_data(simplices.composited(still + [
                simplices.strokes(path, simplices.Topology.chains([len(path)]), to_rgb(colour), light_width, camera)
                for path, colour in lights]))
            for order, marker in enumerate(markers):
                marker.set_data([swept_heights[frame]], [marks[order + 1, frame]])
            yield capture(figure)
        plt.close(figure)
