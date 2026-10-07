"""Equatorial light rays around a rotating black hole."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from examples.animation import capture
from examples.relativity.kerr import core

# The spin sweep's resolution: its thin rays need no more.
SWEEP_DPI = 64
# Wavelengths across the visible band, nm, and the second radiation constant, nm K.
WAVELENGTHS = np.linspace(380.0, 780.0, 81)
RADIATION_CONSTANT = 1.4388e7
# Linear sRGB from CIE XYZ.
XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
# The luminance of linear sRGB, and the exposure of the reference temperature.
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
MID_GREY = 0.5
# A milder lift than a display's gamma of 2.2, so brightness differences stay visible.
GAMMA = 1.6
# Blackbody colours look pale on a display; their saturation is exaggerated to show the shift.
SATURATION = 2.5


# --- plumbing -------------------------------------------------------------------------
def draw_rays(paths: core.Vector, horizons: core.Vector,
              names: tuple[str, ...], extent: float) -> plt.Figure:
    """Compare spins with identical ray colours and spatial scales."""
    tracks = paths.cast(core.ga.subspace("x y")).kernel.transpose(1, 2, 0, 3)
    outlines = horizons.cast(core.ga.subspace("x y")).kernel
    colours = plt.colormaps["coolwarm"](np.linspace(0.05, 0.95, tracks.shape[1]))
    ticks = [-extent / 2, 0, extent / 2]

    figure, panels = plt.subplots(1, len(names), figsize=(4 * len(names), 4),
                                 sharex=True, sharey=True, squeeze=False,
                                 layout="constrained")
    for ax, rays, horizon, name in zip(panels[0], tracks, outlines, names):
        ax.add_collection(LineCollection(rays, colors=colours, linewidths=1.0, alpha=0.9))
        ax.fill(*horizon.T, color="#161b22", zorder=3)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
               xticks=ticks, yticks=ticks, xlabel="$x/M$", title=name)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#cbd5e1")
        ax.tick_params(color="#cbd5e1")
    panels[0, 0].set_ylabel("$y/M$")
    return figure


def animate_rays(paths: core.Vector, horizons: core.Vector,
                 extent: float) -> list[np.ndarray]:
    """Sweep stationary spins with fixed ray colours and spatial scales."""
    tracks = paths.cast(core.ga.subspace("x y")).kernel.transpose(1, 2, 0, 3)
    outlines = horizons.cast(core.ga.subspace("x y")).kernel
    colours = plt.colormaps["coolwarm"](np.linspace(0.05, 0.95, tracks.shape[1]))
    ticks = [-extent / 2, 0, extent / 2]

    figure, ax = plt.subplots(figsize=(5, 5), dpi=SWEEP_DPI, layout="constrained")
    rays = LineCollection(tracks[0], colors=colours, linewidths=1.0, alpha=0.9)
    horizon = Polygon(outlines[0], facecolor="#161b22", edgecolor="none", zorder=3)
    ax.add_collection(rays)
    ax.add_patch(horizon)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
           xticks=ticks, yticks=ticks, xlabel="$x/M$", ylabel="$y/M$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#cbd5e1")
    ax.tick_params(color="#cbd5e1")
    figure.canvas.draw()
    figure.set_layout_engine("none")

    frames = []
    for paths, outline in zip(tracks, outlines):
        rays.set_segments(paths)
        horizon.set_xy(outline)
        frames.append(capture(figure))
    plt.close(figure)
    return frames


def inline(frames: list[np.ndarray], duration_ms: int):
    """Frames as a looping GIF to show in a notebook, kept in memory."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image
    images = [Image.fromarray(pixels) for pixels in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")


def star_texture(width: int, height: int, seed: int) -> np.ndarray:
    """A seamless sky image with unresolved stars and a faint dusty stellar band."""
    random = np.random.default_rng(seed)
    star_count = width * height // 140
    longitude = (np.arange(width) + 0.5) * (2 * np.pi / width) - np.pi
    latitude = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    horizontal = np.cos(latitude[:, None])
    sky_x = horizontal * np.cos(longitude)
    sky_y = horizontal * np.sin(longitude)
    sky_z = np.sin(latitude[:, None])
    band_height = 0.2 * sky_x - 0.6 * sky_y - np.sqrt(0.6) * sky_z
    cloud = (np.sin(13 * sky_x + 9 * sky_y - 7 * sky_z)
             + 0.45 * np.sin(31 * sky_x - 23 * sky_y + 19 * sky_z)
             + 0.18 * np.sin(79 * sky_x + 47 * sky_y + 53 * sky_z))
    band = np.exp(-(band_height / 0.16) ** 2) * (0.75 + 0.2 * cloud)
    dust = np.exp(-((band_height + 0.025 * cloud) / 0.035) ** 2)
    background = np.array([0.006, 0.009, 0.019])
    band_colour = np.array([0.065, 0.058, 0.09])
    texture = background + band[..., None] * band_colour * (1 - 0.7 * dust[..., None])

    # Distribute a catalogue on the sphere, then stamp small Gaussian stellar images.
    star_x = random.uniform(0, width, star_count)
    star_latitude = np.arcsin(random.uniform(-1, 1, star_count))
    star_y = (0.5 - star_latitude / np.pi) * height
    brightness = 0.35 + 2.5 * random.random(star_count) ** 6
    warmth = random.uniform(0, 1, (star_count, 1))
    colours = np.array([0.57, 0.72, 1.0]) * (1 - warmth) + np.array([1.0, 0.79, 0.51]) * warmth
    offsets = np.arange(-2, 3)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    columns = np.floor(star_x).astype(int)[:, None] + offset_x.ravel()
    rows = np.floor(star_y).astype(int)[:, None] + offset_y.ravel()
    distance_squared = (columns + 0.5 - star_x[:, None]) ** 2 + (rows + 0.5 - star_y[:, None]) ** 2
    weights = np.exp(-distance_squared / (2 * 0.6 ** 2)) * brightness[:, None]
    valid = (rows >= 0) & (rows < height)
    contribution = weights[..., None] * colours[:, None]
    np.add.at(texture, (rows[valid], columns[valid] % width), contribution[valid])
    return np.clip(texture, 0, 1)


def sky_colour(direction: core.Vector, texture: np.ndarray) -> np.ndarray:
    """Look up the celestial image using the escaped ray's spatial direction."""
    horizontal_x, horizontal_y, vertical = np.moveaxis(
        direction.cast(core.ga.subspace("x y z")).kernel, -1, 0)
    length = np.sqrt(horizontal_x ** 2 + horizontal_y ** 2 + vertical ** 2)
    height, width = texture.shape[:2]
    longitude = np.arctan2(horizontal_y, horizontal_x)
    latitude = np.arcsin(np.clip(vertical / length, -1, 1))
    column = (longitude / (2 * np.pi) + 0.5) * width - 0.5
    row = np.clip((0.5 - latitude / np.pi) * height - 0.5, 0, height - 1)
    left = np.floor(column).astype(int)
    top = np.floor(row).astype(int)
    right = (left + 1) % width
    bottom = np.minimum(top + 1, height - 1)
    fraction_x = (column - left)[..., None]
    fraction_y = (row - top)[..., None]
    upper = texture[top, left % width] * (1 - fraction_x) + texture[top, right] * fraction_x
    lower = texture[bottom, left % width] * (1 - fraction_x) + texture[bottom, right] * fraction_x
    return upper * (1 - fraction_y) + lower * fraction_y


def disk_emissivity(points: core.Vector, radius: core.Scalar,
                    inner_radius: float, outer_radius: float) -> np.ndarray:
    """An illustrative orbital texture on the disk's emission, tapered at its outer edge, `[pixels]`."""
    horizontal_x, horizontal_y = np.moveaxis(points.cast(core.ga.subspace("x y")).kernel, -1, 0)
    radius = radius.kernel[..., 0]
    angle = np.arctan2(horizontal_y, horizontal_x)
    fraction = np.clip((radius - inner_radius) / (outer_radius - inner_radius), 0, 1)
    rings = 0.90 + 0.045 * np.sin(11 * radius + 2 * np.sin(3 * angle)) + 0.025 * np.sin(19 * radius - 5 * angle)
    filaments = 0.96 + 0.04 * np.sin(9 * angle - 7 * radius)
    # A tapered outer edge keeps the source finite without painting a sharp bright rim.
    taper = np.clip((1 - fraction) / 0.12, 0, 1)
    return rings * filaments * taper


def colour_matching(wavelengths: np.ndarray) -> np.ndarray:
    """The CIE 1931 colour-matching functions, by the multi-lobe fits of Wyman, Sloan and Shirley
    (2013), `[wavelengths, 3]`."""
    def lobe(centre: float, below: float, above: float) -> np.ndarray:
        width = np.where(wavelengths < centre, below, above)
        return np.exp(-0.5 * ((wavelengths - centre) / width) ** 2)
    x = 1.056 * lobe(599.8, 37.9, 31.0) + 0.362 * lobe(442.0, 16.0, 26.7) - 0.065 * lobe(501.1, 20.4, 26.2)
    y = 0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1)
    z = 1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def blackbody(temperature: np.ndarray) -> np.ndarray:
    """The visible light of a blackbody at each temperature, K, in linear sRGB, relative units,
    `[..., 3]`; a temperature of zero is dark."""
    with np.errstate(divide="ignore", over="ignore"):
        spectrum = 1 / (WAVELENGTHS ** 5 * np.expm1(RADIATION_CONSTANT / (WAVELENGTHS * temperature[..., None])))
    return np.clip(spectrum @ colour_matching(WAVELENGTHS) @ XYZ_TO_RGB.T, 0, None)


def exposed(light: np.ndarray, reference: float) -> np.ndarray:
    """Linear light shown on the display: the luminance scaled so a blackbody at the reference
    temperature sits at mid-grey, kept linear below it and rolled off smoothly above, the colour kept,
    then gamma-encoded and its saturation exaggerated, any channel above one scaled back."""
    luminance = light @ LUMINANCE                                                   # [...]
    exposure = luminance / (blackbody(np.array(reference)) @ LUMINANCE)
    mapped = np.where(exposure < 1, MID_GREY * exposure, 1 - (1 - MID_GREY) * np.exp(-(exposure - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(luminance > 0, mapped / luminance, 0)
    shown = (light * scale[..., None]) ** (1 / GAMMA)
    grey = shown @ LUMINANCE
    vivid = np.clip(grey[..., None] + SATURATION * (shown - grey[..., None]), 0, None)
    return vivid / np.maximum(vivid.max(axis=-1, keepdims=True), 1)


def camera_colours(image: core.Image, temperature: core.Scalar, image_shape: tuple[int, int],
                   sky_texture: np.ndarray, inner_radius: float, outer_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """The disk's linear light, a blackbody at each pixel's observed temperature, and the sky's
    colours, each scattered to its pixels, `[pixels, 3]` each."""
    height, width = image_shape
    disk = np.zeros((height * width, 3))
    sky = np.zeros((height * width, 3))
    emissivity = disk_emissivity(image.disk[0], image.disk_radii, inner_radius, outer_radius)
    disk[image.disk_pixels] = blackbody(temperature.kernel[..., 0]) * emissivity[:, None]
    sky[image.sky_pixels] = sky_colour(image.sky_directions, sky_texture)
    return disk, sky


def draw_camera(image: core.Image, temperature: core.Scalar, hottest: float, image_shape: tuple[int, int],
                sky_texture: np.ndarray, inner_radius: float, outer_radius: float) -> plt.Figure:
    """The camera's view: the opaque glowing disk at its observed temperatures, exposed for its
    hottest emitted one, the horizon's shadow and the lensed sky."""
    height, width = image_shape
    disk, sky = camera_colours(image, temperature, image_shape, sky_texture, inner_radius, outer_radius)
    figure = plt.figure(figsize=(9, 9 * height / width), facecolor="black")
    ax = figure.add_axes((0, 0, 1, 1))
    ax.imshow((exposed(disk, hottest) + sky).reshape(height, width, 3), interpolation="lanczos")
    ax.set_axis_off()
    return figure


def animate_polarizer(image: core.Image, temperature: core.Scalar, hottest: float, transmission: core.Scalar,
                      image_shape: tuple[int, int], sky_texture: np.ndarray, inner_radius: float,
                      outer_radius: float) -> list[np.ndarray]:
    """The camera's view through each analyzer: the disk by the power it transmits, `transmission
    [analyzers, disk pixels]`. The exposure is doubled to make up for the half of unpolarized light a
    polarizer stops, so the sky and unpolarized light look as in the still."""
    height, width = image_shape
    disk, sky = camera_colours(image, temperature, image_shape, sky_texture, inner_radius, outer_radius)
    power = np.zeros((transmission.shape[0], height * width))
    power[:, image.disk_pixels] = transmission.kernel[..., 0]
    return [np.round(np.clip(exposed(2 * disk * passed[:, None], hottest) + sky, 0, 1) * 255).astype(np.uint8).reshape(height, width, 3)
            for passed in power]
